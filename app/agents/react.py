"""ReAct 自主决策节点:让 LLM 自己决定「下一步调哪个工具」。

与 business_node(规则式流程)的本质区别:
  - business_node: 开发者用 if-else 写死「检测关键词→调预约函数」,LLM 不参与决策;
  - react_node:   把「工具清单 + 当前对话」交给 LLM,LLM 自主生成 tool_calls,
                  执行完把结果喂回去,LLM 再看结果决定继续调工具还是给最终答复。

这是 "Agent" 区别于 "流程图/接口" 的分水岭。

降级策略(本项目"无 key 不伪装"原则):
  - 配置了真实 LLM key → 走真正的 ReAct 循环(需要模型支持 bind_tools);
  - 未配置 key / key 是占位符 → 委托 business_node(规则流程),行为与旧版一致;
  - awaiting_booking=True(正在跨轮补全预约字段)时同样走规则流程,状态机更稳。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.agents.state import AgentState, current_user_input
from app.config.settings import settings
from app.llm.provider import create_chat_model
from app.mcp.tools import (
    cancel_booking,
    create_booking,
    get_weather,
    query_booking,
)

# ReAct 循环上限:防止 LLM 陷入无限调工具
MAX_TOOL_STEPS = 5


def _create_booking_bound(tenant_id: int, session_id: str):
    """把 tenant/session 固化进工具签名(用普通函数包装,不用 partial)。

    注意:langchain 的 StructuredTool.from_function 需要"模块/类/方法/函数",
    直接传 functools.partial 会抛 TypeError,所以这里用闭包生成普通函数,
    暴露给 LLM 的参数只有业务字段(服务/时间/姓名/备注)。
    """

    def _create_booking(
        service_type: str,
        appointment_time: str,
        customer_name: str = "",
        notes: str = "",
    ) -> dict:
        """创建预约工单。

        Args:
            service_type: 服务类型(如 全身推拿)
            appointment_time: 预约时间(自然语言或具体时间)
            customer_name: 客户姓名(可选)
            notes: 备注(可选)
        """
        return create_booking(
            tenant_id=tenant_id,
            session_id=session_id,
            service_type=service_type,
            appointment_time=appointment_time,
            customer_name=customer_name,
            notes=notes,
        )

    return _create_booking


def _bind_booking_tools(tenant_id: int, session_id: str) -> list[StructuredTool]:
    """把预约类工具绑定上下文(tenant/session),供 LLM 直接调用。"""
    return [
        StructuredTool.from_function(
            func=_create_booking_bound(tenant_id, session_id),
            name="create_booking",
            description="创建预约工单(预约服务)。调用前请确认:服务类型、预约时间;客户姓名/备注可选。",
        ),
        StructuredTool.from_function(
            query_booking,
            name="query_booking",
            description="查询预约工单状态。需要工单号(数字部分)。",
        ),
        StructuredTool.from_function(
            cancel_booking,
            name="cancel_booking",
            description="取消预约工单。需要工单号(数字部分)。",
        ),
    ]


def _execute_tool(tool: StructuredTool, args: dict) -> tuple[str, Any]:
    """执行一个工具,返回 (给 LLM 看的文本, 原始结果);异常也转文本,避免中断循环。"""
    try:
        result = tool.invoke(args)
        return json.dumps(result, ensure_ascii=False), result
    except Exception as exc:  # noqa: BLE001
        return f"工具调用失败: {exc}", None


def _run_react_loop(
    llm: Any,
    tools: list[StructuredTool],
    sys_prompt: str,
    user_input: str,
    history_text: str,
    on_tool_result: Any = None,
) -> str:
    """真正的 ReAct 循环:LLM 自主决策 → 执行工具 → 看结果 → 继续/收尾。"""
    # 只用最近的对话(避免无限膨胀;真正的压缩策略见 context.py)
    recent = history_text[-4000:] if history_text else ""
    messages = [SystemMessage(content=sys_prompt)]
    # 历史按文本放回,让 LLM 感知多轮语境
    if recent:
        messages.append(SystemMessage(content=f"【本会话此前对话(参考)】\n{recent}"))

    llm_with_tools = llm.bind_tools(tools)
    messages.append(HumanMessage(content=user_input))

    for _ in range(MAX_TOOL_STEPS):
        resp = llm_with_tools.invoke(messages)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            # LLM 不再想调工具 → 这就是最终答复
            return (resp.content or "").strip()

        # 逐个执行 LLM 请求的工具,结果以 ToolMessage 回填
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            tool = next((t for t in tools if t.name == name), None)
            if tool is None:
                result_text = f"未知工具: {name}"
            else:
                result_text, raw = _execute_tool(tool, args)
                if on_tool_result is not None and raw is not None:
                    on_tool_result(name, args, raw)
            messages.append(
                ToolMessage(content=result_text, tool_call_id=call.get("id", ""))
            )

    # 达到步数上限仍无最终答复 → 兜底
    return "抱歉,我尝试了多个步骤仍未完成您的请求,建议转人工处理。"


def react_node(state: AgentState) -> AgentState:
    """业务意图的自主决策节点:有 LLM key 且非跨轮补全时走 ReAct,否则降级规则流程。

    降级场景(保持无 key / 预约补全中间轮行为与旧版完全一致):
      - 未配置 API key(或仅为 your_xxx 占位符);
      - awaiting_booking=True(主管正引导用户逐项补齐预约信息)——这种"一问一答
        收字段"的交互,规则式状态机更稳,LLM 自主决策反而容易跑偏。
    """
    from app.agents.business import business_node

    if not settings.has_llm_key or state.get("awaiting_booking"):
        return business_node(state)

    user_input = current_user_input(state)
    tenant_id = state.get("tenant_id", 1)
    session_id = state.get("session_id", "default")
    # 历史/记忆辅助上下文
    history_text = ""
    for m in state.get("messages") or []:
        if getattr(m, "type", "") == "human":
            history_text += f"用户: {m.content}\n"
        elif getattr(m, "type", "") == "ai":
            history_text += f"助手: {m.content}\n"
    long_term = state.get("context") or ""
    summary = state.get("summary") or ""

    sys_prompt = (
        "你是智能客服的业务 Agent,负责预约、查询、取消预约等操作。\n"
        "规则:必须用提供的工具完成操作,不要凭空编造工单号。流程建议:\n"
        "1. 用户想预约 → 调 create_booking(先向用户确认服务类型与时间);\n"
        "2. 用户想查/改/取消预约 → 先 query_booking 查单(没有工单号就问用户,"
        "或从历史/记忆中推断),再决定是否 cancel_booking;\n"
        "3. 预约成功后,回复要友好地复述服务、时间与工单号。\n"
        "4. 需要多次调用工具才能完成的请求(如先查再取消),请分步调用,直到完成。\n"
        "完成后,用中文给用户一段自然的最终答复(不要展示内部调用过程)。"
    )
    if long_term:
        sys_prompt += f"\n\n{long_term}"
    if summary:
        sys_prompt += f"\n\n【更早对话摘要】\n{summary}"

    tools = [
        *_bind_booking_tools(tenant_id, session_id),
        StructuredTool.from_function(
            get_weather,
            name="get_weather",
            description="查询城市天气(演示第三方工具)。",
        ),
    ]

    # MCP 工具优先:知识检索类工具走真实 MCP 协议(需要先启动 scripts/start_mcp_server.py),
    # 服务不可用时 get_mcp_tools() 返回空列表,自动退回本地工具集。
    # 预约类工具仍用本地绑定版本:它们需要固化 tenant_id/session_id,
    # 而 MCP 版的 create_booking 签名要求 LLM 自己填这两个参数,容易填错。
    from app.mcp.client import get_mcp_tools
    from app.observability.trace import record_stage

    mcp_tools = get_mcp_tools()
    if mcp_tools:
        local_names = {t.name for t in tools}
        tools.extend([t for t in mcp_tools if t.name not in local_names])
    record_stage(
        "tool_source",
        source="mcp+local" if mcp_tools else "local",
        local_tools=len(_bind_booking_tools(tenant_id, session_id)) + 1,
        mcp_tools=len(mcp_tools),
        names=[t.name for t in tools],
    )

    user_key = state.get("user_key") or session_id

    def _remember_booking(tool_name: str, args: dict, result: dict) -> None:
        """ReAct 路径下预约成功也要写长期记忆(与 business_node 的口径一致)。"""
        if tool_name != "create_booking" or "ticket_id" not in result:
            return
        try:
            from app.db.database import SessionLocal
            from app.services.memory_service import remember

            db = SessionLocal()
            try:
                remember(
                    db,
                    tenant_id,
                    user_key,
                    session_id,
                    (
                        f"用户预约了「{args.get('service_type') or '服务'}」"
                        f",时间为{args.get('appointment_time') or '待定'}"
                        + (
                            f",姓名{args.get('customer_name')}"
                            if args.get("customer_name")
                            else ""
                        )
                        + f"(工单 T-{int(result['ticket_id']):04d})"
                    ),
                    memory_type="appointment",
                    importance=8,
                )
            finally:
                db.close()
        except Exception:  # noqa: BLE001 - 记忆失败不影响业务
            pass

    llm = create_chat_model(temperature=0)
    try:
        reply = _run_react_loop(
            llm,
            tools,
            sys_prompt,
            user_input,
            history_text,
            on_tool_result=_remember_booking,
        )
    except Exception as exc:  # noqa: BLE001 - LLM/网络失败不致命
        # 模型不支持 function calling / 网络异常 → 回退规则流程,保证业务可完成
        fallback = business_node(dict(state))
        reply = fallback.get("reply") or f"业务处理遇到问题({exc}),建议转人工协助。"
        state.update({k: v for k, v in fallback.items() if k != "messages"})
        state["reply"] = reply
        return state

    state["reply"] = reply
    state["awaiting_booking"] = False
    from langchain_core.messages import AIMessage

    state["messages"] = [AIMessage(content=reply)]
    return state
