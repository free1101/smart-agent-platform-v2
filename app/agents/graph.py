"""LangGraph 多 Agent 图:主管路由 + 咨询/画像/业务(ReAct 自主决策)/转人工。

多轮记忆:图编译时挂持久化 checkpointer(SqliteSaver,见 _default_checkpointer),
按 thread_id 保存状态(含 messages 消息历史与 booking 预约草稿),同一会话的
下一轮自动续上上下文;重启服务后历史仍在磁盘(未装依赖时自动降级 MemorySaver)。

扩展能力:
  - context.py         上下文工程: 超长历史折叠为摘要,控制进入 LLM 的 token;
  - memory_service.py  长期记忆: 按 user_key 跨会话召回用户事实;
  - react.py           ReAct 自主决策: 有真实 LLM key 时,业务意图由 LLM 自主调工具
                       (预约/查询/取消/天气),无 key 或跨轮补全时降级规则流程。

图结构:
                 ┌──────────────┐
   user_input → │  supervisor   │  (意图路由,支持预约流程延续)
                 └──────┬───────┘
       ┌─────────┬──────┴──────┬─────────┐
       ▼         ▼             ▼         ▼
  ┌─────────┐┌─────────┐  ┌─────────┐┌─────────┐
  │consultant││ profile  │  │  react   ││ support  │
  │(RAG咨询) ││(用户画像)│  │(业务 ReAct)││(转人工)  │
  └─────────┘└─────────┘  └─────────┘└─────────┘
       └─────────────┬───────────┴──────┘
                     ▼
                END(返回 reply)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.consultant import consultant_node
from app.agents.context import build_summary, should_compress
from app.agents.intent_router import (
    INTENT_BUSINESS,
    INTENT_CONSULT,
    INTENT_HANDOFF,
    INTENT_PROFILE,
    intent_router,
    is_explicit_handoff,
    looks_like_slot_answer,
)
from app.agents.profile import profile_node
from app.agents.react import react_node
from app.agents.state import AgentState, current_user_input
from app.agents.support import support_node
from app.services.memory_service import format_recall, recall


def _recent_dialogue(state: AgentState, limit: int = 4) -> str:
    """取当前消息之前的最近几轮对话文本(供意图分类器参考上下文)。"""
    messages = state.get("messages") or []
    lines: list[str] = []
    for m in messages[-(limit + 1) : -1]:
        if isinstance(m, dict):
            mtype, content = m.get("type", ""), str(m.get("content", ""))
        else:
            mtype, content = getattr(m, "type", ""), str(getattr(m, "content", ""))
        if mtype == "human":
            lines.append(f"用户: {content[:120]}")
        elif mtype == "ai":
            lines.append(f"助手: {content[:120]}")
    return "\n".join(lines)


def supervisor_node(state: AgentState) -> AgentState:
    """主管节点:意图路由(含多轮记忆的预约流程延续)。

    另外做两件"上下文侧"的事:
      1. 上下文工程:对话超过窗口后,把旧历史折叠成摘要(减少进入 LLM 的 token);
      2. 长期记忆:新会话首轮按 user_key 召回历史事实,注入 state["context"]。
    """
    user_input = current_user_input(state)
    prev_intent = state.get("intent")
    # 把最近几轮对话 + 上一轮意图一并交给分类器
    # (否则「王先生」这类补充句在 LLM 眼里是孤立短语,容易被判成 handoff)
    intent, reason = intent_router.classify(
        user_input,
        context=_recent_dialogue(state),
        prev_intent=prev_intent,
    )

    # 多轮记忆:满足任一条件即视为"继续补全预约",即使 LLM 判成 consult/handoff:
    #   1) 上一轮业务节点已标记 awaiting_booking(规则流程/降级模式);
    #   2) 上一轮 intent 就是 business,且本轮是很短的补充句
    #      (ReAct 路径不落 awaiting 标志,只能靠这条兜底)。
    # 唯一例外:用户明确要求转人工/投诉(is_explicit_handoff),此时尊重其意图。
    if not is_explicit_handoff(user_input) and (
        state.get("awaiting_booking")
        or (prev_intent == INTENT_BUSINESS and looks_like_slot_answer(user_input))
    ):
        intent = INTENT_BUSINESS
        reason = "延续上一轮预约流程:本轮为补充信息(姓名/时间等)"

    # 上下文工程:超窗后生成摘要(只生成一次,后续轮次复用)
    messages = state.get("messages") or []
    if should_compress(messages) and not state.get("summary"):
        state["summary"] = build_summary(messages)

    # 长期记忆(跨会话):仅新会话首轮召回,把"还记得的用户事实"注入辅助上下文。
    # 判断依据:当前 messages 里只有本轮一条 human(此前没有对话历史)→ 新会话开始。
    if not state.get("context") and len(
        [m for m in messages if getattr(m, "type", "") == "human"]
    ) <= 1:
        try:
            from app.db.database import SessionLocal

            db = SessionLocal()
            try:
                mems = recall(
                    db,
                    state.get("tenant_id", 1),
                    state.get("user_key") or state.get("session_id", ""),
                    query=user_input,
                    limit=5,
                )
                ctx = format_recall(mems)
                if ctx:
                    state["context"] = ctx
            finally:
                db.close()
        except Exception:  # noqa: BLE001 - 记忆召回失败不影响主流程
            pass

    state["user_input"] = user_input
    state["intent"] = intent
    state["intent_reason"] = reason
    if intent == INTENT_HANDOFF:
        state["handoff_reason"] = reason
    return state


def route_after_supervisor(state: AgentState) -> str:
    """条件边:根据意图路由到对应 Agent。"""
    return state.get("intent", INTENT_CONSULT)


def _default_checkpointer():
    """默认状态检查点:优先 SqliteSaver(磁盘持久化,重启不丢),无依赖时降级 MemorySaver。

    需安装 langgraph-checkpoint-sqlite(见 requirements.txt);未安装则回退内存模式,
    保证项目"零额外依赖也能跑"的一贯原则。
    """
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

        from app.config.settings import settings

        # 直接持有长连接:langgraph-checkpoint-sqlite>=3.x 的 from_conn_string 是
        # @contextmanager(with 退出即关闭连接),不适合模块级单例场景。
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        db_path = str(settings.DATA_DIR / "langgraph_state.sqlite")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:  # noqa: BLE001 - 无 sqlite checkpointer 依赖时降级内存模式
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


def build_agent_graph(checkpointer=None):
    """构建并编译多 Agent 图(可多次调用,返回可复用的 CompiledGraph)。

    多轮记忆:不传 checkpointer 时默认挂持久化 checkpointer(SqliteSaver,
    磁盘持久化,重启后同 thread_id 会话历史仍在;未安装依赖则降级 MemorySaver)。
    单测请显式传 MemorySaver(),避免固定 thread_id 跨运行残留历史。
    """
    if checkpointer is None:
        checkpointer = _default_checkpointer()

    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("consultant", consultant_node)
    graph.add_node("profile", profile_node)
    graph.add_node("react", react_node)
    graph.add_node("support", support_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            INTENT_CONSULT: "consultant",
            INTENT_PROFILE: "profile",
            INTENT_BUSINESS: "react",  # 业务意图 → ReAct(内部按 key/流程状态自动降级规则)
            INTENT_HANDOFF: "support",
        },
    )
    graph.add_edge("consultant", END)
    graph.add_edge("profile", END)
    graph.add_edge("react", END)
    graph.add_edge("support", END)

    return graph.compile(checkpointer=checkpointer)


# 模块级编译一次,供 API/脚本复用(默认 SqliteSaver,同一会话跨轮 + 跨重启记忆)
agent_graph = build_agent_graph()
