"""LangGraph 状态定义:多 Agent 协作图共享的状态对象。

多轮记忆设计:
- messages: 通过 LangGraph 的 add_messages reducer 自动累积每轮对话历史
  (HumanMessage + AIMessage 交替追加),配合 checkpointer 按 thread_id 持久化
- awaiting_booking: 业务 Agent 追问预约信息时置 True,下一轮无强意图时自动延续预约流程
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """多 Agent 图的共享状态。

    字段说明:
    - messages: 多轮对话消息历史(自动追加,由 checkpointer 按 thread_id 保存)
    - user_input: 当前轮用户消息(兼容旧调用;优先从 messages 取)
    - session_id: 会话标识(用于跨轮关联 / thread_id)
    - tenant_id: 租户/行业 id(知识隔离维度)
    - intent: 主管 Agent 路由出的意图:consult / business / handoff
    - intent_reason: 路由理由(便于调试与日志)
    - awaiting_booking: 是否正在等待用户补齐预约信息
    - retrieved_docs: 咨询 Agent 检索到的知识条目
    - booking: 业务 Agent 解析出的预约信息(跨轮累积)
    - reply: 最终回复文本
    - sources: 回答依据的知识来源
    - handoff_reason: 转人工原因
    - profile: 画像 Agent 返回的用户画像摘要
    - user_key: 用户标识(跨会话长期记忆的隔离维度;缺省退化为 session_id)
    - summary: 早期对话的压缩摘要(上下文工程:超出窗口后由旧历史折叠生成)
    - context: 主管注入的辅助上下文(长期记忆召回结果,不污染 messages 历史)
    """

    messages: Annotated[list[Any], add_messages]
    user_input: str
    session_id: str
    tenant_id: int
    intent: str
    intent_reason: str
    awaiting_booking: bool
    retrieved_docs: list[dict[str, Any]]
    booking: dict[str, Any]
    reply: str
    sources: list[str]
    handoff_reason: str
    profile: dict[str, Any]
    user_key: str
    summary: str
    context: str


# ---------------------------------------------------------------------------
# 消息工具函数(兼容 BaseMessage 对象与 dict 两种形式)
# ---------------------------------------------------------------------------
def _msg_type(m: Any) -> str:
    if isinstance(m, dict):
        return m.get("type", "")
    return getattr(m, "type", "")


def _msg_content(m: Any) -> str:
    if isinstance(m, dict):
        return str(m.get("content", ""))
    return str(getattr(m, "content", ""))


def current_user_input(state: AgentState) -> str:
    """取当前轮用户输入:优先 messages 最后一条 human,兼容旧 user_input 字段。"""
    messages = state.get("messages") or []
    for m in reversed(messages):
        if _msg_type(m) == "human":
            return _msg_content(m)
    return state.get("user_input", "")


def history_user_texts(messages: list[Any]) -> list[str]:
    """当前轮之前的用户消息文本(不含最后一条 human),用于跨轮补全上下文。"""
    texts = [_msg_content(m) for m in messages if _msg_type(m) == "human"]
    return texts[:-1] if len(texts) > 1 else []
