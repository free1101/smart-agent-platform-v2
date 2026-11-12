"""转人工 Agent:处理超出客服范围/需要人工介入的问题,优雅兜底。"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agents.state import AgentState


def support_node(state: AgentState) -> AgentState:
    """转人工节点:给出人工渠道,记录原因。"""
    reason = state.get("handoff_reason") or "超出客服能力范围"

    state["reply"] = (
        "这个问题需要人工客服协助,已为您记录并转接人工。\n"
        "人工客服工作时间:每天 9:00-21:00,电话 400-800-1234,"
        "或留下联系方式,我们会尽快回电。"
    )
    state["handoff_reason"] = reason
    state["sources"] = ["人工客服"]
    state["messages"] = [AIMessage(content=state["reply"])]
    return state
