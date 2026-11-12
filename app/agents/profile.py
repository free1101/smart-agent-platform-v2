"""用户画像 Agent:基于用户行为记录回答个性化问题(我常做的项目/偏好/回访)。

对应原项目 user_behavior 的升级:行为记录在业务节点完成(事件溯源),
本节点只负责「读画像 → 生成个性化回答」,职责单一。

回答策略:把画像摘要 + 最近行为注入提示词,由真实 LLM 生成个性化回答;
无画像记录时如实说明并引导预约/咨询(不伪造个性化内容)。

回访场景:命中「好久没来/回访」等词,且距最近预约超过阈值 → 生成回访提醒。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.state import AgentState, current_user_input
from app.llm.provider import create_chat_model
from app.services.profile_service import (
    build_personalized_context,
    get_profile,
    list_behaviors,
)

# 回访相关关键词(在画像意图内触发回访话术)
_RETURN_KEYWORDS = ["回访", "好久没来", "好久不见", "推荐我", "给我推荐", "最近有优惠", "老客户"]


def profile_node(state: AgentState) -> AgentState:
    """画像节点:读画像 → 生成个性化回答。"""
    messages = state.get("messages") or []
    user_input = current_user_input(state)
    tenant_id = state.get("tenant_id", 1)
    session_id = state.get("session_id", "")

    # ---- 读取画像与行为历史 ----
    if tenant_id and session_id:
        from app.db.database import SessionLocal

        db = SessionLocal()
        try:
            profile = get_profile(db, tenant_id, session_id)
            behaviors = list_behaviors(db, tenant_id, session_id, limit=5)
        finally:
            db.close()
    else:
        profile = {
            "session_id": session_id,
            "favorite_service": "",
            "favorite_time_period": "",
            "favorite_duration": "",
            "total_appointments": 0,
            "last_appointment_at": "",
            "has_profile": False,
        }
        behaviors = []

    is_return = any(kw in user_input for kw in _RETURN_KEYWORDS)

    # ---- 生成回答 ----
    reply = ""
    if not profile["has_profile"]:
        # 无画像记录:如实引导,不编造个性化内容
        reply = (
            "我这边暂时还没有您的服务记录。您可以先告诉我您想预约或咨询的服务,"
            "之后我就能为您提供更个性化的建议啦。"
        )
    else:
        try:
            llm = create_chat_model(temperature=0.3)
            profile_text = build_personalized_context(None, tenant_id, session_id, profile=profile)
            history_text = "\n".join(
                f"- {b['action_type']}: {b['action_data']}" for b in behaviors
            ) or "(无)"
            prompt = (
                f"你是门店的贴心客服。请基于用户画像友好地回应用户,语气亲切、自然。\n\n"
                f"{profile_text}\n最近行为:\n{history_text}\n\n"
                f"{'用户希望被召回/推荐,请结合其偏好给出个性化推荐或回访邀约。' if is_return else '用户询问自己的画像/偏好/历史。'}\n"
                f"用户消息:{user_input}\n"
                "请用不超过 3 句话回答。"
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            reply = str(resp.content).strip()
        except Exception:
            reply = "抱歉,暂时无法生成个性化回答,请稍后再试,或直接告诉我您想预约或咨询的服务。"

    state["reply"] = reply
    state["profile"] = profile
    state["messages"] = messages + [AIMessage(content=reply)]
    return state
