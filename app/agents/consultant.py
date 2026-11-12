"""咨询 Agent:面向知识咨询类问题。

多轮记忆能力:把最近的历史对话注入提示词,支持指代/省略式追问
(例如用户先问「推拿多少钱」,再问「那足底呢」——结合历史才能答得对)。

流程:混合检索 RAG 召回 → 组装上下文(含历史)→ LLM 生成回答。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.context import prepare_history, recent_history_texts
from app.agents.state import AgentState, current_user_input
from app.llm.provider import create_chat_model
from app.services.knowledge_service import knowledge_service


def build_consult_prompt(
    user_input: str,
    docs: list[dict],
    history: list[str] | None = None,
    summary: str | None = None,
    long_term_context: str | None = None,
) -> str:
    """组装咨询提示词:检索知识 + 历史对话(超长时用摘要替代最旧部分)+ 跨会话长期记忆。"""
    hist_block = ""
    if long_term_context:
        hist_block = long_term_context + "\n\n"
    if summary:
        hist_block += "【更早对话摘要(简要回顾)】\n" + summary + "\n\n"
    if history:
        hist_block += "【近期对话(供参考,回答以当前问题为准)】\n" + "\n".join(
            f"用户: {h}" for h in history
        ) + "\n\n"

    if not docs:
        return (
            "你是智能客服咨询助手。用户的问题在知识库中没有找到直接答案。\n"
            "请礼貌告知用户,并引导其咨询其他内容或转人工。\n\n"
            f"{hist_block}"
            f"用户问题:{user_input}"
        )

    context = "\n\n".join(
        f"[知识 {i + 1} | 分类:{d.get('category', '')}]\n{d.get('content', '')}"
        for i, d in enumerate(docs)
    )
    return (
        "你是智能客服咨询助手。请仅依据下面提供的知识库内容回答用户问题。\n"
        "如果知识不足以回答,请如实说明,不要编造。回答要简洁、友好、口语化。\n\n"
        f"{hist_block}"
        "【检索知识】\n"
        f"{context}\n"
        "【END】\n\n"
        f"【用户问题】\n{user_input}\n\n"
        "请给出回答:"
    )


def consultant_node(state: AgentState) -> AgentState:
    """咨询节点:RAG 检索 + 生成回答。"""
    user_input = current_user_input(state)
    tenant_id = state.get("tenant_id", 1)
    # 上下文工程:主管已折叠好摘要时直接复用;否则这里按需准备(返回摘要 + 最近几轮)
    summary = state.get("summary") or ""
    if summary:
        history = recent_history_texts(state.get("messages") or [])
    else:
        summary, history = prepare_history(state.get("messages") or [])

    # 1) 混合检索(结合历史做上下文召回)
    docs = _search_kb(tenant_id, user_input, history)

    state["retrieved_docs"] = docs
    state["sources"] = [d.get("category", "") for d in docs]

    # 2) 生成回答(历史注入提示词,支持指代追问)
    try:
        llm = create_chat_model(temperature=0.5)
        prompt = build_consult_prompt(
            user_input,
            docs,
            history=history or None,
            summary=summary or None,
            long_term_context=state.get("context") or None,
        )
        resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_input)])
        state["reply"] = str(resp.content)
    except Exception as exc:  # pragma: no cover
        state["reply"] = f"抱歉,回答生成失败:{exc}"

    # 3) 回复写回消息历史,供下一轮记忆
    state["messages"] = [AIMessage(content=state["reply"])]
    return state


def _search_kb(tenant_id: int, query: str, history: list[str] | None = None) -> list[dict]:
    """通过 knowledge_service 检索;结合历史词补全查询意图。"""
    from app.db.database import SessionLocal
    from app.observability.trace import record_stage

    db = SessionLocal()
    try:
        # 不写死条数:召回/返回条数由 settings.RETRIEVER_TOP_K / RERANK_TOP_K 决定
        docs = knowledge_service.search(db, tenant_id, query)
        # 阶段快照:召回数量与文档 id 是排查「答非所问」的第一现场
        record_stage(
            "retrieve",
            tenant_id=tenant_id,
            hits=len(docs),
            doc_ids=[d.get("doc_id") for d in docs],
            categories=[d.get("category") for d in docs],
            scores=[d.get("score") for d in docs],
            rerank_scores=[d.get("rerank_score") for d in docs],
        )
        return docs
    finally:
        db.close()
