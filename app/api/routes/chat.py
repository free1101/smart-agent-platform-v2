"""对话与知识库相关 API。

提供:
- POST /api/chat        对话(多 Agent 图)
- GET  /api/tenants    租户列表
- POST /api/kb/search  知识检索(用于调试/展示检索效果)
- GET  /api/kb/stats   知识统计
"""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from app.agents.graph import agent_graph
from app.api.deps import require_api_key
from app.api.ratelimit import check_rate_limit
from app.config.settings import settings
from app.db.database import get_db
from app.db.models import Tenant
from app.observability.trace import current_trace_id, record_stage, trace_scope
from app.services.knowledge_service import knowledge_service

router = APIRouter(
    prefix="/api",
    tags=["对话与知识库"],
    dependencies=[Depends(require_api_key), Depends(check_rate_limit)],
)


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    session_id: str = Field("default", description="会话 id,用于多轮关联")
    tenant_id: int = Field(1, description="租户 id")
    user_key: str = Field(
        "",
        description="用户标识(手机号/账号),用于跨会话长期记忆;缺省退化为 session_id",
    )


class ChatResponse(BaseModel):
    reply: str
    intent: str
    intent_reason: str = ""
    sources: list[str] = []
    booking: dict | None = None
    session_id: str
    trace_id: str = ""


class KBSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant_id: int = Field(1)
    # 默认跟随 settings.RETRIEVER_TOP_K,显式传值时以请求为准
    top_k: int = Field(default_factory=lambda: settings.RETRIEVER_TOP_K, ge=1, le=20)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def get_tenant_or_404(db: Session, tenant_id: int) -> Tenant:
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"租户 {tenant_id} 不存在")
    return tenant


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@router.get("/tenants", summary="租户列表")
def list_tenants(db: Session = Depends(get_db)):
    tenants = knowledge_service.list_tenants(db)
    return [
        {
            "id": t.id,
            "name": t.name,
            "industry": t.industry,
            "description": t.description,
        }
        for t in tenants
    ]


@router.post("/chat", response_model=ChatResponse, summary="对话")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    get_tenant_or_404(db, req.tenant_id)
    # 多轮记忆:以 session_id 作为 LangGraph 的 thread_id,
    # 同一会话的后续轮次自动带上历史消息与预约草稿
    thread_id = f"t{req.tenant_id}-{req.session_id}"
    with trace_scope("query", tenant_id=req.tenant_id, query=req.message) as ctx:
        result = agent_graph.invoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "session_id": req.session_id,
                "tenant_id": req.tenant_id,
                "user_key": req.user_key or req.session_id,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        record_stage(
            "graph_result",
            intent=result.get("intent", ""),
            sources=result.get("sources", []),
            retrieved=len(result.get("retrieved_docs") or []),
        )
        return ChatResponse(
            reply=result.get("reply", ""),
            intent=result.get("intent", "consult"),
            intent_reason=result.get("intent_reason", ""),
            sources=result.get("sources", []),
            booking=result.get("booking"),
            session_id=req.session_id,
            trace_id=ctx.trace_id,
        )


@router.post("/chat/stream", summary="对话(SSE 流式)")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """SSE 流式对话:节点进度 + 打字机效果。

    事件协议(text/event-stream):
      event: meta   data: {"session_id": ...}
      event: node   data: {"node": "supervisor"/"consultant"/...}  每完成一个 Agent 推送一次
      event: delta  data: {"text": "..."}                          回复内容逐块推送(打字机)
      event: done   data: {"reply": ..., "intent": ..., "sources": [...]}
      event: error  data: {"message": "..."}                       图执行异常

    说明:图的执行本身是同步整段完成,delta 是执行完后按固定字符数切块推送
    (演示用;接真实 token 流可把 consultant 的 llm.invoke 换成 llm.stream)。
    """
    get_tenant_or_404(db, req.tenant_id)
    thread_id = f"t{req.tenant_id}-{req.session_id}"
    inputs = {
        "messages": [HumanMessage(content=req.message)],
        "session_id": req.session_id,
        "tenant_id": req.tenant_id,
        "user_key": req.user_key or req.session_id,
    }
    config = {"configurable": {"thread_id": thread_id}}

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def generate():
        # trace 覆盖整个流式会话;退出 with 时统一落库(含失败状态)
        with trace_scope("query", tenant_id=req.tenant_id, query=req.message) as ctx:
            # 1) meta: 会话就绪
            yield _sse("meta", {"session_id": req.session_id, "trace_id": ctx.trace_id})

            # 2) 让图跑起来,同时逐节点推送进度(updates 模式:每完成一个节点给一次状态增量)。
            #    每次增量只含"该节点改动的键",因此把各节点增量合并成完整 result。
            merged: dict = {}
            try:
                for chunk in agent_graph.stream(inputs, config=config, stream_mode="updates"):
                    for node_name, node_state in (chunk or {}).items():
                        yield _sse("node", {"node": node_name})
                        if node_state and isinstance(node_state, dict):
                            merged.update(node_state)  # 增量逐个合并 → 最终态
            except Exception as exc:  # noqa: BLE001 - 图执行失败也要告诉前端,避免挂死
                ctx.status = "failed"
                ctx.error = str(exc)[:300]
                yield _sse("error", {"message": str(exc)})
                return

            result = merged
            reply = result.get("reply") or ""
            record_stage(
                "graph_result",
                intent=result.get("intent", ""),
                sources=result.get("sources", []),
                retrieved=len(result.get("retrieved_docs") or []),
            )
            # 3) 打字机:把回复切块逐段推送
            step = 6  # 每块字符数,越大越快
            for i in range(0, len(reply), step):
                yield _sse("delta", {"text": reply[i : i + step]})
                time.sleep(0.02)  # 模拟打字节奏(仅演示,生产可用真实 token 流)

            # 4) done: 附完整结果(含 intent/sources,便于前端收尾)
            yield _sse(
                "done",
                {
                    "reply": reply,
                    "intent": result.get("intent", "consult"),
                    "intent_reason": result.get("intent_reason", ""),
                    "sources": result.get("sources", []),
                    "booking": result.get("booking"),
                    "trace_id": ctx.trace_id,
                },
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用代理缓冲,保证逐块到达
            "Connection": "keep-alive",
        },
    )


@router.post("/kb/search", summary="知识检索(调试检索效果)")
def kb_search(req: KBSearchRequest, db: Session = Depends(get_db)):
    get_tenant_or_404(db, req.tenant_id)
    with trace_scope("query", tenant_id=req.tenant_id, query=req.query) as ctx:
        results = knowledge_service.search(db, req.tenant_id, req.query, top_k=req.top_k)
        record_stage("kb_search", hits=len(results), top_ids=[d.get("doc_id") for d in results])
        return {
            "query": req.query,
            "tenant_id": req.tenant_id,
            "trace_id": ctx.trace_id,
            "results": results,
        }


@router.get("/kb/stats", summary="知识库统计")
def kb_stats(tenant_id: int = 1, db: Session = Depends(get_db)):
    get_tenant_or_404(db, tenant_id)
    return knowledge_service.stats(db, tenant_id)


@router.get("/kb/docs", summary="知识列表")
def kb_docs(tenant_id: int = 1, limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    get_tenant_or_404(db, tenant_id)
    return {"total": len(knowledge_service.list_docs(db, tenant_id, limit=10000)), "docs": knowledge_service.list_docs(db, tenant_id, limit=limit, offset=offset)}


@router.get("/profile", summary="用户画像与行为历史")
def profile(tenant_id: int = 1, session_id: str = "default", db: Session = Depends(get_db)):
    """查询用户画像、最近行为与回访建议(供前端/调试展示个性化能力)。"""
    from app.services.profile_service import (
        get_profile,
        list_behaviors,
        should_return_reminder,
        generate_return_message,
    )

    get_tenant_or_404(db, tenant_id)
    profile_data = get_profile(db, tenant_id, session_id)
    behaviors = list_behaviors(db, tenant_id, session_id, limit=10)
    return {
        "profile": profile_data,
        "behaviors": behaviors,
        "return_reminder": {
            "should": should_return_reminder(db, tenant_id, session_id),
            "message": generate_return_message(db, tenant_id, session_id),
        },
    }


@router.get("/chat/tickets", summary="最近工单")
def chat_tickets(tenant_id: int = 1, limit: int = 10, db: Session = Depends(get_db)):
    from app.db.models import BookingTicket

    tickets = (
        db.query(BookingTicket)
        .filter(BookingTicket.tenant_id == tenant_id)
        .order_by(BookingTicket.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": t.id,
            "customer_name": t.customer_name,
            "service_type": t.service_type,
            "appointment_time": t.appointment_time,
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tickets
    ]
