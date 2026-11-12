"""MCP 工具定义:业务工具(预约/订单)+ 演示第三方工具(天气)。

说明:
- 工具通过 MCP 标准协议暴露,可被任何 MCP 客户端(本项目的 LangGraph Agent、
  外部 AI 应用)复用,实现「一次定义,处处调用」
- 数据写入与业务 Agent 共用 booking_tickets 表,保证状态一致
"""

from __future__ import annotations

from datetime import datetime

from fastmcp import FastMCP

from app.db.database import SessionLocal
from app.db.models import BookingTicket

mcp = FastMCP(
    "smart-agent-tools",
    instructions="智能客服平台业务工具:预约管理、订单查询、天气等。",
)


@mcp.tool()
def create_booking(
    tenant_id: int,
    session_id: str,
    service_type: str,
    appointment_time: str,
    customer_name: str = "",
    notes: str = "",
) -> dict:
    """创建预约工单。

    Args:
        tenant_id: 租户 id
        session_id: 会话 id
        service_type: 服务类型(如 全身推拿)
        appointment_time: 预约时间(自然语言或具体时间)
        customer_name: 客户姓名(可选)
        notes: 备注(可选)
    Returns:
        包含 ticket_id 与状态的结果
    """
    db = SessionLocal()
    try:
        ticket = BookingTicket(
            tenant_id=tenant_id,
            session_id=session_id,
            service_type=service_type,
            appointment_time=appointment_time,
            customer_name=customer_name,
            notes=notes,
            status="pending",
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        return {"ticket_id": ticket.id, "status": "pending", "message": f"预约已创建,工单号 T-{ticket.id:04d}"}
    finally:
        db.close()


@mcp.tool()
def query_booking(ticket_id: int) -> dict:
    """查询预约工单状态。

    Args:
        ticket_id: 工单号(数字部分)
    Returns:
        工单详情;不存在时返回空结果
    """
    db = SessionLocal()
    try:
        ticket = db.query(BookingTicket).filter(BookingTicket.id == ticket_id).first()
        if ticket is None:
            return {"found": False, "message": f"未找到工单 T-{ticket_id:04d}"}
        return {
            "found": True,
            "ticket_id": ticket.id,
            "service_type": ticket.service_type,
            "appointment_time": ticket.appointment_time,
            "customer_name": ticket.customer_name,
            "status": ticket.status,
            "notes": ticket.notes,
        }
    finally:
        db.close()


@mcp.tool()
def cancel_booking(ticket_id: int) -> dict:
    """取消预约工单。

    Args:
        ticket_id: 工单号(数字部分)
    Returns:
        取消结果
    """
    db = SessionLocal()
    try:
        ticket = db.query(BookingTicket).filter(BookingTicket.id == ticket_id).first()
        if ticket is None:
            return {"success": False, "message": f"未找到工单 T-{ticket_id:04d}"}
        ticket.status = "cancelled"
        db.commit()
        return {"success": True, "ticket_id": ticket.id, "message": f"工单 T-{ticket.id:04d} 已取消"}
    finally:
        db.close()


@mcp.tool()
def get_weather(city: str) -> dict:
    """查询城市天气(演示第三方工具接入)。

    Args:
        city: 城市名,如 北京
    Returns:
        模拟天气数据(演示用,可替换为真实天气 API)
    """
    # 演示用模拟数据;生产环境可替换为真实天气 API
    data = {
        "北京": {"weather": "晴", "temp": 26, "humidity": 40},
        "上海": {"weather": "多云", "temp": 28, "humidity": 55},
        "广州": {"weather": "阵雨", "temp": 30, "humidity": 75},
        "深圳": {"weather": "雷阵雨", "temp": 29, "humidity": 80},
    }
    info = data.get(city, {"weather": "未知", "temp": None, "humidity": None})
    return {"city": city, "weather": info["weather"], "temp": info["temp"], "humidity": info["humidity"], "time": datetime.now().isoformat()}


@mcp.tool()
def query_knowledge_hub(query: str, tenant_id: int = 1, top_k: int = 3) -> dict:
    """在知识库中做混合检索(对外暴露的 RAG 检索能力)。

    Args:
        query: 用户的自然语言问题
        tenant_id: 租户 id(不同行业知识库互相隔离)
        top_k: 返回条数
    Returns:
        {"results": [{doc_id, content, category, score, source}], "count": n}
    """
    from app.services.knowledge_service import knowledge_service

    db = SessionLocal()
    try:
        results = knowledge_service.search(db, tenant_id, query, top_k=max(1, min(top_k, 10)))
        return {"query": query, "tenant_id": tenant_id, "count": len(results), "results": results}
    except Exception as exc:  # noqa: BLE001 - MCP 工具必须返回结构而不是抛栈
        return {"query": query, "tenant_id": tenant_id, "count": 0, "results": [], "error": str(exc)[:300]}
    finally:
        db.close()


@mcp.tool()
def list_collections(tenant_id: int = 1) -> dict:
    """列出知识库集合:所有租户 + 指定租户的分类统计。

    Args:
        tenant_id: 要查看统计的租户 id
    Returns:
        {"tenants": [...], "current": {"total_docs": n, "by_category": {...}}}
    """
    from app.services.knowledge_service import knowledge_service

    db = SessionLocal()
    try:
        tenants = [
            {"id": t.id, "name": t.name, "industry": t.industry}
            for t in knowledge_service.list_tenants(db)
        ]
        current = knowledge_service.stats(db, tenant_id)
        return {"tenants": tenants, "current": current}
    except Exception as exc:  # noqa: BLE001
        return {"tenants": [], "current": {}, "error": str(exc)[:300]}
    finally:
        db.close()


@mcp.tool()
def get_document_summary(doc_id: int) -> dict:
    """获取某个知识块及其同文档上下文的摘要信息。

    Args:
        doc_id: 知识条目 id
    Returns:
        文档元信息 + 同 doc_group 的相邻块内容(最多 5 块)
    """
    from app.db.models import KnowledgeDoc

    db = SessionLocal()
    try:
        doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
        if doc is None:
            return {"found": False, "message": f"未找到知识条目 {doc_id}"}

        siblings = (
            db.query(KnowledgeDoc)
            .filter(
                KnowledgeDoc.tenant_id == doc.tenant_id,
                KnowledgeDoc.doc_group == doc.doc_group,
            )
            .order_by(KnowledgeDoc.chunk_index, KnowledgeDoc.id)
            .limit(5)
            .all()
        ) if doc.doc_group else [doc]

        return {
            "found": True,
            "doc_id": doc.id,
            "tenant_id": doc.tenant_id,
            "category": doc.category,
            "source": doc.source,
            "page": doc.page,
            "chunk_index": doc.chunk_index,
            "doc_group": doc.doc_group,
            "content": doc.content,
            "siblings": [
                {"doc_id": s.id, "chunk_index": s.chunk_index, "content": s.content[:300]}
                for s in siblings
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "message": str(exc)[:300]}
    finally:
        db.close()


if __name__ == "__main__":
    # 以 streamable http 方式启动,供 LangGraph Agent 通过 langchain-mcp-adapters 调用。
    # 容器里需要监听 0.0.0.0,通过 MCP_HOST / MCP_PORT 覆盖。
    import os

    mcp.run(
        transport="streamable-http",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "9000")),
    )
