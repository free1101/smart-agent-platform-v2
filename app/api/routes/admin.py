"""管理端 API:文档摄取、索引重建、链路追踪查询、总览指标。

这些接口同时被 Streamlit 控制台和命令行使用,所以返回结构以「直接可渲染」为准:
- 列表接口统一返回 {"total": n, "items": [...]};
- 详情接口把 trace 的 stages 按 seq 展开,前端不需要再排序。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import require_api_key
from app.api.ratelimit import check_rate_limit
from app.config.settings import settings
from app.db.database import get_db
from app.db.models import EvalRun, IngestionRecord, KnowledgeDoc, Tenant, TraceRun, TraceStage
from app.ingestion.pipeline import ingestion_pipeline
from app.services.knowledge_service import knowledge_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin",
    tags=["管理端"],
    dependencies=[Depends(require_api_key), Depends(check_rate_limit)],
)

ALLOWED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def _tenant_or_404(db: Session, tenant_id: int) -> Tenant:
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"租户 {tenant_id} 不存在")
    return tenant


# ---------------------------------------------------------------------------
# 摄取
# ---------------------------------------------------------------------------
@router.post("/ingest", summary="上传并摄取文档")
async def ingest_document(
    file: UploadFile = File(..., description="PDF / Word / Markdown / 文本"),
    tenant_id: int = Form(1),
    category: str = Form("文档导入"),
    with_caption: bool = Form(True),
    force: bool = Form(False),
    db: Session = Depends(get_db),
):
    _tenant_or_404(db, tenant_id)
    filename = Path(file.filename or "upload.txt").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"不支持的文件类型 {suffix},仅支持 {sorted(ALLOWED_SUFFIXES)}")

    data = await file.read()
    max_bytes = settings.INGEST_MAX_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(413, f"文件超过 {settings.INGEST_MAX_MB}MB 限制")

    target_dir = Path(settings.UPLOAD_DIR) / f"tenant_{tenant_id}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    target.write_bytes(data)

    result = await run_in_threadpool(
        ingestion_pipeline.run,
        db,
        tenant_id,
        target,
        force,
        with_caption,
        category,
    )
    return result.to_dict()


@router.post("/ingest/rebuild", summary="重建租户索引")
def rebuild_index(tenant_id: int = 1, db: Session = Depends(get_db)):
    _tenant_or_404(db, tenant_id)
    knowledge_service.invalidate(tenant_id)
    knowledge_service.rebuild_index(db, tenant_id)
    return knowledge_service.stats(db, tenant_id)


@router.get("/ingestions", summary="摄取记录列表")
def list_ingestions(
    tenant_id: int = 1,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(IngestionRecord).filter(IngestionRecord.tenant_id == tenant_id)
    if status:
        query = query.filter(IngestionRecord.status == status)
    total = query.count()
    rows = query.order_by(IngestionRecord.id.desc()).offset(offset).limit(min(limit, 200)).all()
    return {
        "total": total,
        "items": [
            {
                "id": r.id,
                "filename": r.filename,
                "file_type": r.file_type,
                "status": r.status,
                "chunks": r.chunks,
                "images": r.images,
                "elapsed_ms": round(r.elapsed_ms, 1),
                "error": r.error,
                "trace_id": r.trace_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# 追踪
# ---------------------------------------------------------------------------
@router.get("/traces", summary="链路列表")
def list_traces(
    kind: str | None = None,
    tenant_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(TraceRun)
    if kind:
        query = query.filter(TraceRun.kind == kind)
    if tenant_id is not None:
        query = query.filter(TraceRun.tenant_id == tenant_id)
    if status:
        query = query.filter(TraceRun.status == status)
    total = query.count()
    rows = query.order_by(TraceRun.id.desc()).offset(offset).limit(min(limit, 200)).all()
    return {
        "total": total,
        "items": [
            {
                "trace_id": r.trace_id,
                "kind": r.kind,
                "tenant_id": r.tenant_id,
                "query": r.query,
                "status": r.status,
                "total_ms": round(r.total_ms, 1),
                "error": r.error,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/traces/{trace_id}", summary="链路详情(阶段瀑布)")
def get_trace(trace_id: str, db: Session = Depends(get_db)):
    run = db.query(TraceRun).filter(TraceRun.trace_id == trace_id).first()
    if run is None:
        raise HTTPException(404, f"trace {trace_id} 不存在")
    stages = (
        db.query(TraceStage)
        .filter(TraceStage.trace_id == trace_id)
        .order_by(TraceStage.seq)
        .all()
    )
    return {
        "trace_id": run.trace_id,
        "kind": run.kind,
        "tenant_id": run.tenant_id,
        "query": run.query,
        "status": run.status,
        "total_ms": round(run.total_ms, 1),
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "stages": [
            {
                "seq": s.seq,
                "name": s.name,
                "elapsed_ms": round(s.elapsed_ms, 2),
                "payload": s.payload,
            }
            for s in stages
        ],
    }


# ---------------------------------------------------------------------------
# 总览
# ---------------------------------------------------------------------------
@router.get("/overview", summary="总览指标")
def overview(tenant_id: int = 1, days: int = 7, db: Session = Depends(get_db)):
    _tenant_or_404(db, tenant_id)
    since = datetime.utcnow() - timedelta(days=max(1, min(days, 30)))

    total_docs = db.query(KnowledgeDoc).filter(KnowledgeDoc.tenant_id == tenant_id).count()
    query_runs = (
        db.query(TraceRun)
        .filter(TraceRun.tenant_id == tenant_id, TraceRun.kind == "query", TraceRun.created_at >= since)
        .all()
    )
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_queries = (
        db.query(func.count(TraceRun.id))
        .filter(TraceRun.tenant_id == tenant_id, TraceRun.kind == "query", TraceRun.created_at >= today_start)
        .scalar()
        or 0
    )
    latencies = sorted(r.total_ms for r in query_runs)
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0

    latest_eval = (
        db.query(EvalRun).filter(EvalRun.tenant_id == tenant_id).order_by(EvalRun.id.desc()).first()
    )

    daily: dict[str, dict] = {}
    for i in range(max(1, min(days, 30))):
        day = (today_start - timedelta(days=i)).strftime("%m-%d")
        daily[day] = {"date": day, "queries": 0, "latency_sum": 0.0}
    for run in query_runs:
        key = run.created_at.strftime("%m-%d") if run.created_at else ""
        if key in daily:
            daily[key]["queries"] += 1
            daily[key]["latency_sum"] += run.total_ms

    return {
        "tenant_id": tenant_id,
        "total_docs": total_docs,
        "today_queries": int(today_queries),
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": round(p95, 1),
        "queries_in_range": len(query_runs),
        "failed_runs": sum(1 for r in query_runs if r.status == "failed"),
        "hit_rate": latest_eval.top3_accuracy if latest_eval else None,
        "latest_eval_at": latest_eval.created_at.isoformat() if latest_eval else None,
        "trend": [
            {
                "date": item["date"],
                "queries": item["queries"],
                "avg_latency_ms": round(item["latency_sum"] / item["queries"], 1)
                if item["queries"]
                else 0.0,
            }
            for item in reversed(list(daily.values()))
        ],
    }


@router.get("/collections", summary="租户与集合信息")
def collections(tenant_id: int = 1, db: Session = Depends(get_db)):
    tenants = [
        {"id": t.id, "name": t.name, "industry": t.industry}
        for t in knowledge_service.list_tenants(db)
    ]
    stats = knowledge_service.stats(db, tenant_id) if tenants else {}
    return {"tenants": tenants, "current": stats}
