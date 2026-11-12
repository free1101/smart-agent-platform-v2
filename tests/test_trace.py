"""链路追踪测试:阶段落库、payload 截断、异常状态、无 trace 时静默。"""

from __future__ import annotations

import pytest

from app.db.models import TraceRun, TraceStage
from app.observability.trace import (
    MAX_STAGES,
    MAX_STR,
    current_trace_id,
    record_stage,
    trace_scope,
)


def test_record_stage_without_trace_is_silent():
    """脚本直接调用检索器时没有 trace 上下文,record_stage 不能抛异常。"""
    record_stage("bm25", candidates=3)
    assert current_trace_id() == ""


def test_trace_scope_collects_stages(mem_db):
    with trace_scope("query", tenant_id=7, query="推拿多少钱") as ctx:
        record_stage("bm25", candidates=5)
        record_stage("dense", candidates=4)
        trace_id = ctx.trace_id

    assert trace_id
    run = mem_db.query(TraceRun).filter(TraceRun.trace_id == trace_id).one()
    assert run.kind == "query"
    assert run.tenant_id == 7
    assert run.status == "success"
    assert run.query == "推拿多少钱"

    stages = (
        mem_db.query(TraceStage)
        .filter(TraceStage.trace_id == trace_id)
        .order_by(TraceStage.seq)
        .all()
    )
    assert [s.name for s in stages] == ["bm25", "dense"]
    assert stages[0].payload["candidates"] == 5


def test_trace_payload_truncation(mem_db):
    long_text = "x" * (MAX_STR * 2)
    with trace_scope("query", tenant_id=1, query="q") as ctx:
        record_stage("retrieve", content=long_text, ids=list(range(100)))

    stage = (
        mem_db.query(TraceStage)
        .filter(TraceStage.trace_id == ctx.trace_id)
        .one()
    )
    assert len(stage.payload["content"]) == MAX_STR + 1  # 截断后追加省略号
    assert stage.payload["content"].endswith("…")
    assert stage.payload["ids"][-1] == "..."


def test_trace_stage_limit(mem_db):
    with trace_scope("query", tenant_id=1, query="q") as ctx:
        for i in range(MAX_STAGES + 5):
            record_stage(f"s{i}")
        assert ctx.truncated is True

    count = mem_db.query(TraceStage).filter(TraceStage.trace_id == ctx.trace_id).count()
    assert count == MAX_STAGES


def test_trace_marks_failure_and_reraises(mem_db):
    with pytest.raises(ValueError):
        with trace_scope("ingestion", tenant_id=2, query="bad.pdf"):
            raise ValueError("解析失败")

    run = mem_db.query(TraceRun).filter(TraceRun.tenant_id == 2).one()
    assert run.status == "failed"
    assert "解析失败" in run.error


def test_trace_id_exposed_inside_scope():
    with trace_scope("query", tenant_id=1, query="q", persist=False) as ctx:
        assert current_trace_id() == ctx.trace_id
    assert current_trace_id() == ""


def test_trace_persist_can_be_disabled(mem_db):
    with trace_scope("query", tenant_id=1, query="q", persist=False):
        record_stage("x")
    assert mem_db.query(TraceRun).count() == 0
