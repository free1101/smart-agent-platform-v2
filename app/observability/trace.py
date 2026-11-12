"""全链路追踪:一次查询 / 一次摄取 = 一条 trace,内部由多个 stage 组成。

为什么用 `contextvars` 而不是层层传参:
- `KnowledgeService.search()` 被 Agent、脚本、HTTP 路由三处调用,加参数会污染接口;
- `ContextVar` 在当前执行上下文内隐式生效,拿不到 trace 时 `record_stage()` 静默返回,
  所以单测和 CLI 直接调用检索器不需要任何改动。

落库策略:整个 trace 结束时一次性 `add_all`,避免每个阶段一次 I/O;
payload 做截断(单字段 ≤ 300 字符,单 trace ≤ 50 阶段),防止长文本撑爆数据库。
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

MAX_STAGES = 50
MAX_STR = 300
MAX_LIST = 20


@dataclass
class Stage:
    """链路中的一个阶段。"""

    name: str
    elapsed_ms: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceContext:
    """一条 trace 的运行时容器。"""

    trace_id: str
    kind: str  # query | ingestion
    tenant_id: int = 0
    query: str = ""
    stages: list[Stage] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    status: str = "success"
    error: str = ""
    truncated: bool = False

    def add(self, name: str, payload: dict[str, Any], elapsed_ms: float = 0.0) -> None:
        if len(self.stages) >= MAX_STAGES:
            self.truncated = True
            return
        self.stages.append(
            Stage(name=name, elapsed_ms=round(elapsed_ms, 2), payload=_safe(payload))
        )

    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 2)


_current: ContextVar[TraceContext | None] = ContextVar("smart_agent_trace", default=None)


def current_trace() -> TraceContext | None:
    return _current.get()


def current_trace_id() -> str:
    ctx = _current.get()
    return ctx.trace_id if ctx else ""


def record_stage(name: str, elapsed_ms: float | None = None, **payload: Any) -> None:
    """记录一个阶段;不在 trace 上下文内时静默返回(供脚本/单测直接调用)。

    `elapsed_ms` 支持两种写法:显式参数,或混在 `**payload` 里(历史调用点如
    bm25/dense/index_rebuild 就是这么传的)。两种都统一写进 `Stage.elapsed_ms`,
    否则控制台的时间线会全部显示 0 ms。
    """
    ctx = _current.get()
    if ctx is None:
        return
    if elapsed_ms is None:
        raw = payload.pop("elapsed_ms", None)
        elapsed_ms = float(raw) if isinstance(raw, (int, float)) else 0.0
    ctx.add(name, payload, elapsed_ms)


@contextmanager
def trace_scope(
    kind: str,
    tenant_id: int = 0,
    query: str = "",
    persist: bool = True,
) -> Iterator[TraceContext]:
    """开启一条 trace;退出时落库(异常也会记录失败状态后再抛出)。"""
    ctx = TraceContext(
        trace_id=uuid.uuid4().hex[:16],
        kind=kind,
        tenant_id=int(tenant_id or 0),
        query=(query or "")[:MAX_STR],
    )
    token = _current.set(ctx)
    try:
        yield ctx
    except Exception as exc:  # noqa: BLE001 - 记录失败原因后原样抛出
        ctx.status = "failed"
        ctx.error = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        # StreamingResponse 用线程池迭代同步生成器:生成器的每次 next() 可能跑在
        # 不同线程/不同 Context 里,`reset(token)` 会因 token 不属于当前 Context 而抛
        # ValueError(表现为 SSE 连接中断)。此时无法恢复上一层的值,降级为清空当前
        # Context —— 语义上等价于「这条 trace 结束」,且不会连带把业务打挂。
        try:
            _current.reset(token)
        except (ValueError, RuntimeError):
            _current.set(None)
        if persist:
            _persist(ctx)


def _persist(ctx: TraceContext) -> None:
    """把 trace 写入数据库;失败只记日志,不影响主流程。"""
    try:
        from app.db.database import SessionLocal
        from app.db.models import TraceRun, TraceStage

        db = SessionLocal()
        try:
            run = TraceRun(
                trace_id=ctx.trace_id,
                kind=ctx.kind,
                tenant_id=ctx.tenant_id,
                query=ctx.query,
                status=ctx.status,
                total_ms=ctx.total_ms(),
                error=ctx.error,
            )
            db.add(run)
            db.add_all(
                [
                    TraceStage(
                        trace_id=ctx.trace_id,
                        seq=i,
                        name=stage.name,
                        elapsed_ms=stage.elapsed_ms,
                        payload=stage.payload,
                    )
                    for i, stage in enumerate(ctx.stages)
                ]
            )
            db.commit()
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 - 追踪是旁路能力,不能反过来搞挂业务
        logger.warning("trace 落库失败(trace_id=%s): %s", ctx.trace_id, exc)


def _safe(value: Any, depth: int = 0) -> Any:
    """把任意值裁剪成 JSON 可存的形状。"""
    if depth > 3:
        return "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_STR else value[:MAX_STR] + "…"
    if isinstance(value, dict):
        return {str(k): _safe(v, depth + 1) for k, v in list(value.items())[:MAX_LIST]}
    if isinstance(value, (list, tuple, set)):
        items = list(value)[:MAX_LIST]
        trimmed = [_safe(v, depth + 1) for v in items]
        if len(value) > MAX_LIST:
            trimmed.append("...")
        return trimmed
    return str(value)[:MAX_STR]
