"""控制台公共层:数据库会话、管理端数据读取、格式化工具。

为什么要复用 `app/api/routes/admin.py` 的函数:
- 面板和 API 用的是同一套口径,避免「面板显示 92%、接口返回 90%」;
- 那些函数只依赖一个 `db` 参数,去掉 FastAPI 的依赖注入后可以直接调用。
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 必须无条件把项目根移到最前:只判断 "not in sys.path" 时,若项目根已在列表靠后位置
# 就不会被提前,而 dashboard/ 目录(内含同名 app.py)仍在前面,`import app.db` 会
# 命中 dashboard/app.py,导致本模块 `from app.db.database import ...` 触发循环导入。
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.database import SessionLocal, init_db  # noqa: E402

# 主题色:与 .streamlit/config.toml 保持一致
COLORS = {
    "primary": "#7C5CFF",
    "success": "#16C79A",
    "warning": "#F5A524",
    "danger": "#FF5C7A",
    "info": "#22A6F2",
    "bg": "#0B0E14",
    "surface": "#141926",
    "surface_alt": "#1C2333",
    "text": "#E8ECF4",
    "muted": "#8C96AB",
}

_STATUS_COLOR = {"success": COLORS["success"], "failed": COLORS["danger"], "skipped": COLORS["warning"]}


def setup_page(title: str, icon: str = "📊") -> None:
    """统一页面配置与全局样式。"""
    st.set_page_config(page_title=f"{title} · 知识库控制台", page_icon=icon, layout="wide")
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
        h1, h2, h3 {{ font-family: "Source Sans Pro", "Segoe UI", sans-serif; letter-spacing: .2px; }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(160deg, {COLORS['surface']} 0%, {COLORS['surface_alt']} 100%);
            border: 1px solid rgba(124,92,255,.18);
            border-radius: 14px; padding: 14px 16px;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: "JetBrains Mono", "Consolas", monospace;
            color: {COLORS['text']}; font-size: 1.7rem;
        }}
        div[data-testid="stMetricLabel"] {{ color: {COLORS['muted']}; }}
        section[data-testid="stSidebar"] {{
            background: {COLORS['bg']}; border-right: 1px solid {COLORS['surface_alt']};
        }}
        .stage-row {{ display: flex; gap: 12px; align-items: center; margin: 4px 0; }}
        .stage-name {{ width: 140px; color: {COLORS['muted']}; font-size: 13px; }}
        .pill {{
            display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 12px; font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def call_admin(fn, **kwargs) -> Any:
    """调用管理端函数,自动注入 db 会话。"""
    with session_scope() as db:
        return fn(db=db, **kwargs)


# ----------------------------------------------------------------------
# 常用数据入口
# ----------------------------------------------------------------------
def current_tenant() -> tuple[int, str]:
    """从会话状态取当前租户(入口页会写入)。"""
    return (
        int(st.session_state.get("tenant_id", 1) or 1),
        str(st.session_state.get("tenant_name", "")),
    )


def tenants() -> list[dict]:
    from app.api.routes.admin import collections

    try:
        return call_admin(collections, tenant_id=1).get("tenants", [])
    except Exception:  # noqa: BLE001 - 空库/未初始化时不应让面板崩
        return []


def overview(tenant_id: int, days: int = 7) -> dict:
    from app.api.routes.admin import overview as _overview

    return call_admin(_overview, tenant_id=tenant_id, days=days)


def docs(tenant_id: int, category: str | None, keyword: str, page: int, size: int) -> tuple[int, list[dict]]:
    from sqlalchemy import or_

    from app.db.models import KnowledgeDoc

    with session_scope() as db:
        query = db.query(KnowledgeDoc).filter(KnowledgeDoc.tenant_id == tenant_id)
        if category and category != "全部":
            query = query.filter(KnowledgeDoc.category == category)
        if keyword:
            like = f"%{keyword}%"
            query = query.filter(
                or_(KnowledgeDoc.content.like(like), KnowledgeDoc.source.like(like))
            )
        total = query.count()
        rows = (
            query.order_by(KnowledgeDoc.id.desc())
            .offset(max(0, (page - 1) * size))
            .limit(size)
            .all()
        )
        items = [
            {
                "id": d.id,
                "category": d.category,
                "content": d.content,
                "source": d.source,
                "page": d.page,
                "chunk_index": d.chunk_index,
                "doc_group": d.doc_group,
                "created_at": d.created_at.isoformat(sep=" ", timespec="seconds")
                if d.created_at
                else "",
            }
            for d in rows
        ]
        return total, items


def categories(tenant_id: int) -> list[str]:
    from app.db.models import KnowledgeDoc

    with session_scope() as db:
        rows = (
            db.query(KnowledgeDoc.category)
            .filter(KnowledgeDoc.tenant_id == tenant_id)
            .distinct()
            .all()
        )
        return sorted({r[0] for r in rows if r[0]})


def ingestions(tenant_id: int, status: str | None = None, limit: int = 100) -> dict:
    from app.api.routes.admin import list_ingestions

    return call_admin(list_ingestions, tenant_id=tenant_id, status=status, limit=limit)


def traces(
    kind: str | None = None,
    tenant_id: int | None = None,
    status: str | None = None,
    limit: int = 100,
) -> dict:
    from app.api.routes.admin import list_traces

    # status 由管理端接口原生支持,这里必须一并透传,
    # 否则「查询追踪」页的状态筛选会直接 TypeError。
    return call_admin(
        list_traces, kind=kind, tenant_id=tenant_id, status=status, limit=limit
    )


def trace_detail(trace_id: str) -> dict:
    from app.api.routes.admin import get_trace

    return call_admin(get_trace, trace_id=trace_id)


def eval_history(tenant_id: int, limit: int = 10) -> list[dict]:
    from app.db.models import EvalRun

    with session_scope() as db:
        rows = (
            db.query(EvalRun)
            .filter(EvalRun.tenant_id == tenant_id)
            .order_by(EvalRun.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(sep=" ", timespec="seconds") if r.created_at else "",
                "total": r.total,
                "top_k": r.top_k,
                "top1": round(r.top1_accuracy, 4),
                "top3": round(r.top3_accuracy, 4),
                "metrics": r.metrics or {},
                "config": r.config or {},
            }
            for r in rows
        ]


# ----------------------------------------------------------------------
# 展示辅助
# ----------------------------------------------------------------------
def status_badge(status: str) -> str:
    color = _STATUS_COLOR.get(status, COLORS["muted"])
    return (
        f"<span class='pill' style='background:{color}22;color:{color};"
        f"border:1px solid {color}55'>{status}</span>"
    )


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.2f} s"


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"
