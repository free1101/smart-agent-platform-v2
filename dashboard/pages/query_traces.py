"""查询追踪页:按 trace 查看 bm25 → dense → fuse → retrieve/rerank 的每一步。"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import COLORS, current_tenant, fmt_ms, trace_detail, traces  # noqa: E402

tenant_id, tenant_name = current_tenant()
st.title("查询追踪")
st.caption(f"租户: {tenant_name} · 一次查询的每个中间状态都可回看")

f1, f2 = st.columns((1, 1))
with f1:
    status_filter = st.radio("状态", ["全部", "success", "failed"], horizontal=True)
with f2:
    limit = st.select_slider("显示条数", options=[20, 50, 100, 200], value=50)

data = traces(
    kind="query",
    tenant_id=tenant_id,
    status=None if status_filter == "全部" else status_filter,
    limit=limit,
)
rows = data.get("items", [])

if not rows:
    st.info("还没有查询记录。去「对话」页面问一句,或调用 POST /api/kb/search。")
    st.stop()

st.dataframe(
    pd.DataFrame(
        [
            {
                "trace_id": r["trace_id"],
                "query": (r["query"] or "")[:30],
                "状态": r["status"],
                "耗时": fmt_ms(r["total_ms"]),
                "时间": r["created_at"],
            }
            for r in rows
        ]
    ),
    hide_index=True,
    use_container_width=True,
    height=260,
)

st.divider()
labels = [f"{r['trace_id']} · {r['query'][:28]} · {fmt_ms(r['total_ms'])}" for r in rows]
picked_label = st.selectbox("选择一次查询", labels)
picked = rows[labels.index(picked_label)]

try:
    detail = trace_detail(picked["trace_id"])
except Exception as exc:  # noqa: BLE001
    st.error(f"读取 trace 失败: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("总耗时", fmt_ms(detail["total_ms"]))
c2.metric("阶段数", len(detail["stages"]))
c3.metric("状态", detail["status"])
c4.metric("租户", detail["tenant_id"])

st.markdown(f"**查询**: {detail['query'] or '-'}")
if detail.get("error"):
    st.error(detail["error"])

stages = detail.get("stages", [])
if stages:
    max_ms = max((s["elapsed_ms"] for s in stages), default=1.0) or 1.0
    st.subheader("阶段时间线")
    for stage in stages:
        a, b, c = st.columns((1.2, 3.3, 0.9))
        a.markdown(f"**{stage['name']}**")
        b.progress(min(1.0, stage["elapsed_ms"] / max_ms))
        c.caption(fmt_ms(stage["elapsed_ms"]))

    st.subheader("召回与融合明细")
    recall_rows = []
    for stage in stages:
        payload = stage["payload"] or {}
        if stage["name"] in {"bm25", "dense", "fuse", "retrieve", "kb_search"}:
            recall_rows.append(
                {
                    "阶段": stage["name"],
                    "候选数": payload.get("candidates", payload.get("hits", "")),
                    "Top ids": str(payload.get("top_ids", payload.get("doc_ids", "")))[:60],
                    "说明": str(
                        {
                            k: v
                            for k, v in payload.items()
                            if k not in {"top_ids", "doc_ids", "categories", "scores", "rerank_scores"}
                        }
                    )[:80],
                }
            )
    if recall_rows:
        st.dataframe(pd.DataFrame(recall_rows), hide_index=True, use_container_width=True)

    retrieve = next((s["payload"] for s in stages if s["name"] == "retrieve"), None)
    if retrieve:
        st.subheader("命中知识")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "doc_id": doc_id,
                        "category": cat,
                        "融合分": score,
                        "重排分": (retrieve.get("rerank_scores") or [None] * 99)[idx],
                    }
                    for idx, (doc_id, cat, score) in enumerate(
                        zip(
                            retrieve.get("doc_ids") or [],
                            retrieve.get("categories") or [],
                            retrieve.get("scores") or [],
                        )
                    )
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("查看全部阶段原始 payload"):
        st.json({s["name"]: s["payload"] for s in stages})
else:
    st.info("该 trace 没有阶段记录。")
