"""总览页:四张关键指标卡 + 近七日趋势 + 失败清单。"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (  # noqa: E402
    COLORS,
    current_tenant,
    fmt_ms,
    ingestions,
    overview,
    pct,
    traces,
)

tenant_id, tenant_name = current_tenant()
st.title("总览")
st.caption(f"租户: {tenant_name} · 数据窗口: 近 7 天")

data = overview(tenant_id, days=7)

c1, c2, c3, c4 = st.columns(4)
c1.metric("知识条目", f"{data['total_docs']:,}", help="当前租户已入库的知识块数量")
c2.metric("今日查询", f"{data['today_queries']:,}", help="今天 00:00 起的 query trace 数")
c3.metric("平均检索延迟", fmt_ms(data["avg_latency_ms"]), help="query trace 的总耗时均值")
c4.metric(
    "评估命中率",
    pct(data["hit_rate"]),
    help="最近一次评测的 Top-k 分类命中率",
)

st.divider()

left, right = st.columns((3, 2))

with left:
    st.subheader("近七日查询量")
    trend = data.get("trend") or []
    if trend:
        frame = pd.DataFrame(trend)
        st.bar_chart(frame, x="date", y="queries", height=280, color=COLORS["primary"])
    else:
        st.info("近七日还没有查询记录。")

with right:
    st.subheader("延迟分位")
    st.metric("P95 延迟", fmt_ms(data["p95_latency_ms"]), help="95% 的请求快于该值")
    st.metric("失败请求", data["failed_runs"], help="status=failed 的 trace 数")
    if data.get("latest_eval_at"):
        st.caption(f"最近评测时间: {data['latest_eval_at']}")

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("最近失败的摄取")
    rows = ingestions(tenant_id, status="failed", limit=8).get("items", [])
    if rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "文件": r["filename"],
                        "耗时": fmt_ms(r["elapsed_ms"]),
                        "原因": (r["error"] or "")[:60],
                        "时间": r["created_at"],
                    }
                    for r in rows
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("没有失败的摄取记录。")

with col_b:
    st.subheader("最近异常查询")
    rows = traces(kind="query", tenant_id=tenant_id, limit=30).get("items", [])
    failed = [r for r in rows if r["status"] == "failed"][:8]
    if failed:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "query": (r["query"] or "")[:24],
                        "耗时": fmt_ms(r["total_ms"]),
                        "错误": (r["error"] or "")[:50],
                        "trace_id": r["trace_id"],
                    }
                    for r in failed
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("最近没有失败查询。")
