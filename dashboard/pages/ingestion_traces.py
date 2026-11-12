"""摄取追踪页:左侧记录列表,右侧阶段瀑布与失败详情。"""

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
    status_badge,
    trace_detail,
)

tenant_id, tenant_name = current_tenant()
st.title("摄取追踪")
st.caption(f"租户: {tenant_name} · 一次摄取 = load → caption → split → embed_upsert")

status_filter = st.radio("状态", ["全部", "success", "failed", "skipped"], horizontal=True)
data = ingestions(tenant_id, status=None if status_filter == "全部" else status_filter, limit=100)
rows = data.get("items", [])

if not rows:
    st.info("还没有摄取记录,请先去「摄取管理」上传文档。")
    st.stop()

left, right = st.columns((1.05, 2))

with left:
    st.subheader(f"记录({data['total']})")
    labels = [
        f"#{r['id']} · {r['filename'][:18]} · {r['status']}"
        for r in rows
    ]
    picked_label = st.radio("选择记录", labels, label_visibility="collapsed")
    picked = rows[labels.index(picked_label)]
    st.markdown(status_badge(picked["status"]), unsafe_allow_html=True)
    st.metric("块数", picked["chunks"])
    st.metric("图片", picked["images"])
    st.metric("耗时", fmt_ms(picked["elapsed_ms"]))
    st.caption(f"时间: {picked['created_at']}")
    if picked["error"]:
        st.error(picked["error"])
    if picked["trace_id"]:
        st.code(picked["trace_id"], language=None)

with right:
    st.subheader("阶段瀑布")
    if not picked["trace_id"]:
        st.info("该记录没有关联 trace(可能来自旧版本)。")
    else:
        try:
            detail = trace_detail(picked["trace_id"])
        except Exception as exc:  # noqa: BLE001 - trace 缺失时给出可读提示
            st.warning(f"读取 trace 失败: {exc}")
            detail = None

        if detail:
            stages = detail.get("stages", [])
            if not stages:
                st.info("该 trace 没有阶段记录。")
            else:
                max_ms = max((s["elapsed_ms"] for s in stages), default=1.0) or 1.0
                for stage in stages:
                    c1, c2, c3 = st.columns((1.3, 3.2, 0.9))
                    c1.markdown(f"**{stage['name']}**")
                    c2.progress(min(1.0, stage["elapsed_ms"] / max_ms))
                    c3.caption(fmt_ms(stage["elapsed_ms"]))

                st.divider()
                st.subheader("阶段明细")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "阶段": s["name"],
                                "耗时(ms)": s["elapsed_ms"],
                                "摘要": str(s["payload"])[:120],
                            }
                            for s in stages
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                with st.expander("查看原始 payload"):
                    st.json({s["name"]: s["payload"] for s in stages})

            if detail.get("error"):
                st.error(detail["error"])
            st.caption(
                f"trace {detail['trace_id']} · 状态 {detail['status']} · 总耗时 {fmt_ms(detail['total_ms'])}"
            )
