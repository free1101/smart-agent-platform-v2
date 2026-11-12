"""数据浏览页:按分类/来源/关键词筛选知识条目,支持分页与详情展开。"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import categories, current_tenant, docs  # noqa: E402

tenant_id, tenant_name = current_tenant()
st.title("数据浏览")
st.caption(f"租户: {tenant_name}")

f1, f2, f3, f4 = st.columns((1.2, 1.6, 1.2, 0.8))
with f1:
    category = st.selectbox("分类", ["全部", *categories(tenant_id)])
with f2:
    keyword = st.text_input("关键词", placeholder="正文或来源包含…")
with f3:
    size = st.select_slider("每页条数", options=[10, 20, 50, 100], value=20)
with f4:
    page = st.number_input("页码", min_value=1, value=1, step=1)

total, items = docs(tenant_id, category, keyword, int(page), int(size))
st.markdown(f"共 **{total}** 条,当前第 {page} 页")

if not items:
    st.info("没有匹配的知识条目。")
else:
    frame = pd.DataFrame(
        [
            {
                "id": i["id"],
                "分类": i["category"],
                "正文": i["content"][:70] + ("…" if len(i["content"]) > 70 else ""),
                "来源": i["source"],
                "页": i["page"],
                "块": i["chunk_index"],
                "创建时间": i["created_at"],
            }
            for i in items
        ]
    )
    st.dataframe(frame, hide_index=True, use_container_width=True, height=460)

    st.divider()
    st.subheader("条目详情")
    choice = st.selectbox(
        "选择条目",
        [f"#{i['id']} · {i['category']} · {i['content'][:30]}" for i in items],
    )
    picked = items[[f"#{i['id']} · {i['category']} · {i['content'][:30]}" for i in items].index(choice)]

    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("分类", picked["category"])
    meta2.metric("来源", picked["source"] or "-")
    meta3.metric("页码 / 块号", f"{picked['page']} / {picked['chunk_index']}")

    st.text_area("完整正文", picked["content"], height=200, disabled=True)
    st.caption(f"所属文档组: {picked['doc_group'] or '-'} · 入库时间: {picked['created_at']}")
