"""摄取管理页:上传文件、触发摄取、重建索引。"""

from __future__ import annotations

import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import (  # noqa: E402
    current_tenant,
    fmt_ms,
    ingestions,
    session_scope,
    status_badge,
)

from app.config.settings import settings  # noqa: E402

tenant_id, tenant_name = current_tenant()
st.title("摄取管理")
st.caption(f"租户: {tenant_name} · 单文件上限 {settings.INGEST_MAX_MB}MB")

with st.expander("支持的文件类型与处理流程", expanded=False):
    st.markdown(
        """
- **PDF**(pymupdf):按页抽取文本,页内按纵坐标保序;图片抽出后由视觉模型生成描述,
  以 `【图片描述】…` 回填到原位置,实现「搜文字能出图」。
- **Word**(python-docx):按段落顺序拼接,内联图片附在文末。
- **Markdown / 文本**:UTF-8 优先,失败回退 GBK。
- 重复文件按 `sha256` 跳过;同一段话被多份文档包含时按块级指纹去重。
        """
    )

uploaded = st.file_uploader(
    "选择文件",
    type=["pdf", "docx", "md", "markdown", "txt"],
    accept_multiple_files=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    category = st.text_input("写入分类", value="文档导入")
with c2:
    with_caption = st.toggle("生成图片描述", value=True, help="需要配置 LLM_API_KEY")
with c3:
    force = st.toggle("强制覆盖", value=False, help="忽略已导入记录,重新解析")
with c4:
    st.write("")
    run = st.button("开始摄取", type="primary", use_container_width=True)

if run:
    if not uploaded:
        st.warning("请先选择文件。")
    else:
        from app.ingestion.pipeline import ingestion_pipeline

        results = []
        progress = st.progress(0.0, text="准备中…")
        for idx, item in enumerate(uploaded, start=1):
            progress.progress((idx - 1) / len(uploaded), text=f"正在处理 {item.name}")
            target_dir = Path(settings.UPLOAD_DIR) / f"tenant_{tenant_id}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / Path(item.name).name
            target.write_bytes(item.getbuffer())

            with session_scope() as db:
                result = ingestion_pipeline.run(
                    db,
                    tenant_id,
                    target,
                    force=force,
                    with_caption=with_caption,
                    category=category,
                )
            results.append(result)
            st.markdown(
                f"{item.name} → {status_badge(result.status)} {result.message}",
                unsafe_allow_html=True,
            )

        progress.progress(1.0, text="完成")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "文件": r.filename,
                        "状态": r.status,
                        "块数": r.chunks,
                        "图片": r.images,
                        "耗时": fmt_ms(r.elapsed_ms),
                        "trace": r.trace_id,
                        "说明": r.message,
                    }
                    for r in results
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

st.divider()

left, right = st.columns((2, 1))
with left:
    st.subheader("摄取记录")
    status_filter = st.radio(
        "状态筛选", ["全部", "success", "failed", "skipped"], horizontal=True, label_visibility="collapsed"
    )
    data = ingestions(tenant_id, status=None if status_filter == "全部" else status_filter, limit=100)
    rows = data.get("items", [])
    if rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "id": r["id"],
                        "文件": r["filename"],
                        "类型": r["file_type"],
                        "状态": r["status"],
                        "块数": r["chunks"],
                        "图片": r["images"],
                        "耗时": fmt_ms(r["elapsed_ms"]),
                        "时间": r["created_at"],
                        "错误": (r["error"] or "")[:40],
                    }
                    for r in rows
                ]
            ),
            hide_index=True,
            use_container_width=True,
            height=360,
        )
    else:
        st.info("还没有摄取记录。")

with right:
    st.subheader("索引维护")
    st.caption("重建会按数据库全量重算 embedding,文档多时耗时较长。")
    if st.button("重建当前租户索引", use_container_width=True):
        from app.services.knowledge_service import knowledge_service

        with st.spinner("重建中…"):
            with session_scope() as db:
                knowledge_service.invalidate(tenant_id)
                knowledge_service.rebuild_index(db, tenant_id)
                stats = knowledge_service.stats(db, tenant_id)
        st.success("重建完成")
        st.json(stats)
