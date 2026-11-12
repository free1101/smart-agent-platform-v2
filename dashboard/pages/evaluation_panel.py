"""评估面板:运行 golden set 评测、查看指标卡与失败用例、对比历史。"""

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
    eval_history,
    pct,
    session_scope,
)

from app.config.settings import settings  # noqa: E402
from app.observability.evaluation.runner import EvaluationRunner  # noqa: E402

tenant_id, tenant_name = current_tenant()
st.title("评估面板")
st.caption(f"租户: {tenant_name} · 评测集 data/eval_dataset.json")

with st.form("eval_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        top_k = st.slider("Top-k", min_value=1, max_value=10, value=3)
    with c2:
        with_judge = st.toggle(
            "启用 LLM 判定",
            value=False,
            help="额外计算忠实度与答案相关性;每条用例多两次模型调用,较慢",
        )
    with c3:
        with_ragas = st.toggle("尝试 Ragas", value=False, help="未安装 ragas 时自动跳过")
    submitted = st.form_submit_button("运行评测", type="primary", use_container_width=True)

if submitted:
    if with_judge and not settings.has_llm_key:
        st.warning("未配置 LLM_API_KEY,LLM 判定会标记为 judge_failed,不给默认分。")
    with st.spinner("评测中…"):
        try:
            with session_scope() as db:
                runner = EvaluationRunner(
                    db=db,
                    tenant_id=tenant_id,
                    tenant_name=tenant_name,
                    top_k=int(top_k),
                    with_judge=with_judge,
                    with_ragas=with_ragas,
                )
                report = runner.run()
            st.session_state["last_report"] = report
            st.success(f"评测完成:{report['total']} 条,耗时 {report['elapsed_ms']} ms")
        except Exception as exc:  # noqa: BLE001 - 面板不能因为一次评测失败就白屏
            st.error(f"评测失败: {exc}")

report = st.session_state.get("last_report")
history = eval_history(tenant_id, limit=10)

if report:
    metrics = report["metrics"]
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Top-1 命中", pct(report["top1_accuracy"]))
    c2.metric(f"Top-{report['top_k']} 命中", pct(report["top3_accuracy"]))
    c3.metric("MRR", f"{metrics['mrr']:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall@k", pct(metrics.get("recall")), help="期望关键词在 Top-k 正文中的覆盖率")
    c5.metric("ContextPrecision@k", pct(metrics.get("context_precision")))
    judge = report.get("judge") or {}
    c6.metric("Faithfulness", pct(judge.get("faithfulness")), help="答案是否忠实于上下文")

    st.caption(
        f"配置: 融合 {report['config']['fusion_strategy']} · "
        f"重排 {report['config']['reranker']} · 向量库 {report['config']['vector_store']}"
    )

    ragas = report.get("ragas") or {}
    if ragas.get("enabled"):
        if ragas.get("available"):
            st.json(ragas.get("metrics"))
        else:
            st.info(f"Ragas 不可用: {ragas.get('error', '')}")

    st.subheader("逐条结果")
    frame = pd.DataFrame(
        [
            {
                "query": r["query"][:30],
                "期望分类": r["expected_category"],
                "Top1": r["top1_category"],
                "命中": "✓" if r["hit"] else "✗",
                "MRR": r["mrr"],
                "Recall": r["recall"],
                "ContextP": r["context_precision"],
                "难度": r["difficulty"],
            }
            for r in report["results"]
        ]
    )
    st.dataframe(frame, hide_index=True, use_container_width=True)

    failed = report.get("failed_cases") or []
    if failed:
        st.subheader(f"未命中用例({len(failed)})")
        for row in failed:
            with st.expander(f"[{row['expected_category']}] {row['query'][:36]}"):
                st.write(f"**命中分类**: {row['hit_categories']}")
                st.write(f"**doc_ids**: {row['doc_ids']}")
                for idx, content in enumerate(row["contents"], start=1):
                    st.markdown(f"`{idx}` {content}")
                if row.get("judge_reason"):
                    st.caption(f"判定理由: {row['judge_reason']}")

if history:
    st.divider()
    st.subheader("历史回归")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "时间": h["created_at"],
                    "样本": h["total"],
                    "Top-1": h["top1"],
                    "Top-k": h["top3"],
                    "Recall": (h["metrics"] or {}).get("recall"),
                    "MRR": (h["metrics"] or {}).get("mrr"),
                    "融合": (h["config"] or {}).get("fusion_strategy"),
                    "重排": (h["config"] or {}).get("reranker"),
                }
                for h in history
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    if len(history) >= 2:
        latest, previous = history[0], history[1]
        delta = latest["top3"] - previous["top3"]
        st.metric(
            "Top-k 相对上次",
            pct(latest["top3"]),
            delta=f"{delta:+.2%}",
            delta_color="normal" if delta >= 0 else "inverse",
        )
else:
    st.info("还没有历史评测记录,运行一次后会自动写入 eval_history.jsonl 与 eval_runs 表。")
