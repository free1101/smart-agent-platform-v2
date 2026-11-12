"""RAG 检索评测脚本。

基础指标:Top-1 / Top-k 分类命中率(兼容旧报告字段);
扩展指标:Recall@k(期望关键词覆盖率)、MRR、ContextPrecision@k;
可选指标:`--with-judge` 让 LLM 评估忠实度与答案相关性,`--with-ragas` 接 Ragas。

用法:
    python scripts/run_eval.py
    python scripts/run_eval.py --top_k 5 --with-judge
    python scripts/run_eval.py --dataset data/eval_dataset.json --output data/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK,强制 UTF-8 输出避免特殊字符报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config.settings import settings  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402
from app.db.models import Tenant  # noqa: E402
from app.observability.evaluation.runner import EvaluationRunner  # noqa: E402

EVAL_FILE = settings.DATA_DIR / "eval_dataset.json"
DEFAULT_OUTPUT = settings.DATA_DIR / "eval_report.json"


def resolve_tenant(db, tenant_name: str) -> int:
    tenant = db.query(Tenant).filter(Tenant.name == tenant_name).first()
    if tenant is None:
        print(f"[错误] 租户不存在: {tenant_name}")
        print("请先运行: python scripts/import_knowledge_seed.py")
        sys.exit(1)
    return tenant.id


def print_report(report: dict) -> None:
    rows = report["results"]
    print(f"{'序号':<4} {'期望分类':<10} {'Top1分类':<12} {'命中':<4} 问题")
    print("-" * 96)
    for i, row in enumerate(rows, 1):
        mark = "✓" if row["hit"] else "✗"
        print(f"{i:<4} {row['expected_category']:<10} {row['top1_category']:<12} {mark:<4} {row['query'][:38]}")

    metrics = report["metrics"]
    print("\n" + "=" * 96)
    print("评测报告")
    print(f"  样本数:              {report['total']}")
    print(f"  Top-1 分类命中率:    {report['top1_accuracy']:.1%}")
    print(f"  Top-{report['top_k']} 分类命中率:    {report['top3_accuracy']:.1%}")
    print(f"  MRR:                 {metrics['mrr']:.4f}")
    if metrics.get("recall") is None:
        print("  Recall@k:            未标注 expected_keywords,跳过")
    else:
        print(f"  Recall@k:            {metrics['recall']:.1%} ({metrics['recall_cases']} 条适用)")
    if metrics.get("context_precision") is None:
        print("  ContextPrecision@k:  无标注,跳过")
    else:
        print(
            f"  ContextPrecision@k:  {metrics['context_precision']:.1%} "
            f"({metrics['precision_cases']} 条适用)"
        )

    judge = report.get("judge") or {}
    if judge.get("enabled"):
        print(f"  Faithfulness:        {judge.get('faithfulness')} (判定 {judge.get('judged')} 条)")
        print(f"  AnswerRelevancy:     {judge.get('answer_relevancy')}")
    ragas = report.get("ragas") or {}
    if ragas.get("enabled"):
        if ragas.get("available"):
            print(f"  Ragas:               {ragas.get('metrics')}")
        else:
            print(f"  Ragas:               unavailable ({ragas.get('error', '')[:60]})")
    print(f"  耗时:                {report.get('elapsed_ms', 0)} ms")
    print("=" * 96)

    failed = report.get("failed_cases") or []
    if failed:
        print(f"\n未命中 Top-{report['top_k']} 的案例({len(failed)} 条,重点排查):")
        for row in failed:
            print(f"  - [{row['expected_category']}] {row['query'][:38]} -> {row['hit_categories']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument("--tenant", default=settings.DEFAULT_TENANT_NAME, help="租户名")
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="评测口径的检索条数(固定默认 3,与 RETRIEVER_TOP_K 解耦以保证历史可比)",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="报告输出路径")
    parser.add_argument("--dataset", type=str, default=str(EVAL_FILE), help="评测集路径")
    parser.add_argument("--with-judge", action="store_true", help="启用 LLM 忠实度/相关性判定")
    parser.add_argument("--with-ragas", action="store_true", help="尝试使用 Ragas 指标")
    args = parser.parse_args()

    init_db()
    if not Path(args.dataset).exists():
        print(f"[错误] 评测集不存在: {args.dataset}")
        sys.exit(1)

    db = SessionLocal()
    try:
        tenant_id = resolve_tenant(db, args.tenant)
        runner = EvaluationRunner(
            db=db,
            tenant_id=tenant_id,
            tenant_name=args.tenant,
            top_k=args.top_k,
            with_judge=args.with_judge,
            with_ragas=args.with_ragas,
            dataset_path=args.dataset,
        )
        report = runner.run()
        print_report(report)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n评测详情已保存: {out}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
