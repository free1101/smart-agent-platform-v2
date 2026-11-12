"""文档摄取 CLI:把 PDF/Word/Markdown/文本导入知识库。

用法:
    python scripts/ingest_docs.py --path data/docs --tenant 示例行业-健康养生
    python scripts/ingest_docs.py --path 手册.pdf --force --no-caption
    python scripts/ingest_docs.py --path docs --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config.settings import settings  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402
from app.ingestion.pipeline import ingestion_pipeline  # noqa: E402
from app.services.knowledge_service import knowledge_service  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="文档摄取")
    parser.add_argument("--path", required=True, help="文件或目录路径")
    parser.add_argument("--tenant", default=settings.DEFAULT_TENANT_NAME, help="租户名")
    parser.add_argument("--industry", default="通用", help="租户行业标签")
    parser.add_argument("--category", default="文档导入", help="写入知识条目的分类")
    parser.add_argument("--force", action="store_true", help="忽略已导入记录,强制重新摄取")
    parser.add_argument("--no-caption", action="store_true", help="跳过图片描述(省成本)")
    parser.add_argument("--rebuild", action="store_true", help="摄取前先全量重建索引")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"[错误] 路径不存在: {target}")
        sys.exit(1)

    init_db()
    db = SessionLocal()
    try:
        tenant = knowledge_service.get_or_create_tenant(db, args.tenant, args.industry)
        if args.rebuild:
            knowledge_service.rebuild_index(db, tenant.id)
            print(f"[索引] 租户 {tenant.id} 已重建")

        if target.is_dir():
            results = ingestion_pipeline.run_directory(
                db,
                tenant.id,
                target,
                force=args.force,
                with_caption=not args.no_caption,
                category=args.category,
            )
        else:
            results = [
                ingestion_pipeline.run(
                    db,
                    tenant.id,
                    target,
                    force=args.force,
                    with_caption=not args.no_caption,
                    category=args.category,
                )
            ]

        print(f"\n{'状态':<10} {'块数':>5} {'图片':>5} {'耗时(ms)':>10}  文件")
        print("-" * 78)
        for r in results:
            print(f"{r.status:<10} {r.chunks:>5} {r.images:>5} {r.elapsed_ms:>10.1f}  {r.filename}")
            if r.status == "failed":
                print(f"           └─ {r.message}")

        ok = sum(1 for r in results if r.status == "success")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")
        print(f"\n汇总: 成功 {ok} / 跳过 {skipped} / 失败 {failed}")
        if failed:
            sys.exit(2)
    finally:
        db.close()


if __name__ == "__main__":
    main()
