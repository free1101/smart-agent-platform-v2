"""
知识库种子数据导入脚本(新架构版)

将 data/knowledge_seed.json 导入到默认租户(示例行业-健康养生),并重建混合检索索引。

用法:
    python scripts/import_knowledge_seed.py [--tenant 租户名] [--replace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings  # noqa: E402
from app.db.database import init_db  # noqa: E402
from app.db.database import SessionLocal  # noqa: E402
from app.services.knowledge_service import knowledge_service  # noqa: E402

SEED_FILE = settings.DATA_DIR / "knowledge_seed.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="导入知识种子数据")
    parser.add_argument("--tenant", default=settings.DEFAULT_TENANT_NAME, help="目标租户名")
    parser.add_argument("--industry", default="健康养生", help="行业标签")
    parser.add_argument("--replace", action="store_true", help="先清空该租户已有知识再导入")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if not SEED_FILE.exists():
            print(f"[错误] seed 文件不存在: {SEED_FILE}")
            sys.exit(1)
        result = knowledge_service.import_seed(
            db,
            seed_path=SEED_FILE,
            tenant_name=args.tenant,
            industry=args.industry,
            source_tag="seed",
            replace=args.replace,
        )
        print(f"导入完成:")
        print(f"  租户: {args.tenant} (id={result['tenant_id']})")
        print(f"  本次导入: {result['imported']} 条(含分块)")
        print(f"  该租户知识总数: {result['total_docs']} 条")

        # 分类统计
        stats = knowledge_service.stats(db, result["tenant_id"])
        print("\n--- 分类统计 ---")
        for cat, cnt in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {cnt} 条")
    finally:
        db.close()


if __name__ == "__main__":
    main()
