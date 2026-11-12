"""全量重建向量索引与 BM25 索引。

什么时候需要:
- 首次从旧版本升级(旧数据只有 JSON 字段,没有 Chroma collection);
- 手动改了数据库里的知识条目;
- 向量索引目录损坏或被改名。

用法:
    python scripts/rebuild_index.py                       # 重建默认租户
    python scripts/rebuild_index.py --tenant 示例行业-健康养生
    python scripts/rebuild_index.py --all                 # 重建所有租户
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config.settings import settings  # noqa: E402
from app.db.database import SessionLocal, init_db  # noqa: E402
from app.db.models import Tenant  # noqa: E402
from app.services.knowledge_service import knowledge_service  # noqa: E402


def rebuild(db, tenant: Tenant) -> dict:
    started = time.perf_counter()
    knowledge_service.invalidate(tenant.id)
    knowledge_service.rebuild_index(db, tenant.id)
    elapsed = (time.perf_counter() - started) * 1000
    stats = knowledge_service.stats(db, tenant.id)
    stats["elapsed_ms"] = round(elapsed, 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="重建知识索引")
    parser.add_argument("--tenant", default=settings.DEFAULT_TENANT_NAME, help="租户名")
    parser.add_argument("--all", action="store_true", help="重建所有租户")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.all:
            tenants = db.query(Tenant).order_by(Tenant.id).all()
        else:
            tenant = db.query(Tenant).filter(Tenant.name == args.tenant).first()
            if tenant is None:
                print(f"[错误] 租户不存在: {args.tenant}")
                print("可先运行: python scripts/import_knowledge_seed.py")
                sys.exit(1)
            tenants = [tenant]

        if not tenants:
            print("[提示] 没有任何租户,先导入种子知识。")
            return

        for tenant in tenants:
            print(f"\n重建租户 #{tenant.id} {tenant.name} …")
            stats = rebuild(db, tenant)
            print(
                f"  完成: 文档 {stats['total_docs']} 条 / 向量 {stats['vector_count']} 条 / "
                f"后端 {stats['vector_backend']} / 融合 {stats['fusion_strategy']} / "
                f"耗时 {stats['elapsed_ms']} ms"
            )
            print(f"  分类分布: {stats['by_category']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
