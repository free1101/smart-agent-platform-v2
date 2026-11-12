"""数据库引擎与会话管理(SQLAlchemy 2.0)。"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config.settings import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """FastAPI 依赖:提供数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 旧库补列定义:create_all 只建新表,不会给已存在的表加列。
# 键 = 表名,值 = {列名: SQLite 列类型与默认值}
_LEGACY_COLUMNS: dict[str, dict[str, str]] = {
    "knowledge_docs": {
        "source_hash": "VARCHAR(64) DEFAULT ''",
        "content_hash": "VARCHAR(64) DEFAULT ''",
        "chunk_index": "INTEGER DEFAULT 0",
        "doc_group": "VARCHAR(200) DEFAULT ''",
        "page": "INTEGER DEFAULT 0",
    },
}


def _ensure_legacy_columns() -> None:
    """为已存在的表补齐新增列(幂等,仅 SQLite)。

    为什么要手写:项目使用 create_all 建表,它对已存在的表不做任何改动;
    而 data/smart_agent.db 是长期保留的存量库,直接加列会报 no such column。
    """
    if not settings.DATABASE_URL.startswith("sqlite"):
        return

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in _LEGACY_COLUMNS.items():
            if table not in existing_tables:
                continue
            present = {col["name"] for col in inspector.get_columns(table)}
            for name, ddl in columns.items():
                if name in present:
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                logger.info("数据库迁移: %s 新增列 %s", table, name)


def init_db() -> None:
    """建表(导入数据模型后调用),并为存量 SQLite 库补齐新增列。"""
    from app.db import models  # noqa: F401  确保模型已注册

    Base.metadata.create_all(bind=engine)
    try:
        _ensure_legacy_columns()
    except Exception as exc:  # noqa: BLE001 - 迁移失败不应阻断启动,但必须留痕
        logger.error("数据库列迁移失败(可能影响新功能): %s", exc)
