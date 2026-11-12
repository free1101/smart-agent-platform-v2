"""摄取去重:文件级(整篇重复导入)与块级(同一段话被多份文档包含)。

为什么需要两层:
- 文件级:同一份 PDF 被重复上传,整篇跳过最省事;
- 块级:产品手册与 FAQ 常常大段重复,块级去重能避免同一段话在检索结果里刷屏。
"""

from __future__ import annotations

import hashlib

from sqlalchemy.orm import Session

from app.db.models import IngestionRecord, KnowledgeDoc


def file_hash(data: bytes) -> str:
    """整篇文件指纹。"""
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    """单块内容指纹:去掉全部空白差异,避免排版空格/换行导致去重失效。"""
    normalized = "".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def find_success_record(db: Session, tenant_id: int, source_hash: str) -> IngestionRecord | None:
    """查找同一文件成功摄取的记录(用于幂等跳过)。"""
    return (
        db.query(IngestionRecord)
        .filter(
            IngestionRecord.tenant_id == tenant_id,
            IngestionRecord.source_hash == source_hash,
            IngestionRecord.status == "success",
        )
        .order_by(IngestionRecord.id.desc())
        .first()
    )


def existing_content_hashes(db: Session, tenant_id: int) -> set[str]:
    """该租户已有的块指纹集合。"""
    rows = (
        db.query(KnowledgeDoc.content_hash)
        .filter(KnowledgeDoc.tenant_id == tenant_id, KnowledgeDoc.content_hash != "")
        .all()
    )
    return {row[0] for row in rows if row[0]}
