"""摄取流水线:文件 → 文本块 → 图片描述 → 去重 → 入库 → 建索引。

一次 `run()` 的完整步骤(每一步都写入 trace,可在面板里回放):
1. 算文件指纹,命中成功记录且非 force → 直接跳过(幂等);
2. 按后缀选 Loader,PDF 保序抽文本并保留图片占位符;
3. 图片交给视觉模型生成描述,原位回填成 `【图片描述】...`;
4. 按 `split_text()` 切块,块级去重(同一段话只入库一次);
5. 写 `knowledge_docs`,增量更新 BM25 + 向量索引;
6. 写 `ingestion_records`(块数、图片数、耗时、失败原因)。

一致性:先 `flush()` 拿 id 建索引,成功后再 `commit()`;
索引失败则 `rollback()`,避免出现「库里有、索引里没有」的脏数据。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.db.models import IngestionRecord, KnowledgeDoc
from app.ingestion.dedup import existing_content_hashes, file_hash, find_success_record, text_hash
from app.ingestion.image_captioner import ImageCaptioner
from app.ingestion.loaders import load_document
from app.observability.trace import current_trace_id, record_stage, trace_scope
from app.rag.factory import create_splitter

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """一次摄取的结果(可直接 JSON 序列化给接口)。"""

    status: str  # success / skipped / failed
    filename: str = ""
    chunks: int = 0
    images: int = 0
    elapsed_ms: float = 0.0
    record_id: int = 0
    trace_id: str = ""
    message: str = ""
    skipped_reason: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "filename": self.filename,
            "chunks": self.chunks,
            "images": self.images,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "record_id": self.record_id,
            "trace_id": self.trace_id,
            "message": self.message,
            "skipped_reason": self.skipped_reason,
            "errors": self.errors,
        }


class IngestionPipeline:
    """文档摄取流水线。"""

    def __init__(self, captioner: ImageCaptioner | None = None) -> None:
        self.captioner = captioner or ImageCaptioner()
        self.splitter = create_splitter()

    def run(
        self,
        db: Session,
        tenant_id: int,
        path: str | Path,
        force: bool = False,
        with_caption: bool = True,
        category: str = "文档导入",
    ) -> IngestResult:
        path = Path(path)
        started = time.perf_counter()
        filename = path.name

        if not path.exists():
            return IngestResult(status="failed", filename=filename, message="文件不存在")

        try:
            data = path.read_bytes()
        except OSError as exc:
            return IngestResult(status="failed", filename=filename, message=f"读取失败: {exc}")

        source_hash = file_hash(data)

        if not force:
            existing = find_success_record(db, tenant_id, source_hash)
            if existing is not None:
                return IngestResult(
                    status="skipped",
                    filename=filename,
                    record_id=existing.id,
                    skipped_reason=f"该文件已于 {existing.created_at:%Y-%m-%d %H:%M} 导入过",
                )

        with trace_scope("ingestion", tenant_id=tenant_id, query=filename) as ctx:
            try:
                result = self._run_inner(
                    db=db,
                    tenant_id=tenant_id,
                    path=path,
                    data=data,
                    source_hash=source_hash,
                    filename=filename,
                    with_caption=with_caption,
                    category=category,
                    started=started,
                )
                # 统一在这里补 trace_id:各返回分支不必各自重复写
                result.trace_id = result.trace_id or ctx.trace_id
                if result.status == "failed":
                    ctx.status = "failed"
                    ctx.error = result.message
                return result
            except Exception as exc:  # noqa: BLE001 - 任何异常都要落成失败记录
                db.rollback()
                logger.exception("摄取失败: %s", filename)
                ctx.status = "failed"
                ctx.error = f"{type(exc).__name__}: {exc}"[:500]
                elapsed = (time.perf_counter() - started) * 1000
                record = IngestionRecord(
                    tenant_id=tenant_id,
                    filename=filename,
                    file_type=path.suffix.lstrip(".").lower(),
                    source_hash=source_hash,
                    status="failed",
                    elapsed_ms=elapsed,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    trace_id=ctx.trace_id,
                )
                db.add(record)
                db.commit()
                return IngestResult(
                    status="failed",
                    filename=filename,
                    elapsed_ms=elapsed,
                    record_id=record.id,
                    trace_id=ctx.trace_id,
                    message=f"摄取失败: {exc}",
                )

    # ------------------------------------------------------------------
    def _run_inner(
        self,
        db: Session,
        tenant_id: int,
        path: Path,
        data: bytes,
        source_hash: str,
        filename: str,
        with_caption: bool,
        category: str,
        started: float,
    ) -> IngestResult:
        from app.services.knowledge_service import knowledge_service

        # 1) 解析
        load_started = time.perf_counter()
        document = load_document(path)
        record_stage(
            "load",
            file_type=document.file_type,
            blocks=len(document.blocks),
            chars=len(document.text),
            images=len(document.images),
            elapsed_ms=round((time.perf_counter() - load_started) * 1000, 2),
        )
        if not document.blocks:
            return IngestResult(
                status="failed", filename=filename, message="未解析出任何文本内容"
            )

        # 2) 图片描述(可选)
        #    先对全篇图片统一描述(受全局上限约束),再按块切片回填,
        #    这样既保住分页 metadata,也不会因为按块调用而绕过图片数量上限。
        from app.ingestion.loaders import fill_image_captions

        caption_started = time.perf_counter()
        image_count = 0
        blocks_text: list[str] = []
        if with_caption and document.images:
            global_captions = self.captioner.caption(document.images)
            image_count = sum(1 for c in global_captions if c)
            offset = 0
            for block in document.blocks:
                size = len(block.images)
                local = global_captions[offset : offset + size]
                blocks_text.append(fill_image_captions(block.text, local))
                offset += size
        else:
            blocks_text = [block.text for block in document.blocks]
        record_stage(
            "caption",
            enabled=with_caption,
            images=len(document.images),
            captioned=image_count,
            elapsed_ms=round((time.perf_counter() - caption_started) * 1000, 2),
        )

        # 3) 分块 + 去重
        split_started = time.perf_counter()
        seen = existing_content_hashes(db, tenant_id)
        pending: list[KnowledgeDoc] = []
        duplicate = 0
        for block_index, text in enumerate(blocks_text):
            if not text.strip():
                continue
            page = document.blocks[block_index].page if block_index < len(document.blocks) else 0
            for chunk in self.splitter.split(text):
                digest = text_hash(chunk)
                if digest in seen:
                    duplicate += 1
                    continue
                seen.add(digest)
                pending.append(
                    KnowledgeDoc(
                        tenant_id=tenant_id,
                        content=chunk,
                        category=category,
                        keywords=[],
                        source=f"upload:{filename}",
                        source_hash=source_hash,
                        content_hash=digest,
                        chunk_index=len(pending),
                        doc_group=filename,
                        page=page,
                    )
                )
        record_stage(
            "split",
            chunks=len(pending),
            duplicates=duplicate,
            chunk_size=getattr(self.splitter, "chunk_size", None),
            elapsed_ms=round((time.perf_counter() - split_started) * 1000, 2),
        )

        if not pending:
            elapsed = (time.perf_counter() - started) * 1000
            record = IngestionRecord(
                tenant_id=tenant_id,
                filename=filename,
                file_type=path.suffix.lstrip(".").lower(),
                source_hash=source_hash,
                status="success",
                chunks=0,
                images=image_count,
                elapsed_ms=elapsed,
                error="",
                trace_id=current_trace_id(),
            )
            db.add(record)
            db.commit()
            return IngestResult(
                status="skipped",
                filename=filename,
                images=image_count,
                elapsed_ms=elapsed,
                record_id=record.id,
                message="内容已全部存在,跳过写入",
            )

        # 4) 写库(flush 拿 id,不提交)
        db.add_all(pending)
        db.flush()

        payloads = [knowledge_service._payload(doc) for doc in pending]  # noqa: SLF001
        embed_started = time.perf_counter()
        knowledge_service.ingest_payloads(db, tenant_id, payloads)
        record_stage(
            "embed_upsert",
            docs=len(payloads),
            elapsed_ms=round((time.perf_counter() - embed_started) * 1000, 2),
        )

        elapsed = (time.perf_counter() - started) * 1000
        record = IngestionRecord(
            tenant_id=tenant_id,
            filename=filename,
            file_type=path.suffix.lstrip(".").lower(),
            source_hash=source_hash,
            status="success",
            chunks=len(pending),
            images=image_count,
            elapsed_ms=elapsed,
            error="",
            trace_id=current_trace_id(),
        )
        db.add(record)
        db.commit()

        return IngestResult(
            status="success",
            filename=filename,
            chunks=len(pending),
            images=image_count,
            elapsed_ms=elapsed,
            record_id=record.id,
            message=f"已导入 {len(pending)} 个块(跳过重复 {duplicate} 个)",
        )

    # ------------------------------------------------------------------
    def run_directory(
        self,
        db: Session,
        tenant_id: int,
        directory: str | Path,
        force: bool = False,
        with_caption: bool = True,
        category: str = "文档导入",
    ) -> list[IngestResult]:
        """批量摄取一个目录(只处理受支持的后缀,失败不影响后续文件)。"""
        from app.ingestion.loaders import SUPPORTED_SUFFIXES

        results: list[IngestResult] = []
        for path in sorted(Path(directory).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            results.append(
                self.run(
                    db,
                    tenant_id,
                    path,
                    force=force,
                    with_caption=with_caption,
                    category=category,
                )
            )
        return results


ingestion_pipeline = IngestionPipeline()
