"""知识库服务:负责知识导入、索引构建、检索与管理。

多租户设计:
- 每个 tenant(行业)一套知识库 + 一套向量 collection(`kb_tenant_{id}`)
- 检索时按 tenant_id 隔离,互不干扰
- 支持从 seed JSON 批量导入、增量追加、删除与全量重建

索引生命周期:
- 首次访问时懒加载(lazy),避免服务启动就为所有租户建索引;
- 用 `(文档数, 最大 updated_at)` 作为版本键,数据变化后自动重建;
- 向量库不可用时回退内存实现并记 ERROR,保证服务仍可启动。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import settings
from app.db.models import KnowledgeDoc, Tenant
from app.observability.trace import record_stage
from app.rag.base import EmbeddingIdentityMismatch
from app.rag.factory import create_reranker, create_vector_store
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.text_splitter import split_text

logger = logging.getLogger(__name__)


class KnowledgeService:
    """知识库核心服务(单例,按 tenant 缓存检索器)。"""

    def __init__(self) -> None:
        # tenant_id -> HybridRetriever
        self._retrievers: dict[int, HybridRetriever] = {}
        # tenant_id -> (doc_count, max_updated_at):数据变动后缓存自动失效
        self._index_versions: dict[int, tuple[int, str]] = {}
        self._reranker = create_reranker()

    # ------------------------------------------------------------------
    # 租户管理
    # ------------------------------------------------------------------
    def get_or_create_tenant(self, db: Session, name: str, industry: str = "通用", description: str = "") -> Tenant:
        tenant = db.query(Tenant).filter(Tenant.name == name).first()
        if tenant is None:
            tenant = Tenant(name=name, industry=industry, description=description)
            db.add(tenant)
            db.commit()
            db.refresh(tenant)
        return tenant

    def list_tenants(self, db: Session) -> list[Tenant]:
        return db.query(Tenant).order_by(Tenant.id).all()

    # ------------------------------------------------------------------
    # 知识导入
    # ------------------------------------------------------------------
    def import_seed(
        self,
        db: Session,
        seed_path: str | Path,
        tenant_name: str,
        industry: str = "通用",
        source_tag: str = "seed",
        replace: bool = False,
    ) -> dict:
        """从 seed JSON 导入知识。

        seed 格式:[{"category": "...", "content": "...", "keywords": [...]}, ...]

        Args:
            replace: True 时清空该 tenant 已有文档再导入
        Returns:
            {"tenant_id": int, "imported": int, "total_docs": int}
        """
        seed_path = Path(seed_path)
        if not seed_path.exists():
            raise FileNotFoundError(f"seed 文件不存在: {seed_path}")

        with seed_path.open("r", encoding="utf-8") as f:
            items = json.load(f)

        tenant = self.get_or_create_tenant(db, tenant_name, industry)
        if replace:
            db.query(KnowledgeDoc).filter(KnowledgeDoc.tenant_id == tenant.id).delete()
            db.commit()
            self.invalidate(tenant.id)

        imported = 0
        for item in items:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            # 长文本先分块,一条知识可切成多块,块间共享 category/keywords
            chunks = split_text(content)
            if not chunks:
                chunks = [content]
            for idx, chunk in enumerate(chunks):
                doc = KnowledgeDoc(
                    tenant_id=tenant.id,
                    content=chunk,
                    category=item.get("category", "通用"),
                    keywords=item.get("keywords", []),
                    source=f"{source_tag}:{seed_path.name}",
                    chunk_index=idx,
                    doc_group=seed_path.name,
                )
                db.add(doc)
                imported += 1
        db.commit()

        total = db.query(KnowledgeDoc).filter(KnowledgeDoc.tenant_id == tenant.id).count()
        # 重建索引
        self.rebuild_index(db, tenant.id)
        return {"tenant_id": tenant.id, "imported": imported, "total_docs": total}

    def add_doc(self, db: Session, tenant_id: int, content: str, category: str = "通用", keywords: list[str] | None = None) -> KnowledgeDoc:
        """手动新增一条知识并增量更新索引。"""
        doc = KnowledgeDoc(
            tenant_id=tenant_id,
            content=content.strip(),
            category=category,
            keywords=keywords or [],
            source="manual",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        self.ingest_payloads(db, tenant_id, [self._payload(doc)])
        return doc

    def delete_doc(self, db: Session, doc_id: int) -> bool:
        """删除知识并增量更新索引。"""
        doc = db.query(KnowledgeDoc).filter(KnowledgeDoc.id == doc_id).first()
        if doc is None:
            return False
        tenant_id = doc.tenant_id
        db.delete(doc)
        db.commit()
        self.delete_docs(db, tenant_id, [doc_id])
        return True

    # ------------------------------------------------------------------
    # 索引与检索
    # ------------------------------------------------------------------
    def invalidate(self, tenant_id: int) -> None:
        """让某租户的索引缓存失效(下一次访问重建)。"""
        self._retrievers.pop(tenant_id, None)
        self._index_versions.pop(tenant_id, None)

    def rebuild_index(self, db: Session, tenant_id: int, reset_store: bool = True) -> None:
        """从数据库全量重建该租户的混合检索索引。"""
        started = time.perf_counter()
        docs = (
            db.query(KnowledgeDoc)
            .filter(KnowledgeDoc.tenant_id == tenant_id, KnowledgeDoc.is_active == True)  # noqa: E712
            .order_by(KnowledgeDoc.id)
            .all()
        )
        payloads = [self._payload(d) for d in docs]

        retriever = HybridRetriever(
            vector_store=self._create_store(tenant_id),
            tenant_id=tenant_id,
        )
        retriever.build(payloads, reset_store=reset_store)
        self._retrievers[tenant_id] = retriever
        self._index_versions[tenant_id] = self._version(db, tenant_id)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        logger.info("租户 %s 索引重建完成: %s 条,耗时 %sms", tenant_id, len(payloads), elapsed)
        record_stage(
            "index_rebuild",
            tenant_id=tenant_id,
            docs=len(payloads),
            elapsed_ms=elapsed,
            # 记录真实后端类型,降级到内存时一眼可见
            backend=type(retriever._vector_store).__name__  # noqa: SLF001
            if retriever._vector_store is not None
            else "numpy",
        )

    def ingest_payloads(self, db: Session, tenant_id: int, payloads: list[dict]) -> None:
        """增量写入索引(摄取流水线写库后调用)。"""
        if not payloads:
            return
        retriever = self._get_retriever(db, tenant_id)
        if retriever is None:
            return
        retriever.append_docs(payloads)
        self._index_versions[tenant_id] = self._version(db, tenant_id)

    def delete_docs(self, db: Session, tenant_id: int, doc_ids: list[int]) -> None:
        """增量删除索引。"""
        retriever = self._get_retriever(db, tenant_id)
        if retriever is None:
            return
        retriever.remove_docs(doc_ids)
        self._index_versions[tenant_id] = self._version(db, tenant_id)

    def search(
        self,
        db: Session,
        tenant_id: int,
        query: str,
        top_k: int | None = None,
        rerank: bool = True,
        where: dict | None = None,
    ) -> list[dict]:
        """混合检索 + 可选重排,返回可序列化结果。

        `top_k` 为 None 时走配置默认:召回 `settings.RETRIEVER_TOP_K` 条,
        精排后返回 `settings.RERANK_TOP_K` 条(取两者较小值);
        显式传 `top_k` 时召回与返回都用该值,请求级参数始终优先。

        返回字段保持向后兼容:`doc_id/content/category/score/source`,
        新增 `rerank_score`(与 score 区分,便于前端展示重排前后的变化)。
        """
        retriever = self._get_retriever(db, tenant_id)
        if retriever is None:
            return []

        recall_k = top_k if top_k is not None else settings.RETRIEVER_TOP_K
        return_k = top_k if top_k is not None else settings.RERANK_TOP_K
        recall_k = max(1, recall_k)
        return_k = max(1, min(return_k, recall_k))

        # 候选数要大于最终返回条数,否则精排没有挑选空间
        candidates = max(recall_k, settings.RERANK_CANDIDATES if rerank else recall_k)
        hits = retriever.search(query, top_k=candidates, where=where)
        if rerank and self._reranker is not None:
            hits = self._reranker.rerank(query, hits, top_k=return_k)
        return [h.to_dict() for h in hits[:return_k]]

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self, db: Session, tenant_id: int) -> dict:
        total = db.query(KnowledgeDoc).filter(KnowledgeDoc.tenant_id == tenant_id).count()
        by_category: dict[str, int] = {}
        rows = (
            db.query(KnowledgeDoc.category, KnowledgeDoc.id)
            .filter(KnowledgeDoc.tenant_id == tenant_id)
            .all()
        )
        for cat, _ in rows:
            key = cat or "通用"
            by_category[key] = by_category.get(key, 0) + 1

        vector_count = 0
        retriever = self._retrievers.get(tenant_id)
        if retriever is not None:
            vector_count = retriever.doc_count()
        else:
            # 该进程还没发生过检索时缓存为空,不能为了统计去构建 retriever
            # (build 会对全部文档重算 embedding)。直接读向量库的条目数。
            try:
                vector_count = self._create_store(tenant_id).count()
            except Exception as exc:  # noqa: BLE001 - 统计失败不影响其它字段
                logger.warning("读取向量库条数失败(tenant=%s): %s", tenant_id, exc)
        return {
            "tenant_id": tenant_id,
            "total_docs": total,
            "by_category": by_category,
            "vector_count": vector_count,
            "vector_backend": settings.VECTOR_STORE,
            "fusion_strategy": settings.FUSION_STRATEGY,
            "reranker": self._reranker.name if self._reranker is not None else "none",
        }

    def list_docs(self, db: Session, tenant_id: int, limit: int = 100, offset: int = 0) -> list[dict]:
        docs = (
            db.query(KnowledgeDoc)
            .filter(KnowledgeDoc.tenant_id == tenant_id)
            .order_by(KnowledgeDoc.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            {
                "id": d.id,
                "content": d.content,
                "category": d.category,
                "keywords": d.keywords or [],
                "source": d.source,
                "is_active": d.is_active,
                "page": d.page,
                "chunk_index": d.chunk_index,
                "doc_group": d.doc_group,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _payload(doc: KnowledgeDoc) -> dict:
        return {
            "id": doc.id,
            "content": doc.content,
            "category": doc.category,
            "keywords": doc.keywords or [],
            "source": doc.source,
            "is_active": doc.is_active,
            "content_hash": doc.content_hash,
            "chunk_index": doc.chunk_index,
            "page": doc.page,
            "doc_group": doc.doc_group,
        }

    @staticmethod
    def _version(db: Session, tenant_id: int) -> tuple[int, str]:
        from sqlalchemy import func

        count, latest = (
            db.query(func.count(KnowledgeDoc.id), func.max(KnowledgeDoc.updated_at))
            .filter(KnowledgeDoc.tenant_id == tenant_id)
            .one()
        )
        return int(count or 0), latest.isoformat() if latest else ""

    def _create_store(self, tenant_id: int):
        """创建向量库;身份不匹配时回退内存并留下 ERROR 日志。"""
        try:
            return create_vector_store(tenant_id)
        except EmbeddingIdentityMismatch as exc:
            logger.error(
                "租户 %s 的向量索引与当前 embedding 配置不匹配,已回退内存索引: %s",
                tenant_id,
                exc,
            )
            from app.rag.vector_store import InMemoryVectorStore

            return InMemoryVectorStore(tenant_id)

    def _get_retriever(self, db: Session, tenant_id: int) -> HybridRetriever | None:
        """取租户检索器;缓存版本与库中数据不一致时自动重建。"""
        retriever = self._retrievers.get(tenant_id)
        if retriever is not None:
            version = self._index_versions.get(tenant_id)
            if version is not None and version == self._version(db, tenant_id):
                return retriever
            # 数据变了(新增/编辑/删除),重建
            self.rebuild_index(db, tenant_id, reset_store=False)
            return self._retrievers.get(tenant_id)
        self.rebuild_index(db, tenant_id, reset_store=False)
        return self._retrievers.get(tenant_id)


# 全局单例
knowledge_service = KnowledgeService()
