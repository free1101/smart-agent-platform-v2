"""检索层工厂:按配置创建向量库 / 重排器 / 分块器。

工厂的价值在于「调用方只认接口」:
- `KnowledgeService` 不再关心底层是 Chroma 还是内存;
- 换 embedding 模型、换重排策略只改 `.env`,不改业务代码;
- 新增实现只需在注册表加一行。
"""

from __future__ import annotations

import logging

from app.config.settings import settings
from app.rag.base import BaseReranker, BaseSplitter, BaseVectorStore
from app.rag.text_splitter import split_text
from app.rag.vector_store import ChromaVectorStore, InMemoryVectorStore

logger = logging.getLogger(__name__)


class TextSplitter(BaseSplitter):
    """`split_text()` 的接口化包装(保持原有切块算法不变)。"""

    name = "recursive"

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or settings.INGEST_CHUNK_SIZE
        self.overlap = overlap if overlap is not None else settings.INGEST_CHUNK_OVERLAP

    def split(self, text: str) -> list[str]:
        return split_text(text, self.chunk_size, self.overlap)


def create_vector_store(
    tenant_id: int,
    model: str | None = None,
    dimension: int | None = None,
) -> BaseVectorStore:
    """创建指定租户的向量库实例。

    Raises:
        EmbeddingIdentityMismatch: 索引目录与当前 embedding 配置不匹配(必须人工处理)。
    """
    backend = (settings.VECTOR_STORE or "chroma").lower()
    if backend in {"memory", "numpy", "none"}:
        return InMemoryVectorStore(tenant_id)

    try:
        return ChromaVectorStore(
            tenant_id=tenant_id,
            persist_dir=settings.CHROMA_DIR,
            model=model or settings.EMBEDDING_MODEL,
            dimension=dimension,
        )
    except ImportError as exc:
        logger.error(
            "chromadb 未安装,向量库降级为内存实现(重启后索引丢失)。"
            "安装: pip install chromadb。原因: %s",
            exc,
        )
        return InMemoryVectorStore(tenant_id)


_RERANKERS: dict[str, type[BaseReranker]] = {}


def _register_rerankers() -> dict[str, type[BaseReranker]]:
    """延迟注册:避免 import 时就拉起 sentence_transformers 等重依赖。"""
    global _RERANKERS
    if _RERANKERS:
        return _RERANKERS

    from app.rag.rerankers.cross_encoder import CrossEncoderReranker
    from app.rag.rerankers.llm import LLMReranker
    from app.rag.rerankers.rule import RuleReranker

    _RERANKERS = {
        "rule": RuleReranker,
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker,
    }
    return _RERANKERS


def create_reranker(provider: str | None = None) -> BaseReranker | None:
    """创建重排器;`none` 表示不重排,返回 None。"""
    name = (provider or settings.RERANKER_PROVIDER or "rule").lower()
    if name in {"none", "off", "false"}:
        return None

    registry = _register_rerankers()
    cls = registry.get(name)
    if cls is None:
        logger.warning("未知重排器 %s,回退为 rule", name)
        cls = registry["rule"]
    return cls()


def create_splitter(chunk_size: int | None = None, overlap: int | None = None) -> BaseSplitter:
    return TextSplitter(chunk_size, overlap)
