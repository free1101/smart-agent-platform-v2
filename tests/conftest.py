"""pytest 共享配置。

1. 确保项目根目录在 sys.path;
2. 提供离线 Embedding 测试替身:生产代码已移除本地假模型(缺 key 即 fail-fast,
   `HybridRetriever` 构造会调用真实 API embedding),单测不应依赖真实 key/网络,
   故在测试期把 `app.rag.hybrid_retriever.create_embeddings` 替换为基于 token
   哈希的确定性向量替身(仅测试隔离用,不进入任何生产路径)。
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 测试环境固化:默认不使用 Chroma 落盘(避免污染 data/chroma),
# 鉴权默认放行(需要校验 503/401 的用例会自己覆盖这两个字段)。
os.environ.setdefault("VECTOR_STORE", "memory")
os.environ.setdefault("ALLOW_UNAUTHENTICATED", "true")

from app.rag.text_splitter import tokenize  # noqa: E402


class _HashStubEmbeddings(Embeddings):
    """离线单测替身:与早期 LocalHashEmbeddings 同思路的确定性哈希向量。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text).tolist()


@pytest.fixture(autouse=True)
def _offline_embeddings(monkeypatch):
    """把 hybrid_retriever 中的 create_embeddings 换成测试替身,单测无需真实 key。"""
    import app.rag.hybrid_retriever as hr

    monkeypatch.setattr(hr, "create_embeddings", lambda: _HashStubEmbeddings())


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    """单测不打真实 LLM:把 has_llm_key 固定为 False。

    这样即使开发者本地 .env 配了 key,意图分类也走规则、业务节点走 business_node
    规则流程,测试结果可复现(否则同一断言会因模型输出不同而随机失败)。
    """
    from app.config.settings import Settings

    monkeypatch.setattr(Settings, "has_llm_key", property(lambda self: False))


@pytest.fixture
def mem_db(monkeypatch):
    """内存 SQLite 会话:建好全部表,并把全局 SessionLocal 指向它。

    为什么连 SessionLocal 一起 patch:业务代码(trace 落库、MCP 工具)内部
    直接 `from app.db.database import SessionLocal`,不 patch 就会写到开发库。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import app.db.database as database_module
    from app.db import models  # noqa: F401  注册模型

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database_module.Base.metadata.create_all(engine)
    test_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(database_module, "SessionLocal", test_session)

    db = test_session()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def _reset_knowledge_cache():
    """清空 knowledge_service 的租户索引缓存,避免用例之间互相污染。"""
    from app.services.knowledge_service import knowledge_service

    def _clear() -> None:
        knowledge_service._retrievers.clear()  # noqa: SLF001
        knowledge_service._index_versions.clear()  # noqa: SLF001

    _clear()
    yield
    _clear()
