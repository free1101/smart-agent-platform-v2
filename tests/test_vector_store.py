"""向量库测试:内存实现的语义正确性 + embedding 身份校验。"""

from __future__ import annotations

import json

import pytest

from app.rag.base import EmbeddingIdentityMismatch
from app.rag.vector_store import (
    InMemoryVectorStore,
    check_identity,
    read_identity,
    write_identity,
)


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(tenant_id=1)
    store.upsert(
        ids=["1", "2", "3"],
        documents=["推拿价格120元", "营业时间上午9点", "停车费每小时8元"],
        metadatas=[
            {"category": "服务项目", "source": "seed", "is_active": True},
            {"category": "营业时间", "source": "seed", "is_active": True},
            {"category": "门店信息", "source": "upload:a.pdf", "is_active": True},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
    )
    return store


def test_upsert_and_count():
    store = _store()
    assert store.count() == 3


def test_query_returns_nearest_first():
    store = _store()
    hits = store.query([1.0, 0.0], n_results=2)
    assert len(hits) == 2
    assert hits[0].doc_id == 1
    assert hits[0].score >= hits[1].score


def test_query_metadata_filter():
    store = _store()
    hits = store.query([1.0, 1.0], n_results=10, where={"category": {"$eq": "门店信息"}})
    assert [h.doc_id for h in hits] == [3]


def test_query_filter_in():
    store = _store()
    hits = store.query(
        [1.0, 1.0], n_results=10, where={"category": {"$in": ["服务项目", "营业时间"]}}
    )
    assert {h.doc_id for h in hits} == {1, 2}


def test_upsert_overwrites():
    store = _store()
    store.upsert(["1"], ["改后的内容"], [{"category": "服务项目"}], [[0.0, 1.0]])
    assert store.count() == 3
    hits = store.query([0.0, 1.0], n_results=1)
    assert hits[0].content == "改后的内容"


def test_delete():
    store = _store()
    store.delete(["1", "3"])
    assert store.count() == 1
    assert store.query([1.0, 0.0], 5)[0].doc_id == 2


def test_reset():
    store = _store()
    store.reset()
    assert store.count() == 0
    assert store.query([1.0, 0.0], 3) == []


def test_identity_roundtrip(tmp_path):
    write_identity(tmp_path, "text-embedding-v3", 1024)
    identity = read_identity(tmp_path)
    assert identity["model"] == "text-embedding-v3"
    assert identity["dimension"] == 1024
    assert (tmp_path / "embedding_identity.json").exists()


def test_identity_model_mismatch(tmp_path):
    write_identity(tmp_path, "text-embedding-v3", 1024)
    with pytest.raises(EmbeddingIdentityMismatch):
        check_identity(tmp_path, "bge-large-zh", 1024)


def test_identity_dimension_mismatch(tmp_path):
    write_identity(tmp_path, "text-embedding-v3", 1024)
    with pytest.raises(EmbeddingIdentityMismatch):
        check_identity(tmp_path, "text-embedding-v3", 768)


def test_identity_match(tmp_path):
    write_identity(tmp_path, "text-embedding-v3", 1024)
    assert check_identity(tmp_path, "text-embedding-v3", 1024) is not None


def test_identity_missing_is_new_index(tmp_path):
    assert check_identity(tmp_path, "text-embedding-v3", 1024) is None


def test_identity_broken_file(tmp_path):
    (tmp_path / "embedding_identity.json").write_text("{not json", encoding="utf-8")
    assert read_identity(tmp_path) is None


def test_chroma_persist_and_query(tmp_path):
    """Chroma 落盘后重新打开仍能检索(验证持久化,不只是内存)。"""
    pytest.importorskip("chromadb", reason="未安装 chromadb")
    from app.rag.vector_store import ChromaVectorStore

    store = ChromaVectorStore(tenant_id=9, persist_dir=tmp_path, model="stub", dimension=2)
    store.reset()
    store.upsert(
        ["1", "2"],
        ["价格120元", "停车8元"],
        [{"category": "服务项目"}, {"category": "门店信息"}],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    assert store.count() == 2

    reopened = ChromaVectorStore(tenant_id=9, persist_dir=tmp_path, model="stub", dimension=2)
    hits = reopened.query([1.0, 0.0], n_results=1)
    assert hits and hits[0].doc_id == 1
    assert hits[0].score > 0.5

    filtered = reopened.query([1.0, 1.0], n_results=5, where={"category": {"$eq": "门店信息"}})
    assert [h.doc_id for h in filtered] == [2]


def test_identity_file_is_valid_json(tmp_path):
    write_identity(tmp_path, "m", 3, context="v1")
    data = json.loads((tmp_path / "embedding_identity.json").read_text(encoding="utf-8"))
    assert data["context"] == "v1"
