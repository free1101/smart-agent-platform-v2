"""向量库实现:Chroma(持久化)与内存版(测试/降级)。

关键约束:
1. **必须显式传 embeddings**,绝不依赖 Chroma 默认的 ONNX 嵌入函数
   (它会触发模型下载,且维度与 text-embedding-v3 的 1024 不一致);
2. **embedding 身份校验**:索引目录里记录模型名与维度,不匹配时抛
   `EmbeddingIdentityMismatch`,而不是静默返回不可信的相似度;
3. metadata 过滤支持 category / source / is_active,供管理端按分类检索。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from app.rag.base import BaseVectorStore, EmbeddingIdentityMismatch, VectorHit

logger = logging.getLogger(__name__)

_IDENTITY_FILE = "embedding_identity.json"


def _collection_name(tenant_id: int) -> str:
    return f"kb_tenant_{tenant_id}"


def read_identity(persist_dir: Path) -> dict[str, Any] | None:
    """读取索引身份文件;不存在返回 None。"""
    path = Path(persist_dir) / _IDENTITY_FILE
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 embedding 身份文件失败(将按新索引处理): %s", exc)
        return None


def write_identity(persist_dir: Path, model: str, dimension: int, context: str = "") -> None:
    """写入/更新索引身份。只有第一次或维度变化时才落盘。"""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "dimension": dimension, "context": context}
    with (persist_dir / _IDENTITY_FILE).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def check_identity(persist_dir: Path, model: str, dimension: int | None = None) -> dict[str, Any] | None:
    """校验当前 embedding 配置是否与索引目录匹配。

    Returns:
        已存在的身份 dict;索引目录为空时返回 None。
    Raises:
        EmbeddingIdentityMismatch: 模型名不同,或已知维度与记录不同。
    """
    existing = read_identity(persist_dir)
    if existing is None:
        return None

    old_model = str(existing.get("model", ""))
    if old_model and model and old_model != model:
        raise EmbeddingIdentityMismatch(
            f"向量索引由 embedding 模型 {old_model} 构建,当前配置是 {model}。"
            f"两者的向量不在同一语义空间,请把 {persist_dir} 改名保留后重建索引。"
        )

    old_dim = existing.get("dimension")
    if dimension and old_dim and int(old_dim) != int(dimension):
        raise EmbeddingIdentityMismatch(
            f"向量索引维度为 {old_dim},当前 embedding 维度为 {dimension}。"
            f"请把 {persist_dir} 改名保留后重建索引。"
        )
    return existing


class ChromaVectorStore(BaseVectorStore):
    """基于 Chroma PersistentClient 的持久化向量库(一个租户一个 collection)。"""

    def __init__(
        self,
        tenant_id: int,
        persist_dir: str | Path | None = None,
        model: str = "",
        dimension: int | None = None,
    ) -> None:
        import chromadb  # 延迟导入:未安装 chromadb 时由工厂给出明确提示

        from app.config.settings import settings

        self.tenant_id = tenant_id
        self.persist_dir = Path(persist_dir or settings.CHROMA_DIR)
        self.model = model or settings.EMBEDDING_MODEL
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 身份校验放在建 collection 之前:不匹配时不要产生任何写入
        check_identity(self.persist_dir, self.model, dimension)

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._open_collection()

    def _open_collection(self):
        name = _collection_name(self.tenant_id)
        metadata = {"hnsw:space": "cosine"}
        try:
            # embedding_function=None 表示禁止 Chroma 自动嵌入,向量一律由调用方提供
            return self._client.get_or_create_collection(
                name=name, metadata=metadata, embedding_function=None
            )
        except TypeError:
            # 老版本 chromadb 不接受 embedding_function=None
            return self._client.get_or_create_collection(name=name, metadata=metadata)
        except Exception as exc:  # noqa: BLE001 - collection 冲突等
            logger.warning("打开 collection %s 失败,尝试重建: %s", name, exc)
            try:
                self._client.delete_collection(name)
            except Exception:  # noqa: BLE001
                pass
            return self._client.get_or_create_collection(name=name, metadata=metadata)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not ids:
            return
        dimension = len(embeddings[0]) if embeddings and embeddings[0] else None
        identity = read_identity(self.persist_dir)
        if identity is None and dimension:
            write_identity(self.persist_dir, self.model, dimension)
        elif identity and dimension and int(identity.get("dimension") or 0) not in (0, dimension):
            raise EmbeddingIdentityMismatch(
                f"索引维度 {identity.get('dimension')} 与写入向量维度 {dimension} 不一致,"
                f"请把 {self.persist_dir} 改名保留后重建索引。"
            )

        from app.config.settings import settings

        batch = max(1, int(settings.CHROMA_BATCH_SIZE))
        for start in range(0, len(ids), batch):
            end = start + batch
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=ids)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        total = self.count()
        if total <= 0 or not query_embedding:
            return []
        n = max(1, min(int(n_results), total))

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            result = self._collection.query(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 过滤条件写错不应让检索整体失败
            logger.warning("Chroma 查询失败(已去掉 metadata 过滤重试): %s", exc)
            kwargs.pop("where", None)
            result = self._collection.query(**kwargs)

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[VectorHit] = []
        for i, raw_id in enumerate(ids):
            try:
                doc_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            distance = float(dists[i]) if i < len(dists) else 1.0
            # cosine 空间下 distance = 1 - similarity
            score = max(0.0, min(1.0, 1.0 - distance))
            hits.append(
                VectorHit(
                    doc_id=doc_id,
                    score=score,
                    content=str(docs[i]) if i < len(docs) else "",
                    metadata=dict(metas[i] or {}) if i < len(metas) else {},
                )
            )
        return hits

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma count 失败: %s", exc)
            return 0

    def reset(self) -> None:
        try:
            self._client.delete_collection(_collection_name(self.tenant_id))
        except Exception as exc:  # noqa: BLE001 - 不存在时忽略
            logger.info("删除 collection 时忽略异常: %s", exc)
        self._collection = self._open_collection()


class InMemoryVectorStore(BaseVectorStore):
    """内存向量库:用于单元测试与 Chroma 不可用时的降级(语义与 Chroma 一致)。"""

    def __init__(self, tenant_id: int = 0) -> None:
        self.tenant_id = tenant_id
        self._ids: list[str] = []
        self._docs: dict[str, str] = {}
        self._metas: dict[str, dict[str, Any]] = {}
        self._vecs: dict[str, list[float]] = {}

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        for i, raw_id in enumerate(ids):
            if raw_id not in self._ids:
                self._ids.append(raw_id)
            self._docs[raw_id] = documents[i] if i < len(documents) else ""
            self._metas[raw_id] = metadatas[i] if i < len(metadatas) else {}
            self._vecs[raw_id] = embeddings[i] if i < len(embeddings) else []

    def delete(self, ids: list[str]) -> None:
        for raw_id in ids:
            if raw_id in self._ids:
                self._ids.remove(raw_id)
            self._docs.pop(raw_id, None)
            self._metas.pop(raw_id, None)
            self._vecs.pop(raw_id, None)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        if not query_embedding:
            return []
        q_norm = math.sqrt(sum(v * v for v in query_embedding)) or 1.0
        scored: list[VectorHit] = []
        for raw_id in self._ids:
            meta = self._metas.get(raw_id, {})
            if not _match_where(meta, where):
                continue
            vec = self._vecs.get(raw_id) or []
            if not vec:
                continue
            v_norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            sim = sum(a * b for a, b in zip(vec, query_embedding)) / (v_norm * q_norm)
            try:
                doc_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            scored.append(
                VectorHit(
                    doc_id=doc_id,
                    score=max(0.0, min(1.0, sim)),
                    content=self._docs.get(raw_id, ""),
                    metadata=dict(meta),
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[: max(1, int(n_results))]

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids.clear()
        self._docs.clear()
        self._metas.clear()
        self._vecs.clear()


def _match_where(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """极简 metadata 过滤:支持 {"k": v} 与 {"k": {"$eq"/"$in": ...}}。"""
    if not where:
        return True
    for key, expect in where.items():
        actual = meta.get(key)
        if isinstance(expect, dict):
            if "$eq" in expect and actual != expect["$eq"]:
                return False
            if "$in" in expect and actual not in (expect["$in"] or []):
                return False
        elif actual != expect:
            return False
    return True
