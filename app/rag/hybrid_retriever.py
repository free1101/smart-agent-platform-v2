"""混合检索器:BM25(稀疏)+ 稠密向量,支持 RRF 与分数加权两种融合。

相比纯向量检索的优势:
- BM25 对关键词精确匹配很稳(如价格、时间数字);
- 向量检索对语义近义表达敏感(如「开门」→「营业时间」)。

两种融合策略:
- `rrf`(默认):只看排名,不要求两路分数量纲一致,鲁棒;
- `weighted`:分数归一化后加权,保留分数强度(项目旧行为,可用配置切回)。

两种向量后端:
- 注入 `vector_store`(Chroma):向量持久化,重启不重建;
- 不注入(默认):numpy 内存矩阵,供单测与离线脚本使用。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from app.config.settings import settings
from app.llm.provider import create_embeddings
from app.rag.base import BaseVectorStore
from app.rag.fusion import normalize_scores, rrf_fuse, weighted_fuse
from app.rag.text_splitter import tokenize

logger = logging.getLogger(__name__)


@dataclass
class RetrieverHit:
    """一条检索命中结果。"""

    doc_id: int
    content: str
    category: str = ""
    score: float = 0.0
    sources: list[str] = field(default_factory=list)
    rerank_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "category": self.category,
            "score": round(self.score, 4),
            "source": "|".join(self.sources),
            "rerank_score": round(self.rerank_score, 4),
        }


class HybridRetriever:
    """对单个租户的文档集合构建双索引,并提供混合检索。

    用法:
        retriever = HybridRetriever()
        retriever.build(docs)          # docs: list[dict] 含 id/content/category/keywords
        hits = retriever.search(query, top_k=5)
    """

    def __init__(
        self,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
        vector_store: BaseVectorStore | None = None,
        fusion: str | None = None,
        tenant_id: int = 0,
    ):
        # 关键词加权倍数:提高 BM25 中 keywords 的命中优先级
        self.keyword_boost = 3
        self.bm25_weight = (
            settings.RETRIEVER_WEIGHT_BM25 if bm25_weight is None else bm25_weight
        )
        self.vector_weight = (
            settings.RETRIEVER_WEIGHT_VECTOR if vector_weight is None else vector_weight
        )
        self.fusion = (fusion or settings.FUSION_STRATEGY or "rrf").lower()
        self.rrf_k = settings.RRF_K
        self.candidates = settings.RETRIEVER_CANDIDATES
        self.tenant_id = tenant_id

        self._docs: list[dict] = []
        self._pos: dict[int, int] = {}  # doc_id -> 文档在 _docs 中的下标
        self._bm25: BM25Okapi | None = None
        self._index: np.ndarray | None = None  # (n, dim) 归一化向量(仅内存模式)
        self._embeddings = create_embeddings()
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # 构建索引
    # ------------------------------------------------------------------
    def build(self, docs: list[dict], reset_store: bool = False) -> None:
        """docs: [{"id": int, "content": str, "category": str, "keywords": [...]}]"""
        self._docs = list(docs)
        self._pos = {
            int(d.get("id", 0)): i for i, d in enumerate(self._docs) if d.get("id") is not None
        }
        self._build_bm25()
        if not self._docs:
            self._index = None
            return

        # 向量索引:对正文生成 embedding
        vectors = self._embeddings.embed_documents([d.get("content", "") for d in self._docs])
        if self._vector_store is not None:
            if reset_store:
                self._vector_store.reset()
            self._upsert_to_store(self._docs, vectors)
            self._index = None
        else:
            mat = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self._index = mat / np.where(norms == 0, 1.0, norms)

    def append_docs(self, docs: list[dict]) -> None:
        """增量追加文档:只对新增/更新部分重新算 embedding。

        摄取一个 200 页 PDF 时,如果每次都全量重建,代价随文档数线性增长;
        这里只 embed 新块,BM25 因为本身是内存索引、构建在毫秒级,直接重建。
        """
        if not docs:
            return
        if self._vector_store is None:
            # 内存模式无法增量拼矩阵,退回全量构建
            merged = {int(d.get("id", 0)): d for d in self._docs}
            for d in docs:
                merged[int(d.get("id", 0))] = d
            self.build(list(merged.values()))
            return

        for d in docs:
            doc_id = int(d.get("id", 0))
            if doc_id in self._pos:
                self._docs[self._pos[doc_id]] = d
            else:
                self._pos[doc_id] = len(self._docs)
                self._docs.append(d)
        self._build_bm25()
        vectors = self._embeddings.embed_documents([d.get("content", "") for d in docs])
        self._upsert_to_store(docs, vectors)

    def remove_docs(self, doc_ids: list[int]) -> None:
        """删除文档(同时清理向量)。"""
        ids = {int(i) for i in doc_ids}
        if not ids:
            return
        self._docs = [d for d in self._docs if int(d.get("id", -1)) not in ids]
        self._pos = {int(d.get("id", 0)): i for i, d in enumerate(self._docs)}
        self._build_bm25()
        if self._vector_store is not None:
            self._vector_store.delete([str(i) for i in ids])
        elif self._docs:
            self.build(self._docs)
        else:
            self._index = None

    def _build_bm25(self) -> None:
        """重建 BM25 索引(正文 + 关键词重复加权)。"""
        if not self._docs:
            self._bm25 = None
            return
        corpus: list[str] = []
        for d in self._docs:
            text = d.get("content", "")
            kws = d.get("keywords") or []
            kw_text = (" " + " ".join(kws)) * self.keyword_boost
            corpus.append(text + kw_text)
        # b=0.5 降低对长文档的长度惩罚(长知识条目也能被公平检索)
        self._bm25 = BM25Okapi([tokenize(t) for t in corpus], k1=1.5, b=0.5)

    def _upsert_to_store(self, docs: list[dict], vectors: list[list[float]]) -> None:
        if self._vector_store is None:
            return
        ids = [str(d.get("id", i)) for i, d in enumerate(docs)]
        metadatas = [
            {
                "category": d.get("category", "") or "",
                "source": d.get("source", "") or "",
                "is_active": bool(d.get("is_active", True)),
                "content_hash": d.get("content_hash", "") or "",
                "chunk_index": int(d.get("chunk_index", 0) or 0),
                "page": int(d.get("page", 0) or 0),
                "doc_group": d.get("doc_group", "") or "",
            }
            for d in docs
        ]
        self._vector_store.upsert(ids, [d.get("content", "") for d in docs], metadatas, vectors)

    def rebuild_docs(self, doc_ids: list[int] | None = None) -> None:
        """按 doc_ids 重建(为空则全量)。doc_ids 未命中时忽略。"""
        if doc_ids is None:
            return self.build(self._docs, reset_store=True)
        keep = [d for d in self._docs if d.get("id") in set(doc_ids)]
        self.build(keep, reset_store=True)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[RetrieverHit]:
        if not self._docs or self._bm25 is None:
            return []

        from app.observability.trace import record_stage

        n = len(self._docs)
        n_candidates = min(max(top_k * 4, self.candidates), n)

        # 1) 稀疏召回
        started = time.perf_counter()
        bm25_scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        sparse_order = np.argsort(-bm25_scores)[:n_candidates]
        sparse_rank = [int(i) for i in sparse_order if bm25_scores[i] > 0]
        record_stage(
            "bm25",
            candidates=len(sparse_rank),
            top_ids=[self._doc_id(i) for i in sparse_rank[:5]],
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

        # 2) 稠密召回
        started = time.perf_counter()
        q_vec = np.asarray(self._embeddings.embed_query(query), dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0:
            q_vec = q_vec / q_norm

        dense_scores = np.zeros(n, dtype=np.float64)
        if self._vector_store is not None:
            vhits = self._vector_store.query(
                q_vec.tolist(), n_results=n_candidates, where=where
            )
            dense_rank: list[int] = []
            for hit in vhits:
                pos = self._pos.get(int(hit.doc_id))
                if pos is None:
                    continue
                dense_scores[pos] = hit.score
                dense_rank.append(pos)
        else:
            if self._index is None:
                return []
            dense_scores = np.clip((self._index @ q_vec).astype(np.float64), 0.0, 1.0)
            dense_order = np.argsort(-dense_scores)[:n_candidates]
            dense_rank = [int(i) for i in dense_order if dense_scores[i] > 0]

        record_stage(
            "dense",
            candidates=len(dense_rank),
            top_ids=[self._doc_id(i) for i in dense_rank[:5]],
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            backend="chroma" if self._vector_store is not None else "numpy",
        )

        # 3) 融合
        started = time.perf_counter()
        overlap = len(set(sparse_rank) & set(dense_rank))
        if self.fusion == "weighted":
            fused_arr = weighted_fuse(
                bm25_scores,
                dense_scores,
                sparse_weight=self.bm25_weight,
                dense_weight=self.vector_weight,
                normalize="max",
            )
            fused: dict[int, float] = {i: float(fused_arr[i]) for i in range(n)}
        else:
            fused = rrf_fuse(
                [sparse_rank, dense_rank],
                weights=[self.bm25_weight, self.vector_weight],
                k=self.rrf_k,
            )
        record_stage(
            "fuse",
            strategy=self.fusion,
            rrf_k=self.rrf_k if self.fusion != "weighted" else None,
            sparse_candidates=len(sparse_rank),
            dense_candidates=len(dense_rank),
            overlap=overlap,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

        # 4) 取 top_k(按融合分降序;同分按 doc_id 稳定排序)
        order = sorted(fused.items(), key=lambda kv: (-kv[1], self._doc_id(kv[0])))
        hits: list[RetrieverHit] = []
        for i, score in order:
            if score <= 0:
                continue
            hits.append(self._to_hit(i, score))
            if len(hits) >= top_k:
                break
        return hits

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _doc_id(self, pos: int) -> int:
        try:
            return int(self._docs[pos].get("id", pos))
        except (IndexError, TypeError, ValueError):
            return pos

    def _to_hit(self, pos: int, score: float) -> RetrieverHit:
        d = self._docs[pos]
        return RetrieverHit(
            doc_id=int(d.get("id", pos)),
            content=d.get("content", ""),
            category=d.get("category", ""),
            score=float(score),
            sources=[d.get("source", "")] if d.get("source") else [],
            rerank_score=float(score),
        )

    def doc_count(self) -> int:
        if self._vector_store is not None:
            return self._vector_store.count()
        return len(self._docs)


def normalize_sparse(scores: np.ndarray) -> np.ndarray:
    """对外暴露的归一化工具(评测/调试时可能单独用到)。"""
    return normalize_scores(scores, "max")
