"""检索层抽象接口:向量库 / 重排器 / 分块器。

为什么需要这层抽象:
- 样例项目的核心卖点之一是「乐高积木式替换」,业务代码只依赖接口;
- 本项目已有 BM25 与 numpy 两套实现,新增 Chroma 后需要一个统一入口,
  否则 `KnowledgeService` 里会堆满 if/else;
- 测试可以注入内存实现,不依赖网络与磁盘。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class EmbeddingIdentityMismatch(RuntimeError):
    """向量索引的 embedding 身份与当前配置不一致。

    触发场景:换了 embedding 模型或维度,却复用了旧索引目录。
    此时旧向量与新查询向量不在同一语义空间,检索结果不可信,
    必须显式报错(而不是静默返回错误的相似度)。
    """


@dataclass
class VectorHit:
    """向量检索的一条命中。"""

    doc_id: int
    score: float  # 余弦相似度,越大越相关
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """向量库接口:每个实例对应一个租户的一个 collection。"""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """写入或覆盖向量(id 相同则覆盖)。"""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """按 id 删除向量。"""

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """按向量相似度检索,支持 metadata 过滤。"""

    @abstractmethod
    def count(self) -> int:
        """当前 collection 的向量条数。"""

    @abstractmethod
    def reset(self) -> None:
        """清空当前 collection(重建索引前调用)。"""


class BaseReranker(ABC):
    """重排器接口:hits 为 `RetrieverHit`(含 content 与 score)。"""

    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list:
        """返回按相关度降序的新列表。"""


class BaseSplitter(ABC):
    """分块器接口。"""

    name: str = "base"

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """把长文本切成若干块。"""
