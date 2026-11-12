"""重排序入口:委托给工厂创建的实现,保持 `Reranker().rerank()` 旧签名。

实现可切换(见 `RERANKER_PROVIDER`):
- `rule`(默认):召回分 + 关键词重叠 × 0.05,零依赖、完全可解释;
- `cross_encoder`:bge-reranker 之类的交叉编码模型,精度最高,需要 sentence-transformers;
- `llm`:一次调用让大模型对候选做 listwise 排序;
- `none`:关闭重排。

无论哪种实现,失败都回退为融合顺序并记 WARNING,不伪造分数。
"""

from __future__ import annotations

from app.rag.base import BaseReranker
from app.rag.factory import create_reranker


class Reranker:
    """精排器门面:输入初筛 hits,输出按相关度降序的 hits。"""

    def __init__(self, provider: str | None = None) -> None:
        self._impl: BaseReranker | None = create_reranker(provider)

    @property
    def provider(self) -> str:
        return self._impl.name if self._impl is not None else "none"

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list:
        if not hits:
            return []
        if self._impl is None:
            return hits[:top_k] if top_k is not None else hits
        return self._impl.rerank(query, hits, top_k)
