"""规则精排:召回分 + 关键词重叠数 × 0.05。

这是项目原有行为,保留为默认实现与降级兜底:
- 不依赖任何模型,零延迟、零成本;
- 效果有限,但比「完全不动」强,且完全可解释。
"""

from __future__ import annotations

from app.rag.base import BaseReranker
from app.rag.text_splitter import tokenize


class RuleReranker(BaseReranker):
    name = "rule"

    def __init__(self, overlap_bonus: float = 0.05) -> None:
        self.overlap_bonus = overlap_bonus

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list:
        if not hits:
            return []
        q_tokens = set(tokenize(query))
        scored = []
        for h in hits:
            overlap = len(q_tokens & set(tokenize(h.content)))
            final = h.score + overlap * self.overlap_bonus
            h.rerank_score = final
            scored.append((final, h))
        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [h for _, h in scored]
        return ranked[:top_k] if top_k is not None else ranked
