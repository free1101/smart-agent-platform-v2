"""LLM listwise 精排:把候选列表整体交给模型,让它给出相关性排序。

为什么用 listwise 而不是逐条打分:
- 逐条打分需要 N 次调用,N 条候选就是 N 倍成本;
- listwise 一次调用即可让模型在候选中做「相对比较」,通常比绝对打分更稳。
代价是一次性把候选塞进上下文,所以候选数要截断(`RERANK_CANDIDATES`)。
"""

from __future__ import annotations

import json
import logging
import re

from app.rag.base import BaseReranker

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)

_SYSTEM = (
    "你是检索结果重排器。根据用户问题,判断每个候选知识的相关程度。\n"
    "只输出严格 JSON,不要任何解释,格式:{\"ranking\": [最相关序号, 次相关序号, ...]}\n"
    "序号必须来自给定候选,且不要重复;不相关的候选可以省略。"
)


class LLMReranker(BaseReranker):
    name = "llm"

    def __init__(self, max_candidates: int | None = None) -> None:
        from app.config.settings import settings

        self.max_candidates = max_candidates or settings.RERANK_CANDIDATES

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list:
        if not hits:
            return []
        candidates = hits[: self.max_candidates]
        tail = hits[len(candidates) :]

        listing = "\n".join(
            f"[{i}] {h.content[:300]}" for i, h in enumerate(candidates, start=1)
        )
        prompt = f"【用户问题】\n{query}\n\n【候选知识】\n{listing}\n\n请输出 ranking JSON:"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            from app.llm.provider import create_chat_model

            llm = create_chat_model(temperature=0)
            resp = llm.invoke(
                [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
            )
            order = self._parse_order(str(resp.content), len(candidates))
        except Exception as exc:  # noqa: BLE001 - 解析失败/网络失败都回退
            logger.warning("LLM 重排失败,回退融合序: %s", exc)
            order = []

        if not order:
            for h in hits:
                h.rerank_score = h.score
            return hits[:top_k] if top_k is not None else hits

        ranked = [candidates[i - 1] for i in order]
        # 未进入 ranking 的候选按原序接在后面,保证不丢结果
        used = set(order)
        ranked += [c for i, c in enumerate(candidates, start=1) if i not in used]
        ranked += tail
        for pos, h in enumerate(ranked):
            h.rerank_score = 1.0 - pos / max(1, len(ranked))
        return ranked[:top_k] if top_k is not None else ranked

    @staticmethod
    def _parse_order(text: str, size: int) -> list[int]:
        """从模型输出里抽出 ranking 数组,过滤越界与重复序号。"""
        match = _JSON_BLOCK.search(text or "")
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        ranking = payload.get("ranking") if isinstance(payload, dict) else None
        if not isinstance(ranking, list):
            return []
        order: list[int] = []
        for item in ranking:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= size and idx not in order:
                order.append(idx)
        return order
