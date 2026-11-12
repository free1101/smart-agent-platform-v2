"""Cross-Encoder 精排:查询与文档拼接后一起送进模型打分。

与「双塔向量检索」的区别(面试常问):
- 双塔(Bi-Encoder):query 与 doc 各自编码成向量算余弦,可以预建索引,快但粗糙;
- 交叉编码(Cross-Encoder):[query, doc] 作为一条输入送进 Transformer,
  两段文字在注意力层就能互相看见,精度高但必须逐条现算,只能用于小候选集精排。
所以工程上是「粗排(召回 20~100 条)→ 精排(cross-encoder 打分)→ 取 top_k」。
"""

from __future__ import annotations

import logging
import threading

from app.rag.base import BaseReranker
from app.rag.fusion import normalize_scores

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    name = "cross_encoder"

    def __init__(self, model_name: str | None = None) -> None:
        from app.config.settings import settings

        self.model_name = model_name or settings.CROSS_ENCODER_MODEL
        self._model = None
        self._lock = threading.Lock()
        self._unavailable = False

    def _get_model(self):
        """懒加载模型:首次调用才下载/加载,避免拖慢服务启动。"""
        if self._model is not None or self._unavailable:
            return self._model
        with self._lock:
            if self._model is not None or self._unavailable:
                return self._model
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                logger.info("CrossEncoder 已加载: %s", self.model_name)
            except Exception as exc:  # noqa: BLE001 - 缺依赖/缺模型都回退,不阻断检索
                self._unavailable = True
                logger.warning(
                    "CrossEncoder 不可用(%s),重排回退为融合顺序。安装: pip install sentence-transformers",
                    exc,
                )
        return self._model

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list:
        if not hits:
            return []
        model = self._get_model()
        if model is None:
            # 明确回退:不伪造重排分数,保持融合序
            for h in hits:
                h.rerank_score = h.score
            return hits[:top_k] if top_k is not None else hits

        pairs = [[query, h.content] for h in hits]
        try:
            raw = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossEncoder 打分失败,回退融合序: %s", exc)
            for h in hits:
                h.rerank_score = h.score
            return hits[:top_k] if top_k is not None else hits

        import numpy as np

        normed = normalize_scores(np.asarray(raw, dtype=np.float64), "minmax")
        for h, score in zip(hits, normed):
            h.rerank_score = float(score)
        ranked = sorted(hits, key=lambda h: h.rerank_score, reverse=True)
        return ranked[:top_k] if top_k is not None else ranked
