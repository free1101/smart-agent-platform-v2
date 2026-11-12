"""多路召回融合:RRF(倒数排名融合)与分数归一化加权。

两种策略的差别(面试常问):
- **RRF**:只看「排名」不看分数。第 r 名贡献 weight/(k+r)。好处是不同分数量纲
  (BM25 无上界、余弦在 [0,1])不需要归一化就能直接合并,坏处是丢掉了
  「一字不差命中」与「勉强命中」的分数差距。
- **weighted**:先各自归一化再加权,保留分数强度,但对 BM25 的分数分布很敏感
  (一条文档拉高最大值会把其余文档压扁)。
本项目默认 RRF,配置 `FUSION_STRATEGY=weighted` 可回到旧行为。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def rrf_fuse(
    rankings: Iterable[Iterable[int]],
    weights: list[float] | None = None,
    k: int = 60,
) -> dict[int, float]:
    """倒数排名融合。

    Args:
        rankings: 每一路的文档下标序列,已按相关度降序(例:[[3,1,7], [1,3,9]])。
        weights: 每一路的权重;缺省全部 1.0。
        k: 平滑常数,通常取 60。k 越大,靠前名次的优势越平缓。
    Returns:
        {文档下标: 融合分},分越高越相关。
    """
    rankings = [list(r) for r in rankings]
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 长度必须与 rankings 路数一致")

    k = max(1, int(k))
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        if weight <= 0:
            # 权重为 0 表示这一路不参与融合,不应留下 0 分条目干扰下游
            continue
        for rank, doc_idx in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)
    return fused


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """把一路分数归一到 [0,1]。

    minmax 在「所有分数相同」时返回全 0(而不是除零),含义是这一路没有区分度。
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    if method == "none":
        return arr
    if method == "max":
        top = float(arr.max())
        return arr / top if top > 0 else np.zeros_like(arr)
    low, high = float(arr.min()), float(arr.max())
    if high - low <= 1e-12:
        return np.zeros_like(arr)
    return (arr - low) / (high - low)


def weighted_fuse(
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
    sparse_weight: float = 0.4,
    dense_weight: float = 0.6,
    normalize: str = "minmax",
) -> np.ndarray:
    """分数归一化后加权相加(项目旧行为,保留以便 A/B 对比)。"""
    sparse = normalize_scores(sparse_scores, normalize)
    dense = np.clip(np.asarray(dense_scores, dtype=np.float64), 0.0, 1.0)
    return sparse_weight * sparse + dense_weight * dense
