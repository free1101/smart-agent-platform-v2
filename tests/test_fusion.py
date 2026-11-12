"""融合策略单元测试:RRF 与分数加权。"""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.fusion import normalize_scores, rrf_fuse, weighted_fuse


def test_rrf_basic_order():
    """两路都把 doc2 排第一,融合后 doc2 必须第一。"""
    fused = rrf_fuse([[2, 1, 3], [2, 3, 1]], k=60)
    assert max(fused, key=fused.get) == 2


def test_rrf_rank_weighting():
    """第 1 名得分必须高于第 2 名,且差距随 k 增大而缩小。"""
    small_k = rrf_fuse([[1, 2]], k=1)
    large_k = rrf_fuse([[1, 2]], k=1000)
    assert small_k[1] > small_k[2]
    assert (small_k[1] - small_k[2]) > (large_k[1] - large_k[2])


def test_rrf_weights():
    """权重为 0 的那一路不应贡献分数。"""
    fused = rrf_fuse([[1], [2]], weights=[1.0, 0.0], k=60)
    assert 1 in fused
    assert 2 not in fused


def test_rrf_empty():
    assert rrf_fuse([[], []]) == {}


def test_rrf_length_mismatch():
    with pytest.raises(ValueError):
        rrf_fuse([[1], [2]], weights=[1.0])


def test_normalize_minmax():
    arr = normalize_scores(np.array([2.0, 4.0, 6.0]))
    assert arr[0] == 0.0 and arr[-1] == 1.0


def test_normalize_constant():
    """全部同分时应返回全 0(表示这一路没有区分度),而不是除零。"""
    arr = normalize_scores(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(arr, 0.0)


def test_weighted_fuse_direction():
    sparse = np.array([10.0, 0.0])
    dense = np.array([0.1, 0.9])
    fused = weighted_fuse(sparse, dense, 0.4, 0.6)
    assert fused[1] > fused[0], "向量分数占 0.6,第二条应更高"
