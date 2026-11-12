"""评测指标测试:口径正确性与「无标注不臆造」的降级行为。"""

from __future__ import annotations

from app.observability.evaluation.metrics import (
    aggregate,
    context_precision_at_k,
    evaluate_case,
    hit_at_k,
    mrr_at_k,
    recall_at_k,
)


def _results(*items: tuple[int, str, str]) -> list[dict]:
    return [
        {"doc_id": doc_id, "category": category, "content": content, "score": 0.5}
        for doc_id, category, content in items
    ]


def test_hit_at_k():
    results = _results((1, "营业时间", "9点到22点"), (2, "服务项目", "肩颈80元"))
    assert hit_at_k(results, "服务项目", 2) is True
    assert hit_at_k(results, "服务项目", 1) is False
    assert hit_at_k(results, "门店信息", 2) is False


def test_mrr_at_k():
    results = _results((1, "营业时间", "x"), (2, "服务项目", "y"))
    assert mrr_at_k(results, "营业时间", 2) == 1.0
    assert mrr_at_k(results, "服务项目", 2) == 0.5
    assert mrr_at_k(results, "门店信息", 2) == 0.0


def test_recall_at_k():
    results = _results((1, "服务项目", "肩颈推拿80元30分钟"))
    assert recall_at_k(results, ["80元", "30分钟"], 1) == 1.0
    assert recall_at_k(results, ["80元", "120元"], 1) == 0.5
    # 没有标注时不臆造相关性
    assert recall_at_k(results, [], 1) is None


def test_context_precision():
    results = _results(
        (1, "服务项目", "肩颈推拿80元"),
        (2, "门店信息", "停车每小时8元"),
    )
    # 只有第一条命中关键词
    assert context_precision_at_k(results, ["肩颈"], 2) == 0.5
    assert context_precision_at_k(results, [], 2) is None


def test_evaluate_case_shape():
    case = {
        "query": "肩颈多少钱",
        "expected_category": "服务项目",
        "expected_keywords": ["80元"],
        "difficulty": "easy",
    }
    row = evaluate_case(_results((1, "服务项目", "肩颈80元")), case, k=3)
    assert row["hit"] is True
    assert row["mrr"] == 1.0
    assert row["recall"] == 1.0
    assert row["difficulty"] == "easy"
    assert row["doc_ids"] == [1]


def test_aggregate_skips_missing_dimensions():
    rows = [
        evaluate_case(
            _results((1, "服务项目", "肩颈80元")),
            {"query": "a", "expected_category": "服务项目", "expected_keywords": ["80元"]},
        ),
        evaluate_case(
            _results((2, "门店信息", "停车8元")),
            {"query": "b", "expected_category": "门店信息"},
        ),
    ]
    metrics = aggregate(rows)
    assert metrics["total"] == 2
    assert metrics["hit_rate"] == 1.0
    assert metrics["recall_cases"] == 1
    assert metrics["recall"] == 1.0
    assert metrics["precision_cases"] == 1


def test_aggregate_empty():
    metrics = aggregate([])
    assert metrics["total"] == 0
    assert metrics["recall"] is None
