"""检索指标:命中率 / 召回率 / MRR / 上下文精确度。

口径说明(必须写清楚,否则报告没法横向比较):
- **Hit@k**:Top-k 里是否有一条的分类等于 `expected_category`。最宽松。
- **Recall@k**:`expected_keywords` 中有多少比例出现在 Top-k 的正文里。
  比 Hit 严格,因为它要求「答对了,而且关键事实也在」。
- **MRR**:第一条分类命中结果的倒数排名(1/rank),衡量「排得够不够前」。
- **ContextPrecision@k**:Top-k 中「至少命中一个期望关键词」的条目占比,
  衡量「召回的上下文里有多少是有用的」,防止塞一堆无关内容。

`relevant_doc_ids` 为空时不臆造相关性:相关维度返回 None,报告里标 `not_applicable`。
"""

from __future__ import annotations

from typing import Any


def _content_of(results: list[dict]) -> str:
    return "\n".join(str(r.get("content", "")) for r in results)


def hit_at_k(results: list[dict], expected_category: str, k: int = 3) -> bool:
    if not expected_category:
        return False
    return any(str(r.get("category", "")) == expected_category for r in results[:k])


def mrr_at_k(results: list[dict], expected_category: str, k: int = 3) -> float:
    if not expected_category:
        return 0.0
    for rank, item in enumerate(results[:k], start=1):
        if str(item.get("category", "")) == expected_category:
            return 1.0 / rank
    return 0.0


def recall_at_k(results: list[dict], expected_keywords: list[str], k: int = 3) -> float | None:
    """期望关键词在 Top-k 正文中的覆盖率;无关键词标注时返回 None。"""
    keywords = [kw for kw in (expected_keywords or []) if kw]
    if not keywords:
        return None
    text = _content_of(results[:k])
    hit = sum(1 for kw in keywords if kw in text)
    return hit / len(keywords)


def context_precision_at_k(
    results: list[dict], expected_keywords: list[str], k: int = 3
) -> float | None:
    keywords = [kw for kw in (expected_keywords or []) if kw]
    if not keywords or not results:
        return None
    top = results[:k]
    useful = sum(1 for r in top if any(kw in str(r.get("content", "")) for kw in keywords))
    return useful / len(top)


def evaluate_case(results: list[dict], case: dict[str, Any], k: int = 3) -> dict[str, Any]:
    """对单条用例计算全部指标。"""
    expected_category = str(case.get("expected_category", ""))
    expected_keywords = list(case.get("expected_keywords") or [])
    hit_categories = [str(r.get("category", "")) for r in results[:k]]

    return {
        "query": case.get("query", ""),
        "difficulty": case.get("difficulty", "normal"),
        "expected_category": expected_category,
        "top1_category": hit_categories[0] if hit_categories else "",
        "hit_categories": hit_categories,
        "hit": hit_at_k(results, expected_category, k),
        "mrr": round(mrr_at_k(results, expected_category, k), 4),
        "recall": _round(recall_at_k(results, expected_keywords, k)),
        "context_precision": _round(context_precision_at_k(results, expected_keywords, k)),
        "doc_ids": [r.get("doc_id") for r in results[:k]],
        "scores": [r.get("score") for r in results[:k]],
        "contents": [str(r.get("content", ""))[:200] for r in results[:k]],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总多条的指标;None 的维度不参与平均,单独标注适用条数。"""
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "hit_rate": 0.0,
            "recall": None,
            "mrr": 0.0,
            "context_precision": None,
            "recall_cases": 0,
            "precision_cases": 0,
        }

    recalls = [r["recall"] for r in rows if r.get("recall") is not None]
    precisions = [r["context_precision"] for r in rows if r.get("context_precision") is not None]
    return {
        "total": total,
        "hit_rate": round(sum(1 for r in rows if r["hit"]) / total, 4),
        "recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "recall_cases": len(recalls),
        "mrr": round(sum(float(r["mrr"]) for r in rows) / total, 4),
        "context_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
        "precision_cases": len(precisions),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)
