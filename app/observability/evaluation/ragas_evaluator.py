"""Ragas 适配(可选依赖)。

为什么做成可选:
- ragas 会拉入 datasets / langchain 生态的一大串依赖,和本项目现有的
  langchain-openai 版本很容易打架;
- 评估本身不该成为服务启动的硬依赖。装得上就用行业标准指标,
  装不上就退回自研 metrics + LLM judge,并在报告里明确写 `ragas: unavailable`。

安装(可选): pip install ragas datasets
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict:
    """用 Ragas 计算 faithfulness / answer_relevancy / context_precision。

    Returns:
        {"available": bool, "metrics": {...}, "error": str}
        任何一环境装不起来都返回 available=False,不抛异常。
    """
    if not ragas_available():
        return {"available": False, "metrics": {}, "error": "ragas 未安装"}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            from ragas.metrics import context_recall

            data["ground_truth"] = ground_truths
            metrics.append(context_recall)

        result = evaluate(Dataset.from_dict(data), metrics=metrics)
        scores = {k: float(v) for k, v in dict(result).items() if isinstance(v, (int, float))}
        return {"available": True, "metrics": scores, "error": ""}
    except Exception as exc:  # noqa: BLE001 - 版本冲突/打分失败都降级
        logger.warning("Ragas 评估失败,已降级为自研指标: %s", exc)
        return {"available": False, "metrics": {}, "error": str(exc)[:300]}
