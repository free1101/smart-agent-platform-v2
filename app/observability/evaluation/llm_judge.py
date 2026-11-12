"""LLM-as-judge:评估「答案是否忠实于上下文」与「答案是否切题」。

两个指标:
- **Faithfulness(忠实度)**:答案里的每个事实是否都能在检索到的上下文里找到依据。
  分数低说明模型在编(幻觉),这是 RAG 最该盯的指标。
- **AnswerRelevancy(答案相关性)**:答案是否直接回应了用户问题,而不是答非所问。

诚实性约定:解析失败时返回 `judge_failed=True` 且分数为 None,**不给默认分**。
否则报告里的平均分会把「没评估到」伪装成「评估得很差/很好」。
"""

from __future__ import annotations

import json
import logging
import re

from app.config.settings import settings

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)

_SYSTEM = (
    "你是严格的 RAG 答案评审员。只输出严格 JSON,不要解释,格式:\n"
    '{"faithfulness": 0~1 的小数, "answer_relevancy": 0~1 的小数, "reason": "一句话依据"}\n'
    "评分标准:\n"
    "- faithfulness:答案中的事实是否都能在给定上下文找到依据。有编造内容就大幅扣分。\n"
    "- answer_relevancy:答案是否直接回答了用户问题。答非所问给低分。\n"
    "上下文不足以回答而答案明确说明「不知道」时,faithfulness 给 1.0。"
)


def judge_available() -> bool:
    return settings.has_llm_key


def judge_answer(query: str, context: str, answer: str) -> dict:
    """对一条答案打分。

    Returns:
        {"faithfulness": float|None, "answer_relevancy": float|None,
         "reason": str, "judge_failed": bool}
    """
    if not answer.strip():
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "reason": "答案为空",
            "judge_failed": False,
        }

    if not judge_available():
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "reason": "未配置 LLM_API_KEY,跳过判定",
            "judge_failed": True,
        }

    prompt = (
        f"【用户问题】\n{query}\n\n"
        f"【检索上下文】\n{context[:4000]}\n\n"
        f"【待评答案】\n{answer[:2000]}\n\n请输出评分 JSON:"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.llm.provider import create_chat_model

        llm = create_chat_model(temperature=0)
        resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
        return _parse(str(resp.content))
    except Exception as exc:  # noqa: BLE001 - 评估失败不能搞挂整轮评测
        logger.warning("LLM 判定失败: %s", exc)
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "reason": f"judge 调用失败: {exc}"[:200],
            "judge_failed": True,
        }


def _parse(text: str) -> dict:
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "reason": "模型输出不是 JSON",
            "judge_failed": True,
        }
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "reason": "JSON 解析失败",
            "judge_failed": True,
        }

    return {
        "faithfulness": _clamp(payload.get("faithfulness")),
        "answer_relevancy": _clamp(payload.get("answer_relevancy")),
        "reason": str(payload.get("reason", ""))[:300],
        "judge_failed": False,
    }


def _clamp(value) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
