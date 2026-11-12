"""上下文工程: 控制进入 LLM 的 token 用量。

解决真实系统中最常见的问题——对话一长,塞进 LLM 的历史会:
  1. 超过上下文窗口 → 直接报错;
  2. 稀释模型注意力,让它"抓不住重点"。

策略是经典的 "摘要 + 截断"(summarization + truncation),但做了零破坏设计:
  - 不物理删除 messages(删除会破坏 business 节点的跨轮预约补全,它依赖全部用户历史);
  - 超窗时把"最旧的对话"折叠成一段 summary 存入 state["summary"];
  - 咨询节点组装提示词时,用 summary + 最近 N 轮 替代"全量历史",
    从而在"进入 LLM 的那一段"上完成截断。

效果: 对话 > MAX_ROUNDS 轮后,旧历史不再整体进入 LLM,取而代之的是一段摘要。
"""

from __future__ import annotations

from typing import Any

from app.config.settings import settings
from app.llm.provider import create_chat_model

# 对话轮次窗口: 超过则触发压缩(一轮 = 一条用户消息)
MAX_ROUNDS = 8
# 触发压缩后, 保留进 LLM 的"最近轮数"
MAX_RECENT_ROUNDS = 4
# 规则摘要时, 每条旧用户消息最多保留的字符数
RULE_SUMMARY_LEN = 30


def _human_texts(messages: list[Any]) -> list[str]:
    """提取全部用户消息文本(兼容 BaseMessage 与 dict 两种形式)。"""
    texts = []
    for m in messages:
        if isinstance(m, dict):
            if m.get("type") == "human":
                texts.append(str(m.get("content", "")))
        elif getattr(m, "type", "") == "human":
            texts.append(str(getattr(m, "content", "")))
    return texts


def should_compress(messages: list[Any]) -> bool:
    """对话轮数是否超过窗口(是否需要折叠旧历史)。"""
    return len(_human_texts(messages)) > MAX_ROUNDS


def _rule_summary(texts: list[str]) -> str:
    """无 LLM key 时的降级摘要: 每条旧消息取前 N 字拼成要点。"""
    lines = [
        f"- 用户: {t.strip().replace(chr(10), ' ')[:RULE_SUMMARY_LEN]}"
        f"{'...' if len(t.strip()) > RULE_SUMMARY_LEN else ''}"
        for t in texts
        if t.strip()
    ]
    return "\n".join(lines) if lines else ""


def _llm_summary(texts: list[str]) -> str:
    """用真实 LLM 压缩旧对话(失败自动降级为规则摘要,不引入本地假模型)。"""
    try:
        llm = create_chat_model()
        transcript = "\n".join(f"用户: {t}" for t in texts)
        prompt = (
            "请把下面的客服对话压缩成一段简洁的摘要(中文, 80 字以内),"
            "保留关键信息: 用户意图、提到的服务/时间/价格、已完成的预约动作等。\n\n"
            f"对话:\n{transcript}"
        )
        reply = llm.invoke(prompt)
        return (str(reply.content) or "").strip()[:200]
    except Exception:  # noqa: BLE001 - 摘要失败不影响主流程, 降级即可
        return _rule_summary(texts)


def build_summary(messages: list[Any]) -> str:
    """对超窗的旧对话生成摘要文本。"""
    texts = _human_texts(messages)
    if settings.has_llm_key:
        return _llm_summary(texts)
    return _rule_summary(texts)


def recent_history_texts(messages: list[Any], max_rounds: int = MAX_RECENT_ROUNDS) -> list[str]:
    """截断后的历史: 只保留最近的 max_rounds 条用户消息(进入 LLM 的部分)。"""
    return _human_texts(messages)[-max_rounds:]


def prepare_history(messages: list[Any]) -> tuple[str, list[str]]:
    """给咨询节点用的完整准备函数。

    返回 (summary_block, recent_history):
      - 未超窗: ("", 全部历史)        → 与现状一致, 零变化;
      - 超窗:   (摘要文本, 最近N轮)   → 旧历史被折叠成摘要。
    """
    if not should_compress(messages):
        return "", _human_texts(messages)
    summary = build_summary(messages)
    recent = recent_history_texts(messages)
    return summary, recent
