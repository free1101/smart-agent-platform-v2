"""意图路由:判断用户消息属于 咨询 / 业务 / 转人工。

判断顺序:
- 强信号规则命中优先(转人工 / 高置信画像词),确定性高,不走 LLM;
- 其余表述调用 LLM 分类(可解释、可扩展);
- LLM 未配置或失败时,回落关键词规则兜底(关键词在下方配置)
"""

from __future__ import annotations

from app.config.settings import settings
from app.llm.provider import create_chat_model

# 意图标识
INTENT_CONSULT = "consult"      # 知识咨询:营业时间/价格/政策等
INTENT_BUSINESS = "business"    # 业务动作:预约/取消/下单等
INTENT_PROFILE = "profile"      # 用户画像/个性化:我常做的、我的偏好、回访
INTENT_HANDOFF = "handoff"      # 超出范围:转人工

# 关键词规则(强信号短路 + LLM 失败兜底,与知识分类/业务动词对齐)
_HANDOFF_KEYWORDS = [
    "人工", "客服", "投诉", "退款", "理赔", "紧急", "人不在",
    "联系真人", "转人工", "人工客服",
]
# 强信号画像词:自指/回访/推荐,高置信,规则命中优先于 LLM。
# (避免 LLM 把「好久没来了,给我推荐一下」这类回访话术误判成 consult)
_STRONG_PROFILE_KEYWORDS = [
    "我常做", "我常约", "我常来", "我常点", "我经常做", "我经常约",
    "我的偏好", "我的喜好", "我的习惯", "我的画像", "了解我",
    "我上次", "我约过", "我预约过", "我的历史", "我的记录",
    "常做项目", "常约时间",
    "给我推荐", "推荐我", "回访", "好久没来", "好久不见",
]
# 用户画像/个性化查询(优先级高于咨询,避免被当成普通知识问答;仅 LLM 兜底时使用)
_PROFILE_KEYWORDS = [
    "我常做", "我常约", "我常来", "我常点", "我经常做", "我经常约",
    "我的偏好", "我的喜好", "我的习惯", "我的画像", "了解我",
    "我上次", "我约过", "我预约过", "我的历史", "我的记录",
    "常做项目", "常约时间", "喜欢什么服务", "喜欢什么时候",
    "给我推荐", "推荐我", "回访", "好久没来", "好久不见",
]
_BUSINESS_KEYWORDS = [
    # 强动作词:明确表达「执行预约/订单」意图
    "帮我预约", "我要预约", "想预约", "预约一下", "预约一个",
    "预约明天", "预约今天", "帮我约", "我要约", "想约",
    "取消预约", "改期", "订一下", "帮我订", "我要订", "预订一个",
    "下单", "购买", "报名", "办卡", "充值", "安排",
    # 多轮记忆相关动作:改期/取消/修改
    "改到", "修改", "换个时间", "换到", "变更", "调整",
    "取消", "退掉", "退订", "不去了",
]


class IntentRouter:
    """意图路由器。"""

    def classify(
        self,
        text: str,
        context: str = "",
        prev_intent: str = "",
    ) -> tuple[str, str]:
        """返回 (intent, reason)。

        判断顺序:
        1. 强信号规则命中(转人工 / 画像词):确定性高,直接短路,不交给 LLM;
        2. 配置了真实 LLM key 时再走 LLM 泛化分类(覆盖规则覆盖不到的表述);
        3. 仍无结果:完整规则兜底(consult/business)。

        context / prev_intent:近期对话与上一轮意图。分类器原本只看当前一句,
        导致「王先生」「明天下午3点」这种补充句被误判成 handoff;带上上下文后
        LLM 才能判断"这是在回答上一轮的问题"。
        """
        intent, reason = self._classify_with_rules(text, strong_only=True)
        if intent:
            return intent, reason

        if settings.has_llm_key:
            intent, reason = self._classify_with_llm(
                text, context=context, prev_intent=prev_intent
            )
            if intent:
                return intent, reason


        # 兜底:完整规则分类
        intent, reason = self._classify_with_rules(text)
        if intent:
            return intent, reason
        return INTENT_CONSULT, "未命中业务/转人工关键词,按知识咨询处理"

    @staticmethod
    def _classify_with_rules(text: str, strong_only: bool = False) -> tuple[str, str]:
        """关键词规则分类。strong_only=True 时只匹配强信号词(转人工/画像)。"""
        text_lower = text.lower()
        if strong_only:
            for kw in _HANDOFF_KEYWORDS:
                if kw in text_lower:
                    return INTENT_HANDOFF, f"命中转人工关键词「{kw}」"
            for kw in _STRONG_PROFILE_KEYWORDS:
                if kw in text_lower:
                    return INTENT_PROFILE, f"命中用户画像关键词「{kw}」"
            return "", ""
        for kw in _HANDOFF_KEYWORDS:
            if kw in text_lower:
                return INTENT_HANDOFF, f"命中转人工关键词「{kw}」"
        for kw in _PROFILE_KEYWORDS:
            if kw in text_lower:
                return INTENT_PROFILE, f"命中用户画像关键词「{kw}」"
        for kw in _BUSINESS_KEYWORDS:
            if kw in text_lower:
                return INTENT_BUSINESS, f"命中业务关键词「{kw}」"
        return "", ""

    def _classify_with_llm(
        self,
        text: str,
        context: str = "",
        prev_intent: str = "",
    ) -> tuple[str, str]:
        """LLM 分类(仅 API 模式)。失败时返回空串,走规则兜底。

        context / prev_intent 会拼进 prompt,让模型知道"上一轮在聊什么"。
        """
        try:
            llm = create_chat_model(temperature=0)
            ctx_block = ""
            if prev_intent:
                ctx_block += f"【上一轮意图】{prev_intent}\n"
            if context:
                ctx_block += f"【近期对话】\n{context}\n"
            prompt = (
                "你是一个客服意图分类器。请判断用户消息的意图,只输出以下四种之一:\n"
                "- consult:用户想了解信息(价格/营业时间/政策/服务内容等)\n"
                "- business:用户想执行动作(预约、取消、下单、充值等)\n"
                "- profile:用户询问自己的历史/偏好/画像(如「我常做什么项目」「我的偏好」"
                "「我上次约了什么」),需要基于用户画像回答\n"
                "- handoff:用户要求转人工、投诉、或超出客服能力范围\n"
                "注意:如果用户只是补充上一轮预约所需的信息(姓名、时间、服务项目、时长等),"
                "例如只回「王先生」「明天下午3点」「60分钟」,应判为 business,不要判 handoff。\n\n"
                f"{ctx_block}"
                f"【本轮用户消息】{text}\n"
                "只输出意图单词,不要解释。"
            )
            resp = llm.invoke(prompt)
            intent = str(resp.content).strip().lower()
            if intent in {INTENT_CONSULT, INTENT_BUSINESS, INTENT_PROFILE, INTENT_HANDOFF}:
                return intent, "LLM 意图分类"
            return "", ""
        except Exception:
            return "", ""


def is_explicit_handoff(text: str) -> bool:
    """用户是否「明确」要求转人工/投诉(强信号关键词)。

    用途:多轮预约补全中,用户只回姓名/时间这类短句时,LLM 容易误判成 handoff;
    supervisor 只有在"明确转人工"时才允许打断补全流程。
    """
    text_lower = (text or "").lower()
    return any(kw in text_lower for kw in _HANDOFF_KEYWORDS)


# 出现这些词就不是"补充字段",而是新问题或寒暄
_QUESTION_OR_CHITCHAT = [
    "吗", "呢", "多少", "怎么", "几", "什么", "哪", "如何", "能不能", "可以",
    "谢谢", "好的", "不用", "算了", "嗯", "行", "ok",
]


def looks_like_slot_answer(text: str, max_len: int = 10) -> bool:
    """短句补充启发式:判断本轮是否只是补一个字段(姓名/时间/时长)。

    用于 ReAct 路径:LLM 走完一轮没有落 `awaiting_booking` 状态,
    下一轮用户只回「王先生」「明天下午3点」时,靠这条规则把意图拉回 business。

    排除条件(避免误伤):带问号、含疑问词(多少钱/几点)、寒暄词(好的/谢谢)。
    """
    t = (text or "").strip()
    if not t or len(t) > max_len:
        return False
    if "?" in t or "？" in t:
        return False
    return not any(kw in t.lower() for kw in _QUESTION_OR_CHITCHAT)


intent_router = IntentRouter()
