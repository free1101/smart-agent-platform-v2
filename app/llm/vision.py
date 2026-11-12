"""视觉模型:给图片生成文字描述(多模态 RAG 的 Image-to-Text 策略)。

为什么不直接存图片向量:
- 文本检索链路(BM25 + 文本 embedding)已经成熟,把图片「翻译」成文字后
  可以直接复用,不需要额外的跨模态向量库;
- 代价是描述质量取决于视觉模型,所以 prompt 要强调「与业务相关的可见信息」,
  不要输出「这是一张图片」这类无用描述。

未配置 key 时抛错,由调用方决定跳过(不伪造描述)。
"""

from __future__ import annotations

import base64
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

_PROMPT = (
    "请用中文客观描述这张图片的内容,重点提取其中与业务相关的可见信息:"
    "如果是图表,说明标题、坐标轴、趋势与关键数值;"
    "如果是截图或文档,尽量转写其中可辨认的文字;"
    "如果是照片,说明场景、物体与可见标识。"
    "不要猜测图片里没有的信息,不要输出「这是一张图片」这类空话,控制在 120 字以内。"
)


def vision_available() -> bool:
    """是否具备调用视觉模型的条件。"""
    return bool(settings.VISION_ENABLED and settings.has_llm_key)


def describe_image(data: bytes, prompt: str | None = None) -> str:
    """给一张图片生成描述。

    Raises:
        RuntimeError: 未配置 key 或模型调用失败(调用方应捕获并跳过该图)。
    """
    if not vision_available():
        raise RuntimeError("未配置 LLM_API_KEY 或已关闭 VISION_ENABLED,无法生成图片描述")

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    encoded = base64.b64encode(data).decode("ascii")
    llm = ChatOpenAI(
        model=settings.VISION_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=(settings.VISION_BASE_URL or settings.LLM_BASE_URL) or None,
        temperature=0,
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt or _PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]
    )
    resp = llm.invoke([message])
    text = str(getattr(resp, "content", "") or "").strip()
    if not text:
        raise RuntimeError("视觉模型返回空描述")
    return text
