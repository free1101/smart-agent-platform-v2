"""图片描述器:把文档中的图片转成可检索的文字,并原位回填。

容错原则:
- 单张图失败只跳过该图(正文保留 `[[IMAGE:n]]` 占位符),不阻断整篇文档;
- 单张图过大(> `CAPTION_MAX_BYTES`)直接跳过,避免请求体爆掉;
- 每篇文档最多描述 `CAPTION_MAX_IMAGES` 张,控制成本。
"""

from __future__ import annotations

import logging

from app.config.settings import settings
from app.ingestion.loaders import fill_image_captions

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """对一批图片生成描述(顺序与输入一致,失败位置为空串)。"""

    def __init__(self, max_images: int | None = None, max_bytes: int | None = None) -> None:
        self.max_images = max_images or settings.CAPTION_MAX_IMAGES
        self.max_bytes = max_bytes or settings.CAPTION_MAX_BYTES

    def caption(self, images: list[bytes]) -> list[str]:
        if not images:
            return []

        from app.llm.vision import describe_image, vision_available

        if not vision_available():
            logger.info("未配置视觉模型,跳过 %s 张图片描述", len(images))
            return ["" for _ in images]

        captions: list[str] = []
        for idx, data in enumerate(images):
            if idx >= self.max_images:
                logger.info("图片数量超过上限 %s,其余跳过", self.max_images)
                captions.append("")
                continue
            if not data or len(data) > self.max_bytes:
                logger.warning("图片 %s 为空或超过 %s 字节,跳过", idx, self.max_bytes)
                captions.append("")
                continue
            try:
                captions.append(describe_image(data))
            except Exception as exc:  # noqa: BLE001 - 单图失败不影响整篇
                logger.warning("图片 %s 描述失败,已跳过: %s", idx, exc)
                captions.append("")
        return captions

    def caption_document(self, text: str, images: list[bytes]) -> tuple[str, int]:
        """给整篇文档的图片做描述并回填。

        Returns:
            (回填后的文本, 成功描述的图片数)
        """
        captions = self.caption(images)
        filled = fill_image_captions(text, captions)
        return filled, sum(1 for c in captions if c)
