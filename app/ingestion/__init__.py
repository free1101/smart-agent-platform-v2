"""文档摄取:解析 → 图片描述 → 分块 → 去重 → 入库。"""

from app.ingestion.pipeline import IngestionPipeline, ingestion_pipeline

__all__ = ["IngestionPipeline", "ingestion_pipeline"]
