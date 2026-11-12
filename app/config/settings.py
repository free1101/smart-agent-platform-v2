"""应用配置中心:所有可调参数集中在这里,支持 .env 覆盖。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 项目根目录


class Settings(BaseSettings):
    """全局配置,字段默认值与 .env.example 对齐。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- 项目 ----
    PROJECT_NAME: str = "Smart Agent Platform"
    PROJECT_DESC: str = "通用智能客服 Agent 平台:多 Agent 协作 + 混合检索 RAG + MCP 工具"

    # ---- LLM ----
    # 不设 provider 枚举:任何 OpenAI 兼容端点都靠 LLM_BASE_URL 指定
    # (qwen / deepseek / zhipu / 火山方舟等),换厂商只改 URL + 模型名 + key。
    LLM_MODEL: str = "qwen-plus"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""

    # ---- Embedding ----
    EMBEDDING_MODEL: str = "text-embedding-v3"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""

    # ---- 视觉模型(图片描述,可选;未配置时跳过图片描述) ----
    VISION_MODEL: str = "qwen-vl-max"
    VISION_BASE_URL: str = ""
    VISION_ENABLED: bool = True

    # ---- 数据库 ----
    DATABASE_URL: str = "sqlite:///./data/smart_agent.db"

    # ---- 服务 ----
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # ---- 检索 ----
    # 不显式传 top_k 时:先召回 RETRIEVER_TOP_K 条,精排后返回 RERANK_TOP_K 条
    # (实现里会取两者较小值,避免返回条数超过召回条数)。
    RETRIEVER_TOP_K: int = 5
    RETRIEVER_WEIGHT_BM25: float = 0.4
    RETRIEVER_WEIGHT_VECTOR: float = 0.6
    RERANK_TOP_K: int = 3

    # 融合策略:rrf(倒数排名融合)| weighted(分数归一化加权,旧行为)
    FUSION_STRATEGY: str = "rrf"
    RRF_K: int = 60
    # 每一路召回的候选数,融合前先粗排到该规模
    RETRIEVER_CANDIDATES: int = 20

    # 向量库:chroma(持久化)| memory(进程内 numpy,旧行为)
    VECTOR_STORE: str = "chroma"
    CHROMA_DIR: Path = BASE_DIR / "data" / "chroma"
    CHROMA_BATCH_SIZE: int = 64

    # 重排器:rule(规则,旧行为)| cross_encoder | llm | none
    RERANKER_PROVIDER: str = "rule"
    CROSS_ENCODER_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_CANDIDATES: int = 20

    # ---- MCP ----
    MCP_SERVER_URL: str = "http://127.0.0.1:9000/mcp"
    MCP_ENABLED: bool = True

    # ---- 摄取 ----
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    INGEST_MAX_MB: int = 20
    INGEST_CHUNK_SIZE: int = 200
    INGEST_CHUNK_OVERLAP: int = 40
    CAPTION_MAX_IMAGES: int = 3
    CAPTION_MAX_BYTES: int = 2 * 1024 * 1024

    # ---- 访问控制 ----
    API_KEY: str = ""
    ALLOW_UNAUTHENTICATED: bool = False
    RATE_LIMIT_PER_MINUTE: int = 120

    # ---- 数据目录 ----
    # 种子 JSON、评测集、SQLite 库、Chroma 索引统一放在 DATA_DIR 下,
    # 脚本一律从 settings.DATA_DIR 拼路径,避免各处重复写死 "data/"。
    DATA_DIR: Path = BASE_DIR / "data"

    # ---- 内置行业示例 ----
    DEFAULT_TENANT_NAME: str = "示例行业-健康养生"

    @property
    def has_llm_key(self) -> bool:
        """是否配置了可用的 LLM key(占位符视为未配置)。"""
        key = (self.LLM_API_KEY or "").strip()
        if not key:
            return False
        lowered = key.lower()
        return not (lowered.startswith("your") or lowered in {"xx", "xxx", "sk-xxx"})


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
