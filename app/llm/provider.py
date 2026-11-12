"""LLM 与 Embedding Provider 工厂。

设计要点:
- 任何 OpenAI 兼容端点(qwen / deepseek / zhipu / 火山方舟等)统一走 ChatOpenAI 接口,
  服务商地址由 LLM_BASE_URL / EMBEDDING_BASE_URL 指定,不维护 provider 枚举;
- 必须配置 LLM_API_KEY / EMBEDDING_API_KEY 才能创建模型,未配置时直接报错;
  不提供任何「本地模板 LLM」「哈希向量 Embedding」之类的假模型降级,
  保证所有回答与向量都来自真实 API,项目无 key 时宁可报错也不伪装。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config.settings import settings


def create_chat_model(temperature: float = 0) -> ChatOpenAI:
    """创建对话模型(ChatOpenAI 兼容接口,各家 provider 统一)。

    必须配置真实 LLM_API_KEY(占位符 your_xxx 也算未配置),否则抛错——不提供模板假模型。
    """
    if not settings.has_llm_key:
        raise RuntimeError("未配置 LLM_API_KEY,无法创建对话模型")
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL or None,
        temperature=temperature,
    )


@lru_cache
def create_embeddings() -> Embeddings:
    """创建 Embedding 模型(OpenAIEmbeddings 兼容接口)。

    必须配置 EMBEDDING_API_KEY 或 LLM_API_KEY,否则抛错——不提供本地哈希向量。
    """
    api_key = (settings.EMBEDDING_API_KEY or settings.LLM_API_KEY or "").strip()
    if not api_key or api_key.startswith("your_"):
        raise RuntimeError("未配置 EMBEDDING_API_KEY 或 LLM_API_KEY,无法创建 Embedding 模型")
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=api_key,
        base_url=settings.EMBEDDING_BASE_URL or None,
        # langchain-openai>=1.x 默认会把输入 tokenize 成整数数组再发请求,
        # 千问 DashScope 兼容模式只接受字符串,这里关闭预编码,直发原文
        check_embedding_ctx_length=False,
        # DashScope text-embedding-v3 单次请求最多 10 条 input,
        # 调小 chunk_size 让客户端自动分批
        chunk_size=8,
    )
