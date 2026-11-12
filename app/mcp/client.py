"""MCP 客户端:连接 MCP 工具服务,将标准工具转成 LangChain 工具。

设计:
- 懒加载连接,避免启动时依赖 MCP server 存活;
- MCP server 不可用时优雅降级:返回空列表,上层 Agent 走本地逻辑;
- 使用 langchain-mcp-adapters 标准适配,符合 MCP 协议规范。
"""

from __future__ import annotations

import asyncio
import logging

from app.config.settings import settings

logger = logging.getLogger(__name__)

_mcp_tools: list | None = None


def get_mcp_tools() -> list:
    """获取 MCP 工具列表(懒加载 + 缓存 + 失败降级)。"""
    global _mcp_tools
    if _mcp_tools is not None:
        return _mcp_tools
    if not settings.MCP_ENABLED:
        _mcp_tools = []
        return _mcp_tools

    try:
        _mcp_tools = _load_tools()
        if _mcp_tools:
            logger.info("MCP 工具已加载: %s 个", len(_mcp_tools))
        else:
            logger.info("MCP 未返回任何工具,Agent 使用本地工具")
    except Exception as exc:  # pragma: no cover - 网络/连接类异常
        logger.warning("MCP 工具加载失败,降级为本地工具: %s", exc, exc_info=True)
        _mcp_tools = []
    return _mcp_tools


def reset_mcp_tools() -> None:
    """清空缓存(测试或 MCP server 重启后使用)。"""
    global _mcp_tools
    _mcp_tools = None


def _load_tools() -> list:
    """实际加载:连接 MCP server 并转换工具。"""
    # 使用 sync 桥接 async 调用
    return asyncio.run(_async_load_tools())


async def _async_load_tools() -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "smart-agent-tools": {
                "url": settings.MCP_SERVER_URL,
                "transport": "streamable-http",
            }
        }
    )
    tools = await client.get_tools()
    # 注意:get_tools() 返回的是 list[BaseTool],不是 dict。
    # 早期实现写成 list(tools.values())[0],会抛 AttributeError 并被外层吞掉,
    # 导致「MCP 永远降级为本地工具」——这是本次修复的 bug。
    return list(tools or [])


def mcp_enabled() -> bool:
    """MCP 是否启用且已成功加载工具。"""
    return bool(settings.MCP_ENABLED and _mcp_tools)
