"""MCP 协议客户端演示:通过 langchain-mcp-adapters 连接 MCP server。

验证「标准 MCP 协议 → LangChain 工具」完整链路。

前置:先启动 MCP server
    python app/mcp/tools.py          (streamable-http @ 127.0.0.1:9000)

用法:
    python scripts/test_mcp_client.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def main() -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    print("连接 MCP server...")
    client = MultiServerMCPClient(
        {
            "smart-agent-tools": {
                "url": "http://127.0.0.1:9000/mcp",
                "transport": "streamable-http",
            }
        }
    )
    try:
        tools = await client.get_tools()
        print(f"已加载 {len(tools)} 个 MCP 工具:")
        for t in tools:
            print(f"  - {t.name}: {t.description[:60]}")

        if not tools:
            print("未加载到工具,请确认 MCP server 已启动")
            return

        # 调用 create_booking 工具(通过 MCP 协议)
        booking_tool = next((t for t in tools if t.name == "create_booking"), None)
        if booking_tool:
            result = await booking_tool.ainvoke(
                {
                    "tenant_id": 1,
                    "session_id": "mcp-protocol-test",
                    "service_type": "肩颈推拿",
                    "appointment_time": "明天上午10点",
                    "customer_name": "协议测试",
                }
            )
            print(f"\ncreate_booking(经 MCP 协议) -> {result}")
        else:
            print("\n未找到 create_booking 工具")

        # 调用 get_weather
        weather_tool = next((t for t in tools if t.name == "get_weather"), None)
        if weather_tool:
            result = await weather_tool.ainvoke({"city": "上海"})
            print(f"get_weather(经 MCP 协议) -> {result}")
    finally:
        # MultiServerMCPClient 0.1.x 不能作为 context manager,不主动关闭
        pass


if __name__ == "__main__":
    asyncio.run(main())
