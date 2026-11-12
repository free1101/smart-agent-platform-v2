"""启动 MCP 工具服务(streamable-http @ 127.0.0.1:9000)。

用法:
    python scripts/start_mcp_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.mcp.tools import mcp


def main() -> None:
    print("启动 MCP 工具服务: http://127.0.0.1:9000/mcp (Ctrl+C 退出)")
    mcp.run(transport="streamable-http", host="127.0.0.1", port=9000)


if __name__ == "__main__":
    main()
