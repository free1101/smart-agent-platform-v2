"""启动 Streamlit 控制台。

用法:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8501 --no-browser

为什么要包一层:
- `streamlit run` 需要以项目根为工作目录,否则 `.streamlit/config.toml` 找不到;
- 首次运行需要 init_db(),把新增的 4 张表建好。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_APP = PROJECT_ROOT / "dashboard" / "app.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="启动知识库控制台")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not DASHBOARD_APP.exists():
        print(f"[错误] 找不到面板入口: {DASHBOARD_APP}")
        sys.exit(1)

    try:
        import app.db.database as db_module

        db_module.init_db()
    except Exception as exc:  # noqa: BLE001 - 建表失败也要让用户看到面板的报错
        print(f"[警告] 数据库初始化失败: {exc}")

    url = f"http://127.0.0.1:{args.port}"
    print(f"控制台启动: {url}(Ctrl+C 退出)")
    if not args.no_browser:
        webbrowser.open(url)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_APP),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    except FileNotFoundError:
        print("未安装 streamlit,请执行: pip install streamlit")


if __name__ == "__main__":
    main()
