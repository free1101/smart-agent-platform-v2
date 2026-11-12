"""一键启动 Web 演示(自动初始化数据)。

用法:
    python scripts/run_web.py [--port 8000] [--no-seed]

默认行为:
    1. 初始化数据库(建表)
    2. 若知识库为空,自动导入 data/knowledge_seed.json
    3. 启动 FastAPI 服务,浏览器打开 http://127.0.0.1:8000/web/
"""

from __future__ import annotations

import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

from app.config.settings import settings
from app.db.database import SessionLocal, init_db
from app.db.models import KnowledgeDoc
from app.services.knowledge_service import knowledge_service


def ensure_seed() -> None:
    """知识库为空时自动导入种子数据。"""
    db = SessionLocal()
    try:
        count = db.query(KnowledgeDoc).count()
        if count == 0:
            seed = PROJECT_ROOT / "data" / "knowledge_seed.json"
            if seed.exists():
                print(f"[init] 导入种子知识: {seed.name} ...")
                knowledge_service.import_seed(
                    db,
                    seed_path=seed,
                    tenant_name=settings.DEFAULT_TENANT_NAME,
                    industry="健康养生",
                    source_tag="seed",
                    replace=False,
                )
                print("[init] 种子知识导入完成")
            else:
                print(f"[warn] 未找到种子文件: {seed}")
        else:
            print(f"[init] 知识库已有 {count} 条,跳过导入")
    finally:
        db.close()


def check_auth_config() -> None:
    """启动时说明鉴权状态,避免"页面能打开但接口全 503"的困惑。"""
    if settings.API_KEY:
        print("[auth] 已启用 API Key 鉴权(请求需带 X-API-Key 或 Bearer)")
        return
    if settings.ALLOW_UNAUTHENTICATED:
        print("[auth] 未配置 API_KEY,已按 ALLOW_UNAUTHENTICATED=true 放行(仅限本地开发)")
        return
    print("[warn] 未配置 API_KEY 且未开启 ALLOW_UNAUTHENTICATED:")
    print("       受保护接口(/api/chat、/api/kb/*、/api/admin/*)将返回 503。")
    print("       本地开发请在 .env 增加: ALLOW_UNAUTHENTICATED=true")
    print("       生产环境请配置: API_KEY=<你的密钥>")


def main() -> None:
    # 默认读 .env 的 PORT;命令行 --port 优先级更高(便于同时跑多份副本)
    port = settings.PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    print("=" * 60)
    print("  Smart Agent Platform · 智能客服演示台")
    print("=" * 60)

    init_db()
    print("[init] 数据库就绪")
    check_auth_config()
    ensure_seed()

    url = f"http://127.0.0.1:{port}/web/"
    print(f"\nWeb 演示: {url}")
    print(f"API 文档: http://127.0.0.1:{port}/docs")
    print("按 Ctrl+C 退出\n")

    # 延迟打开浏览器
    def _open_browser():
        time.sleep(2.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run("app.main:app", host=settings.HOST, port=port, reload=False)


if __name__ == "__main__":
    main()
