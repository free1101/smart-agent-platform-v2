"""应用入口:FastAPI 主应用。

启动:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from app.api.routes.eval import router as eval_router
from app.config.settings import settings
from app.db.database import init_db

# 初始化数据库(建表)
init_db()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESC,
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(chat_router)
app.include_router(eval_router)
app.include_router(admin_router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "docs": "/docs",
        "chat": "/api/chat",
        "web": "/web/index.html",
        "llm": "configured" if settings.has_llm_key else "missing-key",
        "chat_stream": "/api/chat/stream",
    }


@app.get("/api/health", tags=["系统"])
async def health():
    """健康检查(免鉴权):供容器 HEALTHCHECK 与负载均衡探测。"""
    if settings.API_KEY:
        auth = "required"
    elif settings.ALLOW_UNAUTHENTICATED:
        auth = "disabled"
    else:
        auth = "missing-key"
    return {
        "status": "ok",
        "llm": "configured" if settings.has_llm_key else "missing-key",
        "auth": auth,
        "vector_store": settings.VECTOR_STORE,
        "fusion": settings.FUSION_STRATEGY,
        "reranker": settings.RERANKER_PROVIDER,
    }


# 静态前端(web/ 目录)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
