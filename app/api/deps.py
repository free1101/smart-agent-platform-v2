"""API 依赖:密钥鉴权与限流。

鉴权语义(刻意做成三级,避免「静默裸奔」):
- 配置了 `API_KEY`:请求必须带 `X-API-Key` 或 `Authorization: Bearer <key>`,否则 401;
- 没配置 `API_KEY` 且没设 `ALLOW_UNAUTHENTICATED=true`:返回 **503**,
  明确告诉运维「你还没配密钥」,而不是当作匿名放行;
- 显式设置 `ALLOW_UNAUTHENTICATED=true`:放行(仅用于本地开发与 CI)。
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Header, HTTPException, Request, status

from app.config.settings import settings

logger = logging.getLogger(__name__)


def _extract_key(x_api_key: str | None, authorization: str | None) -> str:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in {"bearer", "token"}:
            return parts[1].strip()
    return ""


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """校验 API Key,返回调用方标识(用于限流分桶)。"""
    expected = (settings.API_KEY or "").strip()

    if not expected:
        if settings.ALLOW_UNAUTHENTICATED:
            return "anonymous"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务未配置 API_KEY,已拒绝访问。请设置 API_KEY,或显式设置 ALLOW_UNAUTHENTICATED=true 仅用于本地开发。",
        )

    provided = _extract_key(x_api_key, authorization)
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key(请使用 X-API-Key 或 Authorization: Bearer <key>)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 恒定时间比较,避免通过响应时间逐字节猜密钥
    if not hmac.compare_digest(provided, expected):
        logger.warning("API Key 校验失败: path=%s client=%s", request.url.path, _client_ip(request))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key 无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
