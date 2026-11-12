"""进程内滑动窗口限流。

为什么不用 Redis:单机部署、单进程 uvicorn 足够;多副本时应换成 Redis/网关限流,
这里保留清晰的接口边界(`check_rate_limit` 作为一个 FastAPI 依赖)方便替换。
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import HTTPException, Request, status

from app.config.settings import settings

_WINDOW_SECONDS = 60.0
_buckets: dict[str, deque[float]] = {}
_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    """每分钟最多 `RATE_LIMIT_PER_MINUTE` 次;0 表示关闭限流。"""
    limit = int(settings.RATE_LIMIT_PER_MINUTE or 0)
    if limit <= 0:
        return

    api_key = request.headers.get("x-api-key") or request.headers.get("authorization") or ""
    bucket_key = f"{(api_key[:16] or 'anon')}@{_client_ip(request)}"
    now = time.monotonic()

    with _lock:
        bucket = _buckets.setdefault(bucket_key, deque())
        while bucket and now - bucket[0] > _WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(_WINDOW_SECONDS - (now - bucket[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁,请 {retry_after} 秒后重试",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)
        # 顺手清理空桶,避免长期运行后字典无限增长
        if len(_buckets) > 4096:
            for key in [k for k, v in _buckets.items() if not v]:
                _buckets.pop(key, None)


def reset_rate_limit() -> None:
    """清空计数(测试用)。"""
    with _lock:
        _buckets.clear()
