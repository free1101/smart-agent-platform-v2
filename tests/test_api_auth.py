"""鉴权与限流测试:503(未配置)/ 401(错误)/ 放行 / 429(限流)。"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.deps import require_api_key
from app.api.ratelimit import check_rate_limit, reset_rate_limit
from app.config.settings import settings


def _request(headers: dict[str, str] | None = None, ip: str = "127.0.0.1") -> Request:
    raw = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/chat",
            "headers": raw,
            "client": (ip, 12345),
        }
    )


def _auth(headers: dict[str, str] | None = None, ip: str = "127.0.0.1") -> str:
    """模拟 FastAPI 的依赖注入:从 header 里解析出参数再调用。

    直接调用函数时 `x_api_key` 的默认值是 `Header(...)` 对象而不是 None,
    所以这里显式按 HTTP 语义解析一遍。
    """
    lower = {key.lower(): value for key, value in (headers or {}).items()}
    return require_api_key(
        _request(headers, ip=ip),
        x_api_key=lower.get("x-api-key"),
        authorization=lower.get("authorization"),
    )


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """每个用例从「已配置密钥」开始,避免受本地 .env 影响。"""
    monkeypatch.setattr(settings, "API_KEY", "secret-key")
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 0)
    reset_rate_limit()
    yield
    reset_rate_limit()


def test_missing_config_returns_503(monkeypatch):
    """没配密钥又没显式放行:必须 503,不能静默裸奔。"""
    monkeypatch.setattr(settings, "API_KEY", "")
    with pytest.raises(HTTPException) as exc:
        _auth()
    assert exc.value.status_code == 503


def test_explicit_allow_unauthenticated(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(settings, "ALLOW_UNAUTHENTICATED", True)
    assert _auth() == "anonymous"


def test_missing_key_returns_401():
    with pytest.raises(HTTPException) as exc:
        _auth()
    assert exc.value.status_code == 401


def test_wrong_key_returns_401():
    with pytest.raises(HTTPException) as exc:
        _auth({"X-API-Key": "wrong"})
    assert exc.value.status_code == 401


def test_header_key_ok():
    assert _auth({"X-API-Key": "secret-key"}) == "secret-key"


def test_bearer_token_ok():
    assert _auth({"Authorization": "Bearer secret-key"}) == "secret-key"


def test_token_prefix_ok():
    assert _auth({"Authorization": "Token secret-key"}) == "secret-key"


def test_malformed_authorization_returns_401():
    with pytest.raises(HTTPException) as exc:
        _auth({"Authorization": "secret-key"})
    assert exc.value.status_code == 401


def test_rate_limit_blocks_after_quota(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 2)
    request = _request({"X-API-Key": "secret-key"}, ip="10.0.0.1")

    check_rate_limit(request)
    check_rate_limit(request)
    with pytest.raises(HTTPException) as exc:
        check_rate_limit(request)
    assert exc.value.status_code == 429
    assert "Retry-After" in (exc.value.headers or {})


def test_rate_limit_isolated_by_ip(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 1)
    key = {"X-API-Key": "secret-key"}

    check_rate_limit(_request(key, ip="10.0.0.2"))
    # 另一个 IP 有自己的桶,不应被前一个 IP 拖累
    check_rate_limit(_request(key, ip="10.0.0.3"))


def test_rate_limit_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 0)
    request = _request({"X-API-Key": "secret-key"}, ip="10.0.0.4")
    for _ in range(10):
        check_rate_limit(request)
