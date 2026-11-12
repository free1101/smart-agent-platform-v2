"""用户画像服务与 profile 意图测试。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from app.agents.intent_router import INTENT_PROFILE, intent_router
from app.services import profile_service
from app.services.profile_service import (
    generate_return_message,
    get_profile,
    list_behaviors,
    record_behavior,
    should_return_reminder,
)


@pytest.fixture()
def db():
    """独立内存 SQLite,避免污染/清空真实开发数据库 data/smart_agent.db。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db import models  # noqa: F401  注册模型
    from app.db.database import Base

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestSession()
    yield session
    session.close()
    engine.dispose()


def test_intent_profile():
    intent, _ = intent_router.classify("我常做什么服务?")
    assert intent == INTENT_PROFILE


def test_intent_profile_return():
    intent, _ = intent_router.classify("好久没来了,给我推荐一下")
    assert intent == INTENT_PROFILE


def test_record_and_aggregate(db):
    record_behavior(
        db, 1, "s1", "appointment",
        {"service_type": "肩颈推拿", "appointment_time": "明天下午3点", "duration": "60"},
    )
    record_behavior(
        db, 1, "s1", "appointment",
        {"service_type": "肩颈推拿", "appointment_time": "周六晚上7点", "duration": "90"},
    )
    record_behavior(
        db, 1, "s1", "consultation", {"question": "价格"},
    )

    profile = get_profile(db, 1, "s1")
    assert profile["favorite_service"] == "肩颈推拿"
    assert profile["total_appointments"] == 2
    assert profile["has_profile"] is True
    # 两个时间分属下午/晚上,取频次最高者(此处各 1 次,取较晚写入)
    assert profile["favorite_time_period"] in {"下午", "晚上"}

    behaviors = list_behaviors(db, 1, "s1", limit=10)
    assert len(behaviors) == 3
    assert behaviors[0]["action_type"] == "consultation"


def test_empty_profile(db):
    profile = get_profile(db, 1, "no_such_session")
    assert profile["has_profile"] is False
    assert profile["total_appointments"] == 0


def test_return_message(db):
    record_behavior(
        db, 1, "s2", "appointment",
        {"service_type": "足底按摩", "appointment_time": "2020-01-01 10:00", "duration": "60"},
    )
    profile = get_profile(db, 1, "s2")
    assert should_return_reminder(db, 1, "s2", days_threshold=1) is True
    msg = generate_return_message(db, 1, "s2", profile=profile)
    assert "足底按摩" in msg


def test_infer_period():
    from app.services.profile_service import _infer_time_period

    assert _infer_time_period("明天上午9点") == "上午"
    assert _infer_time_period("下午3点") == "下午"
    assert _infer_time_period("晚上8点") == "晚上"
    assert _infer_time_period("周六") == ""
