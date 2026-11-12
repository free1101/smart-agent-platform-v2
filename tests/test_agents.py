"""Agent 图与业务节点测试。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.business import extract_booking_info
from app.agents.intent_router import INTENT_BUSINESS, INTENT_CONSULT, INTENT_HANDOFF, intent_router


def test_intent_consult():
    intent, _ = intent_router.classify("你们几点开门几点关门?")
    assert intent == INTENT_CONSULT


def test_intent_business():
    intent, _ = intent_router.classify("帮我预约明天下午3点的肩颈推拿")
    assert intent == INTENT_BUSINESS


def test_intent_handoff():
    intent, _ = intent_router.classify("我要找人工客服投诉")
    assert intent == INTENT_HANDOFF


def test_extract_booking():
    info = extract_booking_info("帮我预约明天下午3点的全身推拿,我叫张三")
    assert info.get("service_type") == "全身推拿"
    assert info.get("customer_name") == "张三"
    assert "明天" in info.get("appointment_time", "")
