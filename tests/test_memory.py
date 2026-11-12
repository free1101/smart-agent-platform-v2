"""多轮回话记忆测试:验证同一会话跨轮上下文(补全/改期/取消/隔离)。"""

from __future__ import annotations

import pytest

from langchain_core.messages import HumanMessage

from app.agents.graph import build_agent_graph, supervisor_node


@pytest.fixture(autouse=True)
def _in_memory_db(monkeypatch):
    """业务节点落库走全局 SessionLocal(指向开发库 data/smart_agent.db)。

    为避免单测污染开发库,测试期把 app.db.database.SessionLocal 替换为内存
    SQLite(StaticPool 共享同一连接,兼容业务代码内多次 SessionLocal() 开合会话)。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import app.db.database as database_module
    import app.mcp.tools as mcp_tools
    from app.db import models  # noqa: F401  注册模型

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database_module.Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(database_module, "SessionLocal", TestSession)
    # 业务节点优先走 MCP 注册工具(app.mcp.tools 在模块顶部 import 了 SessionLocal),
    # 这里一并替换,否则工单会写进真实开发库、后续取消/改期查不到。
    monkeypatch.setattr(mcp_tools, "SessionLocal", TestSession)
    yield
    engine.dispose()


@pytest.fixture()
def graph():
    """每个测试独立编译一个图(显式 MemorySaver)。

    注意:不传时 build_agent_graph() 默认用 SqliteSaver(磁盘持久化),固定
    thread_id 会跨 pytest 运行残留历史,导致断言不稳定;这里强制内存
    checkpointer,每次运行都是干净会话。
    """
    from langgraph.checkpoint.memory import MemorySaver

    return build_agent_graph(checkpointer=MemorySaver())


def _chat(graph, thread_id: str, session_id: str, text: str, tenant_id: int = 1) -> dict:
    return graph.invoke(
        {
            "messages": [HumanMessage(content=text)],
            "session_id": session_id,
            "tenant_id": tenant_id,
        },
        config={"configurable": {"thread_id": f"t{tenant_id}-{thread_id}"}},
    )


def test_multi_turn_complete_booking(graph):
    """第1轮只给服务,第2轮补时间 → 应能记住服务并完成预约。"""
    tid = "mem-complete-1"

    r1 = _chat(graph, tid, "s1", "我想预约肩颈推拿")
    # 缺时间 → 追问并标记等待补全
    assert "补充" in r1["reply"]
    assert r1.get("awaiting_booking") is True
    assert r1.get("intent") == "business"

    r2 = _chat(graph, tid, "s1", "明天下午3点")
    # 无强意图但上一轮等待补全 → 延续预约流程
    assert r2.get("intent") == "business"
    assert "已为您预约" in r2["reply"]
    assert "肩颈推拿" in r2["reply"]
    assert "T-" in r2["reply"]
    assert r2.get("awaiting_booking") is False


def test_multi_turn_reschedule(graph):
    """完整预约后,第二轮改期 → 应更新最近工单时间而不是新建。"""
    tid = "mem-reschedule-1"

    r1 = _chat(graph, tid, "s2", "帮我预约明天下午3点的全身推拿,我叫李四")
    assert "已为您预约" in r1["reply"]

    r2 = _chat(graph, tid, "s2", "改到周五下午2点")
    assert "已为您改期" in r2["reply"]
    assert "周五" in r2["reply"]
    assert "全身推拿" in r2["reply"]


def test_multi_turn_cancel(graph):
    """完整预约后,第二轮取消 → 应取消最近工单。"""
    tid = "mem-cancel-1"

    r1 = _chat(graph, tid, "s3", "帮我预约明天下午2点的足底按摩")
    assert "已为您预约" in r1["reply"]

    r2 = _chat(graph, tid, "s3", "取消我的预约")
    assert "已为您取消" in r2["reply"]


def test_thread_isolation(graph):
    """不同 thread_id(会话)之间的记忆相互隔离。"""
    # 会话 A 完成预约
    r1 = _chat(graph, "iso-a", "sa", "帮我预约明天下午3点的头部刮痧,我叫王五")
    assert "已为您预约" in r1["reply"]

    # 会话 B 没有预约历史,改期应提示先预约
    r2 = _chat(graph, "iso-b", "sb", "改到周五")
    assert "没有找到可改期的预约" in r2["reply"]
    assert "T-" not in r2["reply"]


def test_awaiting_booking_survives_llm_handoff(graph, monkeypatch):
    """预约补全中用户只回姓名:即使 LLM 误判 handoff,也要继续补全而不是转人工。"""
    tid = "mem-handoff-1"

    r1 = _chat(graph, tid, "s1", "我想预约背部推拿")
    assert r1.get("awaiting_booking") is True

    # 模拟真实 LLM 把「王先生」这种孤立短句判成 handoff
    from app.agents.intent_router import IntentRouter

    monkeypatch.setattr(
        IntentRouter, "classify", lambda self, text, **kw: ("handoff", "LLM 意图分类")
    )

    r2 = _chat(graph, tid, "s1", "王先生")
    assert r2.get("intent") == "business"
    assert "人工客服" not in r2["reply"]
    assert r2.get("booking", {}).get("customer_name") == "王先生"


def test_awaiting_booking_explicit_handoff_still_wins(graph):
    """补全中用户明确说「转人工/投诉」时,仍应尊重用户意图转人工。"""
    tid = "mem-handoff-2"

    r1 = _chat(graph, tid, "s1", "我想预约背部推拿")
    assert r1.get("awaiting_booking") is True

    r2 = _chat(graph, tid, "s1", "我要转人工投诉")
    assert r2.get("intent") == "handoff"
    assert "人工客服" in r2["reply"]


def test_extract_honorific_name():
    """纯称谓短句应被识别为姓名(跨轮补全常见)。"""
    from app.agents.business import extract_booking_info

    assert extract_booking_info("王先生").get("customer_name") == "王先生"
    assert extract_booking_info("李女士").get("customer_name") == "李女士"
    # 时间短句不能被误判成姓名
    assert "customer_name" not in extract_booking_info("明天下午")


def test_react_path_short_answer_continues(monkeypatch):
    """ReAct 路径不落 awaiting 标志:上轮 business + 本轮短句(姓名)仍应继续业务,

    即使 LLM 把它误判成 handoff(这是用户实测「回王先生 → 转人工」的根因)。
    """
    from app.agents.intent_router import IntentRouter

    monkeypatch.setattr(
        IntentRouter, "classify", lambda self, text, **kw: ("handoff", "LLM 意图分类")
    )
    state = {
        "messages": [HumanMessage(content="王先生")],
        "session_id": "s1",
        "tenant_id": 1,
        "user_key": "test-no-memory-user",
        "intent": "business",        # 上一轮 checkpoint 里的值
        "awaiting_booking": False,   # ReAct 路径不会设置它
    }
    out = supervisor_node(state)
    assert out["intent"] == "business"
    assert "补充信息" in out["intent_reason"]


def test_classifier_receives_dialogue_context(monkeypatch):
    """意图分类必须带上近期对话 + 上一轮意图,而不是只看当前一句。"""
    from langchain_core.messages import AIMessage

    from app.agents.intent_router import IntentRouter

    captured = {}

    def fake_classify(self, text, context="", prev_intent=""):
        captured["text"] = text
        captured["context"] = context
        captured["prev"] = prev_intent
        return "business", "fake"

    monkeypatch.setattr(IntentRouter, "classify", fake_classify)

    state = {
        "messages": [
            HumanMessage(content="我想预约背部推拿"),
            AIMessage(content="好的,请问您想约哪天?怎么称呼?"),
            HumanMessage(content="王先生"),
        ],
        "session_id": "s1",
        "tenant_id": 1,
        "user_key": "test-no-memory-user",
        "intent": "business",
        "awaiting_booking": False,
    }
    supervisor_node(state)

    assert captured["text"] == "王先生"
    assert "背部推拿" in captured["context"]      # 上一轮用户消息
    assert "怎么称呼" in captured["context"]       # 上一轮助手回复
    assert "王先生" not in captured["context"]     # 当前消息不重复塞进上下文
    assert captured["prev"] == "business"


def test_react_path_question_not_treated_as_slot_answer(monkeypatch):
    """上轮 business 但本轮是完整提问(带问号/疑问词) → 不强行拉回业务。"""
    from app.agents.intent_router import IntentRouter

    monkeypatch.setattr(
        IntentRouter, "classify", lambda self, text, **kw: ("consult", "LLM 意图分类")
    )
    state = {
        "messages": [HumanMessage(content="你们几点关门?")],
        "session_id": "s1",
        "tenant_id": 1,
        "user_key": "test-no-memory-user",
        "intent": "business",
        "awaiting_booking": False,
    }
    out = supervisor_node(state)
    assert out["intent"] == "consult"
