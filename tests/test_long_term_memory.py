"""扩展能力测试:上下文工程 + 长期记忆(跨会话召回)。

约定:
- 不依赖真实 LLM / 网络:把各模块引用的 settings 换成 has_llm_key=False 的桩,
  业务意图自动降级到规则流程(与"无 key 也能跑"的降级语义一致);
- 不污染开发库:把 app.db.database 与 app.mcp.tools 的 SessionLocal 都换成内存
  SQLite(StaticPool 共享同一连接,兼容业务代码内多次 SessionLocal() 开合)。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.agents.context import (
    MAX_RECENT_ROUNDS,
    prepare_history,
    should_compress,
)
from app.agents.graph import build_agent_graph, supervisor_node
from app.services.memory_service import recall, remember


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """强制进入"无真实 key"的降级路径,避免单测打真实 API。"""
    from types import SimpleNamespace

    import app.agents.context as context_module
    import app.agents.intent_router as intent_module
    import app.agents.react as react_module

    stub = SimpleNamespace(has_llm_key=False, LLM_API_KEY="")
    monkeypatch.setattr(context_module, "settings", stub)
    monkeypatch.setattr(intent_module, "settings", stub)
    monkeypatch.setattr(react_module, "settings", stub)


@pytest.fixture()
def test_db(monkeypatch):
    """内存 SQLite:同时替换 database 与 mcp.tools 两处 SessionLocal。"""
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
    monkeypatch.setattr(mcp_tools, "SessionLocal", TestSession)

    db = TestSession()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def graph(test_db):
    """每个测试独立编译图(显式 MemorySaver,不落盘、不跨运行残留)。"""
    from langgraph.checkpoint.memory import MemorySaver

    return build_agent_graph(checkpointer=MemorySaver())


def _chat(graph, thread_id: str, session_id: str, text: str, user_key: str = "") -> dict:
    return graph.invoke(
        {
            "messages": [HumanMessage(content=text)],
            "session_id": session_id,
            "tenant_id": 1,
            "user_key": user_key or session_id,
        },
        config={"configurable": {"thread_id": f"t1-{thread_id}"}},
    )


# ---------------------------------------------------------------------------
# 上下文工程
# ---------------------------------------------------------------------------
def test_context_compress_threshold():
    """未超窗不压缩;超过 MAX_ROUNDS 才折叠并只保留最近 N 轮。"""
    few = [HumanMessage(content=f"第{i}问") for i in range(5)]
    assert should_compress(few) is False
    summary, recent = prepare_history(few)
    assert summary == ""
    assert len(recent) == 5

    many = [HumanMessage(content=f"第{i}问") for i in range(9)]
    assert should_compress(many) is True
    summary, recent = prepare_history(many)
    assert summary != ""
    assert len(recent) == MAX_RECENT_ROUNDS
    assert recent[-1] == "第8问"


# ---------------------------------------------------------------------------
# 长期记忆
# ---------------------------------------------------------------------------
def test_recall_isolation_by_user_and_tenant(test_db):
    """记忆按 user_key 与 tenant_id 双维度隔离。"""
    remember(test_db, 1, "u-alice", "s1", "用户预约了「肩颈推拿」(工单 T-0001)", "appointment", 8)

    hits = recall(test_db, 1, "u-alice", query="我上次约了什么")
    assert hits and "肩颈推拿" in hits[0]["content"]

    assert recall(test_db, 1, "u-bob") == []
    assert recall(test_db, 2, "u-alice") == []


def test_cross_session_recall_in_supervisor(test_db):
    """新会话首轮:主管按 user_key 召回老记忆并注入 state['context']。"""
    remember(test_db, 1, "u-42", "old-session", "用户预约了「足底按摩」", "appointment", 8)

    state = supervisor_node(
        {
            "messages": [HumanMessage(content="你还记得我吗")],
            "tenant_id": 1,
            "session_id": "brand-new-session",
            "user_key": "u-42",
        }
    )
    assert "足底按摩" in state.get("context", "")


def test_booking_writes_long_term_memory(graph, test_db):
    """无 key 走规则预约,成功后应在 memory_entries 落一条跨会话事实。"""
    r1 = _chat(graph, "lt-a", "sess-a", "帮我预约明天下午3点的肩颈推拿", user_key="u-42")
    assert "已为您预约" in r1["reply"]

    hits = recall(test_db, 1, "u-42", query="我上次约了什么")
    assert hits, "预约成功后应写入长期记忆"
    assert "肩颈推拿" in hits[0]["content"]

    # 换新会话(同 user_key)→ 首轮召回,业务节点能看到老预约
    r2 = _chat(graph, "lt-b", "sess-b", "你还记得我吗", user_key="u-42")
    assert "肩颈推拿" in r2.get("context", "")
