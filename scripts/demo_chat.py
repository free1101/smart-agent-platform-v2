"""命令行对话演示:多 Agent 图端到端测试(含多轮回话记忆)。

用法:
    python scripts/demo_chat.py [--tenant 租户名]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage

from app.agents.graph import agent_graph
from app.config.settings import settings
from app.db.database import SessionLocal, init_db
from app.db.models import Tenant


def resolve_tenant_id(db, tenant_name: str) -> int:
    tenant = db.query(Tenant).filter(Tenant.name == tenant_name).first()
    if tenant is None:
        print(f"[错误] 租户不存在: {tenant_name},请先运行 import_knowledge_seed.py")
        sys.exit(1)
    return tenant.id


def invoke(query: str, session_id: str, tenant_id: int) -> dict:
    """带 thread_id 调用图(同一 session_id 共享多轮记忆)。"""
    return agent_graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "tenant_id": tenant_id,
        },
        config={"configurable": {"thread_id": f"t{tenant_id}-{session_id}"}},
    )


def print_result(seq: int, query: str, result: dict) -> None:
    print("=" * 80)
    print(f"[{seq}] 用户: {query}")
    print(f"    意图: {result.get('intent')} ({result.get('intent_reason')})")
    reply = result.get("reply", "")
    print(f"    回答: {reply[:200]}")
    if result.get("sources"):
        print(f"    来源: {result['sources']}")
    if result.get("booking"):
        print(f"    工单: {result['booking']}")


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        tenant_id = resolve_tenant_id(db, settings.DEFAULT_TENANT_NAME)
    finally:
        db.close()

    print(f"租户: {settings.DEFAULT_TENANT_NAME} (id={tenant_id})\n")

    # 1) 单轮用例(各自独立会话)
    test_cases = [
        "你们几点开门几点关门?",
        "全身推拿多少钱?",
        "帮我预约明天下午3点的肩颈推拿,我叫张三",
        "我要找人工客服投诉",
        "你们店在哪里?",
    ]
    for i, query in enumerate(test_cases, 1):
        print_result(i, query, invoke(query, f"demo-{i}", tenant_id))

    # 2) 多轮回话记忆演示(同一会话连续 4 轮:预约 → 补全 → 改期 → 取消)
    print("\n" + "=" * 80)
    print("多轮回话记忆演示(同一会话 s-memory):")
    memory_turns = [
        "我想预约肩颈推拿",
        "明天下午3点,我叫赵六",
        "改到周四下午5点吧",
        "取消预约",
    ]
    for j, query in enumerate(memory_turns, 1):
        print_result(j, query, invoke(query, "s-memory", tenant_id))


if __name__ == "__main__":
    main()
