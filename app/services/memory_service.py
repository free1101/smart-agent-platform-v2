"""长期记忆服务(跨会话语义记忆)。

解决的问题:传统客服 Agent 的记忆被绑定在"一次会话(session)"内,
用户换一个新会话(新 session_id)后,之前说的偏好/已办事项全部丢失,
只能重新问一遍。这对真实场景是不可接受的。

做法(可持久化的"记忆层"):
  - 会话中产出关键事实时(预约成功、取消预约等),
    `remember()` 把它落库为一条 MemoryEntry(按 tenant + user_key 归属);
  - 新会话开始时,`recall()` 按 user_key 检索相关记忆,注入主管上下文,
    让 Agent 在新会话里"还记得老客户"。

user_key 是什么: 能唯一标识一个终端用户的字符串。真实系统常用
手机号/用户 ID;本项目演示用 chat 请求里的 user_key 字段(不传时退化为
session_id,那样就退化成"同会话内记忆",不影响演示召回链路)。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.db.models import MemoryEntry

# 每个 user_key 最多保留的记忆条数(防无限增长)
MAX_MEMORIES_PER_USER = 100
# 召回时默认取最近几天内的记忆
RECALL_DAYS = 90


def remember(
    db: Session,
    tenant_id: int,
    user_key: str,
    session_id: str,
    content: str,
    memory_type: str = "fact",
    importance: int = 5,
) -> MemoryEntry | None:
    """写入一条长期记忆(失败静默,不阻断主流程)。"""
    try:
        entry = MemoryEntry(
            tenant_id=tenant_id,
            user_key=user_key,
            session_id=session_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)

        # 顺手裁剪:超过上限时删除该用户最旧的记忆(保持表体积可控)
        stale = (
            db.query(MemoryEntry)
            .filter(
                MemoryEntry.tenant_id == tenant_id,
                MemoryEntry.user_key == user_key,
            )
            .order_by(MemoryEntry.created_at.desc())
            .offset(MAX_MEMORIES_PER_USER)
            .all()
        )
        for s in stale:
            db.delete(s)
        if stale:
            db.commit()
        return entry
    except Exception:  # noqa: BLE001 - 记忆写入失败不应影响预约/咨询主流程
        db.rollback()
        return None


def _tokenize(text: str) -> set[str]:
    """极简中文分词:切出 2/3 字片段,用于关键词重叠打分。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    if not text:
        return set()
    tokens: set[str] = set()
    for ch_start in range(len(text) - 1):
        tokens.add(text[ch_start : ch_start + 2])
    for ch_start in range(len(text) - 2):
        tokens.add(text[ch_start : ch_start + 3])
    return tokens


def recall(
    db: Session,
    tenant_id: int,
    user_key: str,
    query: str = "",
    limit: int = 5,
) -> list[dict[str, str | int]]:
    """按 user_key 召回记忆,与当前问题相关度高的排前面。

    打分思路(和知识检索同款"混合"味儿,只是轻量版):
      - 关键词重叠度(与用户当前问题);
      - importance 权重;
      - 时间衰减(越近越优先)。
    """
    if not user_key:
        return []

    since = datetime.utcnow() - timedelta(days=RECALL_DAYS)
    rows = (
        db.query(MemoryEntry)
        .filter(
            MemoryEntry.tenant_id == tenant_id,
            MemoryEntry.user_key == user_key,
            MemoryEntry.created_at >= since,
        )
        .order_by(MemoryEntry.created_at.desc())
        .limit(MAX_MEMORIES_PER_USER)
        .all()
    )
    if not rows:
        return []

    q_tokens = _tokenize(query) if query else set()
    scored: list[tuple[float, MemoryEntry]] = []
    for m in rows:
        score = 0.0
        # 1) 关键词重叠
        if q_tokens:
            m_tokens = _tokenize(m.content)
            overlap = len(q_tokens & m_tokens)
            if overlap:
                score += overlap * 2.0
        # 2) importance 权重(0~10 → 0~1)
        score += (m.importance or 5) / 10.0
        # 3) 时间衰减: 3 个月内线性降到 60%
        age_days = max(0.0, (datetime.utcnow() - m.created_at).total_seconds() / 86400)
        score *= max(0.6, 1.0 - age_days / (RECALL_DAYS * 2))
        scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[:limit]
    return [
        {
            "memory_type": m.memory_type,
            "content": m.content,
            "created_at": m.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for _, m in picked
    ]


def format_recall(memories: list[dict[str, str | int]]) -> str:
    """把召回的记忆渲染成一段可注入 prompt 的文本。"""
    if not memories:
        return ""
    lines = ["【长期记忆(还记得这位用户)】"]
    for m in memories:
        tag = {
            "preference": "偏好",
            "appointment": "预约记录",
            "fact": "已知事实",
        }.get(str(m.get("memory_type")), "记忆")
        lines.append(f"- [{tag}] {m['content']}")
    return "\n".join(lines)
