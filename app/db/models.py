"""数据模型:平台化设计,以 tenant(行业/租户)隔离知识库与业务数据。

通用智能客服平台,可同时服务多个行业:
- 知识库按 tenant 隔离(每个行业一套语料)
- 会话、工单(预约/订单)同样按 tenant 隔离
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Tenant(Base):
    """租户/行业:例如「健康养生」「电商零售」……"""

    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    industry: Mapped[str] = mapped_column(String(50), default="通用")  # 行业标签
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    docs: Mapped[list["KnowledgeDoc"]] = relationship(back_populates="tenant")


class KnowledgeDoc(Base):
    """知识文档:每个租户一套,支持关键词与向量双重检索。"""

    __tablename__ = "knowledge_docs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    content: Mapped[str] = mapped_column(Text)  # 知识正文
    category: Mapped[str] = mapped_column(String(50), index=True)  # 分类,如「营业时间」
    keywords: Mapped[list] = mapped_column(JSON, default=list)  # 扩展关键词
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)  # 稠密向量
    source: Mapped[str] = mapped_column(String(200), default="")  # 来源(如 seed 文件名)
    # 摄取溯源字段(旧库由 init_db 的轻量迁移补列,默认空)
    source_hash: Mapped[str] = mapped_column(String(64), default="", index=True)  # 整篇文件 hash
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)  # 单块内容 hash
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)  # 该块在文档内的序号
    doc_group: Mapped[str] = mapped_column(String(200), default="")  # 同一次导入的逻辑文档名
    page: Mapped[int] = mapped_column(Integer, default=0)  # PDF 页码(从 1 开始,0 表示无页码)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    tenant: Mapped[Tenant] = relationship(back_populates="docs")


class BookingTicket(Base):
    """业务工单:预约/订单等需要执行动作的业务,由业务 Agent 创建。"""

    __tablename__ = "booking_tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    customer_name: Mapped[str] = mapped_column(String(50), default="")
    service_type: Mapped[str] = mapped_column(String(50), default="")  # 预约的服务/商品
    appointment_time: Mapped[str] = mapped_column(String(50), default="")  # 预约时间
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/done/cancelled
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UserBehavior(Base):
    """用户行为记录:记录咨询/预约/取消/改期等行为,用于个性化画像与回访。

    设计:每次会话的关键动作追加一条事件(事件溯源思路),画像由聚合查询实时计算,
    避免冗余维护。原项目 user_behavior/behavior_recorder 的升级版。
    """

    __tablename__ = "user_behaviors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    action_type: Mapped[str] = mapped_column(String(20), index=True)  # consultation/appointment/cancel/reschedule
    action_data: Mapped[dict] = mapped_column(JSON, default=dict)  # 行为上下文(服务/时间/时长等)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class UserProfile(Base):
    """用户画像:由 UserBehavior 聚合而来,缓存常用偏好供个性化回答。

    偏好字段:
    - favorite_service:    高频服务(如「肩颈推拿」)
    - favorite_time_period: 高频时段(上午/下午/晚上)
    - favorite_duration:    高频时长(如 60)
    - total_appointments:   累计预约次数
    - last_appointment_at:  最近一次预约时间(用于回访判断)
    """

    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    favorite_service: Mapped[str] = mapped_column(String(50), default="")
    favorite_time_period: Mapped[str] = mapped_column(String(20), default="")  # 上午/下午/晚上
    favorite_duration: Mapped[str] = mapped_column(String(20), default="")
    total_appointments: Mapped[int] = mapped_column(Integer, default=0)
    last_appointment_at: Mapped[str] = mapped_column(String(50), default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class MemoryEntry(Base):
    """长期记忆(跨会话语义记忆):按用户标识沉淀"关键事实",新会话可召回。

    与 UserBehavior 的区别:
    - UserBehavior = 行为事件流(按 session, 事件溯源, 用于聚合画像)
    - MemoryEntry   = 提炼后的事实(按 user_key 跨会话保留,如「用户喜欢下午的肩颈推拿」)

    典型事实类型 memory_type:
    - preference  偏好(喜欢的服务/时段/时长)
    - appointment 预约动作结果(约了什么/何时)
    - fact        对话中提到的个人事实(如称呼/特殊要求)
    """

    __tablename__ = "memory_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    user_key: Mapped[str] = mapped_column(String(64), index=True)  # 用户标识(手机号/账号),跨会话
    session_id: Mapped[str] = mapped_column(String(64), default="")  # 产生该记忆的会话
    memory_type: Mapped[str] = mapped_column(String(20), default="fact", index=True)
    content: Mapped[str] = mapped_column(Text)  # 事实文本
    importance: Mapped[int] = mapped_column(Integer, default=5)  # 1-10,越高越优先被召回
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class IngestionRecord(Base):
    """摄取记录:一次文件导入的完整快照。

    幂等设计:同一 (tenant_id, source_hash) 已有成功记录时默认跳过,
    除非显式传 force=True;失败记录可重试。
    """

    __tablename__ = "ingestion_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    filename: Mapped[str] = mapped_column(String(255), default="")
    file_type: Mapped[str] = mapped_column(String(20), default="")  # pdf/docx/md/txt
    source_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    status: Mapped[str] = mapped_column(String(20), default="success", index=True)  # success/failed/skipped
    chunks: Mapped[int] = mapped_column(Integer, default=0)  # 写入的块数
    images: Mapped[int] = mapped_column(Integer, default=0)  # 识别并描述成功的图片数
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")  # 失败原因(首行)
    trace_id: Mapped[str] = mapped_column(String(32), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class TraceRun(Base):
    """一次链路执行的根记录:kind=query(检索问答)或 ingestion(摄取)。"""

    __tablename__ = "trace_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(20), default="query", index=True)
    tenant_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    query: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="success")  # success/failed
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class TraceStage(Base):
    """链路中的一个阶段:按 seq 顺序展示为瀑布图。payload 为 JSON 快照(已截断)。"""

    __tablename__ = "trace_stages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(32), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    name: Mapped[str] = mapped_column(String(50), default="")  # bm25/dense/fuse/rerank/load/caption...
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class EvalRun(Base):
    """一次评测报告摘要:用于评估面板做历史回归对比。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    total: Mapped[int] = mapped_column(Integer, default=0)
    top_k: Mapped[int] = mapped_column(Integer, default=3)
    top1_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    top3_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
