"""用户画像服务:行为记录 → 偏好聚合 → 个性化回答与回访判断。

对应原项目 user_behavior 三件套(behavior_recorder / pattern_analyzer /
preference_manager)的平台化升级:
- 事件溯源:UserBehavior 存原始行为,画像(UserProfile)由聚合实时计算并缓存
- 平台化:按 tenant_id + session_id 隔离,不依赖单机内存
- 回访话术:基于画像数据拼装模板文案(供 /api/profile 展示与回流引导,不冒充 LLM)
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.db.models import UserBehavior, UserProfile

# 行为类型
ACTION_CONSULTATION = "consultation"
ACTION_APPOINTMENT = "appointment"
ACTION_CANCEL = "cancel"
ACTION_RESCHEDULE = "reschedule"


# ---------------------------------------------------------------------------
# 行为记录
# ---------------------------------------------------------------------------
def record_behavior(
    db: Session,
    tenant_id: int,
    session_id: str,
    action_type: str,
    action_data: dict | None = None,
) -> UserBehavior | None:
    """记录一条用户行为,并同步更新用户画像。失败返回 None(不阻断主流程)。"""
    if not session_id:
        return None
    try:
        behavior = UserBehavior(
            tenant_id=tenant_id,
            session_id=session_id,
            action_type=action_type,
            action_data=action_data or {},
        )
        db.add(behavior)
        # 先 flush,确保聚合查询能读到本条行为(autoflush=False 的会话也正确)
        db.flush()
        # 业务动作同步更新画像缓存
        if action_type in (ACTION_APPOINTMENT, ACTION_CANCEL, ACTION_RESCHEDULE):
            _refresh_profile(db, tenant_id, session_id)
        db.commit()
        return behavior
    except Exception:
        db.rollback()
        return None


def list_behaviors(
    db: Session,
    tenant_id: int,
    session_id: str,
    action_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """查询用户行为记录(按时间倒序)。"""
    q = db.query(UserBehavior).filter(
        UserBehavior.tenant_id == tenant_id,
        UserBehavior.session_id == session_id,
    )
    if action_type:
        q = q.filter(UserBehavior.action_type == action_type)
    rows = q.order_by(UserBehavior.id.desc()).limit(limit).all()
    return [
        {
            "action_type": b.action_type,
            "action_data": b.action_data,
            "created_at": b.created_at.isoformat() if b.created_at else None,
        }
        for b in rows
    ]


# ---------------------------------------------------------------------------
# 画像聚合
# ---------------------------------------------------------------------------
def get_profile(db: Session, tenant_id: int, session_id: str) -> dict:
    """获取用户画像摘要:偏好 + 行为统计(无记录时返回空画像)。"""
    profile = (
        db.query(UserProfile)
        .filter(UserProfile.tenant_id == tenant_id, UserProfile.session_id == session_id)
        .first()
    )
    if profile is None:
        return {
            "session_id": session_id,
            "favorite_service": "",
            "favorite_time_period": "",
            "favorite_duration": "",
            "total_appointments": 0,
            "last_appointment_at": "",
            "has_profile": False,
        }
    return {
        "session_id": session_id,
        "favorite_service": profile.favorite_service,
        "favorite_time_period": profile.favorite_time_period,
        "favorite_duration": profile.favorite_duration,
        "total_appointments": profile.total_appointments,
        "last_appointment_at": profile.last_appointment_at,
        "has_profile": bool(
            profile.favorite_service
            or profile.favorite_time_period
            or profile.favorite_duration
        ),
    }


def _refresh_profile(db: Session, tenant_id: int, session_id: str) -> None:
    """从行为记录聚合最新偏好,更新 UserProfile 缓存(upsert)。"""
    behaviors = (
        db.query(UserBehavior)
        .filter(
            UserBehavior.tenant_id == tenant_id,
            UserBehavior.session_id == session_id,
            UserBehavior.action_type.in_(
                [ACTION_APPOINTMENT, ACTION_CANCEL, ACTION_RESCHEDULE]
            ),
        )
        .order_by(UserBehavior.id.asc())
        .all()
    )
    if not behaviors:
        return

    service_counter: Counter[str] = Counter()
    period_counter: Counter[str] = Counter()
    duration_counter: Counter[str] = Counter()
    appointments = [b for b in behaviors if b.action_type == ACTION_APPOINTMENT]

    for b in behaviors:
        data = b.action_data or {}
        if b.action_type == ACTION_APPOINTMENT:
            if data.get("service_type"):
                service_counter[str(data["service_type"])] += 1
            if data.get("appointment_time"):
                period = _infer_time_period(str(data["appointment_time"]))
                if period:
                    period_counter[period] += 1
            if data.get("duration"):
                duration_counter[str(data["duration"])] += 1

    last = appointments[-1] if appointments else None
    last_at = last.action_data.get("appointment_time") if last else ""
    if last and last.created_at:
        # 优先用事件时间;若只有相对表达(如「明天」),退回创建时间
        last_at = last_at or last.created_at.strftime("%Y-%m-%d %H:%M")

    profile = (
        db.query(UserProfile)
        .filter(UserProfile.tenant_id == tenant_id, UserProfile.session_id == session_id)
        .first()
    )
    if profile is None:
        profile = UserProfile(tenant_id=tenant_id, session_id=session_id)
        db.add(profile)

    profile.favorite_service = service_counter.most_common(1)[0][0] if service_counter else ""
    profile.favorite_time_period = period_counter.most_common(1)[0][0] if period_counter else ""
    profile.favorite_duration = duration_counter.most_common(1)[0][0] if duration_counter else ""
    profile.total_appointments = len(appointments)
    profile.last_appointment_at = last_at or profile.last_appointment_at


def _infer_time_period(text: str) -> str:
    """从预约时间表达推断时段:上午/下午/晚上。"""
    if any(k in text for k in ("上午", "早上", "早晨")):
        return "上午"
    if any(k in text for k in ("晚上", "傍晚", "夜间", "晚")):
        return "晚上"
    if any(k in text for k in ("下午", "中午", "午")):
        return "下午"
    # 数字小时推断
    import re

    m = re.search(r"(\d{1,2})\s*[:：点]", text)
    if m:
        hour = int(m.group(1))
        if 6 <= hour < 12:
            return "上午"
        if 12 <= hour < 18:
            return "下午"
        return "晚上"
    return ""


# ---------------------------------------------------------------------------
# 回访与个性化话术
# ---------------------------------------------------------------------------
def should_return_reminder(
    db: Session,
    tenant_id: int,
    session_id: str,
    days_threshold: int = 30,
    min_appointments: int = 1,
) -> bool:
    """判断是否应发送回访提醒:有预约且距最近预约超过阈值天数。"""
    profile = get_profile(db, tenant_id, session_id)
    if profile["total_appointments"] < min_appointments or not profile["last_appointment_at"]:
        return False

    last_at = profile["last_appointment_at"]
    try:
        dt = datetime.fromisoformat(last_at.replace("Z", "+00:00").replace(" ", "T"))
    except ValueError:
        return False
    # 相对表达(如「明天」)无法解析,按最近事件时间兜底
    days_since = (datetime.now() - dt).days
    return days_since >= days_threshold


def generate_return_message(
    db: Session | None, tenant_id: int, session_id: str, profile: dict | None = None
) -> str:
    """生成个性化回访话术(基于画像的模板文案,供 /api/profile 与回访提醒使用)。

    profile 可传已查好的画像(避免重复查库);为空时内部查询。
    """
    profile = profile or get_profile(db, tenant_id, session_id)
    if not profile["has_profile"]:
        return "您好!好久没见了,要不要再来体验一次放松一下?"

    parts = ["您好!好久没见了"]
    if profile["favorite_service"]:
        parts.append(f"您之前常做的「{profile['favorite_service']}」最近有优惠")
    else:
        parts.append("最近有新优惠活动")
    if profile["favorite_time_period"]:
        parts.append(f"按您习惯的{profile['favorite_time_period']}时段")
    if profile["favorite_duration"]:
        parts.append(f"预约{profile['favorite_duration']}分钟的体验")
    parts.append("要不要帮您安排一下?")
    return "、".join(parts[:-1]) + "," + parts[-1] if len(parts) > 1 else parts[0]


def build_personalized_context(
    db: Session | None, tenant_id: int, session_id: str, profile: dict | None = None
) -> str:
    """生成注入咨询/业务提示词的个性化上下文(供 Agent 使用)。

    profile 可传已查好的画像(避免重复查库);为空时内部查询。
    """
    profile = profile or get_profile(db, tenant_id, session_id)
    if not profile["has_profile"]:
        return ""
    bits = []
    if profile["favorite_service"]:
        bits.append(f"常做服务:{profile['favorite_service']}")
    if profile["favorite_time_period"]:
        bits.append(f"常约时段:{profile['favorite_time_period']}")
    if profile["favorite_duration"]:
        bits.append(f"常约时长:{profile['favorite_duration']}分钟")
    bits.append(f"累计预约:{profile['total_appointments']}次")
    return "【用户画像】" + "、".join(bits)
