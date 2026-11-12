"""业务 Agent:处理预约/下单等业务动作。

多轮记忆能力:
1. 跨轮补全:本轮信息缺失时,自动从历史消息中补齐(服务/时间/姓名)
2. 改期:识别「改到/改期/换个时间」等动作,更新最近一笔未取消的工单时间
3. 取消:识别「取消/退掉」等动作,将最近工单置为 cancelled

工具策略(两层):
1. MCP 注册工具优先:进程内直调 @mcp.tool() 注册的 create_booking
   (数据逻辑与走 MCP 协议一致,不依赖独立 server;
    完整协议链路由 scripts/test_mcp_client.py 单独验证)
2. 本地兜底:进程内直调异常时直接写 booking_tickets 表,保证流程不中断

流程:提取业务要素(服务/时间/姓名)→ 建工单 → 确认回复。
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from app.agents.state import AgentState, current_user_input, history_user_texts
from app.llm.provider import create_chat_model

# 时间表达规则:日期/时段/时间点三个部件尽量整体匹配(如「明天下午3点」)
_DAY_WORDS = (
    r"今天|明天|后天|下周一|下周二|下周三|下周四|下周五|下周六|下周日|"
    r"周一|周二|周三|周四|周五|周六|周日"
)
_PERIOD_WORDS = r"上午|中午|下午|晚上|早上|傍晚"
_CLOCK_POINT = r"\d{1,2}\s*[:：]\s*\d{2}|\d{1,2}\s*点(?:半|钟)?"
_TIME_RE = re.compile(
    rf"(?:({_DAY_WORDS})(?:的)?\s*)?"
    rf"(?:({_PERIOD_WORDS})\s*)?"
    rf"({_CLOCK_POINT})?"
)


def _extract_time(text: str) -> str:
    """提取用户消息中的完整时间表达(如「明天下午3点」),无则返回空串。"""
    best = ""
    for m in _TIME_RE.finditer(text):
        seg = m.group(0)
        if seg and len(seg) > len(best):
            best = seg
    return best
_NAME_RE = re.compile(r"(?:我叫|我是|姓|名字(?:是|叫)?)\s*([\u4e00-\u9fa5]{2,4})")
# 纯称谓短句:跨轮补全时用户可能只回「王先生」「李女士」「张老师」
_NAME_HONORIFIC_RE = re.compile(r"^([\u4e00-\u9fa5]{1,2}(?:先生|女士|小姐|老师))$")

# 常见服务类型(与示例知识库分类对齐)
_SERVICES = ["全身推拿", "肩颈推拿", "足底按摩", "背部推拿", "精油开背", "头部刮痧"]

# 改期 / 取消动作词(命中即走对应流程)
_ACTION_RESCHEDULE = ["改到", "改期", "修改", "换个时间", "换到", "变更", "调整"]
_ACTION_CANCEL = ["取消", "退掉", "退订", "不去了", "撤销"]


def extract_booking_info(text: str) -> dict:
    """从用户消息中提取预约要素(本地规则,降级用)。"""
    info: dict[str, str] = {}

    m = _extract_time(text)
    if m:
        info["appointment_time"] = m

    m = _NAME_RE.search(text)
    if m:
        info["customer_name"] = m.group(1)
    else:
        honorific = _NAME_HONORIFIC_RE.match(text.strip())
        if honorific:
            info["customer_name"] = honorific.group(1)

    for svc in _SERVICES:
        if svc in text:
            info["service_type"] = svc
            break

    return info


_DURATION_RE = re.compile(r"(\d{2,3})\s*分钟")


def _extract_duration(text: str) -> str:
    """从用户消息中提取预约时长(如「60分钟」),用于画像偏好。"""
    m = _DURATION_RE.search(text)
    return m.group(1) if m else ""


def _record_behavior(tenant_id: int, session_id: str, action_type: str, data: dict) -> None:
    """记录用户行为(失败静默,不阻断预约主流程)。"""
    try:
        from app.services.profile_service import record_behavior

        from app.db.database import SessionLocal

        db = SessionLocal()
        try:
            record_behavior(db, tenant_id, session_id, action_type, data)
        finally:
            db.close()
    except Exception:
        pass


def _remember(
    tenant_id: int,
    user_key: str,
    session_id: str,
    content: str,
    memory_type: str = "fact",
    importance: int = 5,
) -> None:
    """写入一条长期记忆(跨会话,失败静默,不阻断主流程)。"""
    try:
        from app.db.database import SessionLocal
        from app.services.memory_service import remember

        db = SessionLocal()
        try:
            remember(db, tenant_id, user_key, session_id, content, memory_type, importance)
        finally:
            db.close()
    except Exception:
        pass


def _extract_with_llm(text: str) -> dict:
    """LLM 提取预约要素(仅 API 模式,失败返回空)。"""
    try:
        llm = create_chat_model(temperature=0)
        resp = llm.invoke(
            "从用户消息中提取预约信息,只输出 JSON,字段:customer_name(姓名),"
            "service_type(服务类型),appointment_time(预约时间)。没有的字段填空字符串。\n"
            f"消息:{text}"
        )
        content = str(resp.content).strip()
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            import json

            data = json.loads(content[start : end + 1])
            return {k: (v or "") for k, v in data.items() if isinstance(v, str)}
    except Exception:
        pass
    return {}


def _merge_booking_from_text(text: str, booking: dict) -> dict:
    """补缺式合并:把一段用户消息中提取到的预约要素合并进 booking(已有值不覆盖,用于跨轮补全)。"""
    info = extract_booking_info(text)
    if not info:
        info = _extract_with_llm(text)
    for k, v in info.items():
        if v and not booking.get(k):
            booking[k] = v
    return booking


def _overlay_booking_from_text(text: str, booking: dict) -> dict:
    """覆盖式合并:以最新消息提取到的要素为准(可覆盖历史值,用于改期/更新)。"""
    info = extract_booking_info(text)
    if not info:
        info = _extract_with_llm(text)
    booking.update({k: v for k, v in info.items() if v})
    return booking


def _create_booking_via_mcp(booking: dict, tenant_id: int, session_id: str, notes: str) -> dict | None:
    """通过 MCP 注册工具创建预约;失败返回 None(上层走本地兜底)。

    说明:create_booking 在 app/mcp/tools.py 中通过 @mcp.tool() 注册,
    是标准 MCP 工具;进程内直接同步调用它,等价于走 MCP 协议(数据逻辑一致)。
    如需验证完整 MCP 协议链路,运行 scripts/test_mcp_client.py。
    """
    try:
        from app.mcp.tools import create_booking

        result = create_booking(
            tenant_id=tenant_id,
            session_id=session_id,
            service_type=booking.get("service_type", ""),
            appointment_time=booking.get("appointment_time", ""),
            customer_name=booking.get("customer_name", ""),
            notes=notes,
        )
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _create_booking_local(booking: dict, tenant_id: int, session_id: str, notes: str) -> dict:
    """本地建单(离线降级路径)。"""
    from app.db.database import SessionLocal
    from app.db.models import BookingTicket

    db = SessionLocal()
    try:
        ticket = BookingTicket(
            tenant_id=tenant_id,
            session_id=session_id,
            customer_name=booking.get("customer_name", ""),
            service_type=booking.get("service_type", ""),
            appointment_time=booking.get("appointment_time", ""),
            notes=notes,
            status="pending",
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        return {"ticket_id": ticket.id, "status": "pending", "message": f"预约已创建,工单号 T-{ticket.id:04d}"}
    finally:
        db.close()


def _reschedule_recent(tenant_id: int, session_id: str, new_time: str):
    """把最近一笔工单改期到 new_time;无工单返回 None。"""
    from app.db.database import SessionLocal
    from app.db.models import BookingTicket

    db = SessionLocal()
    try:
        ticket = (
            db.query(BookingTicket)
            .filter(
                BookingTicket.tenant_id == tenant_id,
                BookingTicket.session_id == session_id,
                BookingTicket.status != "cancelled",
            )
            .order_by(BookingTicket.id.desc())
            .first()
        )
        if ticket is None:
            return None
        ticket.appointment_time = new_time
        db.commit()
        db.refresh(ticket)  # 刷新属性,避免 session 关闭后 detached 访问失败
        return ticket
    finally:
        db.close()


def _cancel_recent(tenant_id: int, session_id: str):
    """取消该会话最近一笔工单;无工单返回 None。"""
    from app.db.database import SessionLocal
    from app.db.models import BookingTicket

    db = SessionLocal()
    try:
        ticket = (
            db.query(BookingTicket)
            .filter(
                BookingTicket.tenant_id == tenant_id,
                BookingTicket.session_id == session_id,
                BookingTicket.status != "cancelled",
            )
            .order_by(BookingTicket.id.desc())
            .first()
        )
        if ticket is None:
            return None
        ticket.status = "cancelled"
        db.commit()
        db.refresh(ticket)  # 刷新属性,避免 session 关闭后 detached 访问失败
        return ticket
    finally:
        db.close()


def business_node(state: AgentState) -> AgentState:
    """业务节点:取消 → 改期 → 普通预约(含跨轮补全)。"""
    messages = state.get("messages") or []
    user_input = current_user_input(state)
    history = history_user_texts(messages)
    tenant_id = state.get("tenant_id", 1)
    session_id = state.get("session_id", "")
    booking = dict(state.get("booking") or {})

    # ---- 1) 取消 ----
    if any(kw in user_input for kw in _ACTION_CANCEL):
        ticket = _cancel_recent(tenant_id, session_id)
        state["booking"] = booking
        state["awaiting_booking"] = False
        if ticket is not None:
            state["reply"] = (
                f"已为您取消预约(工单号 T-{ticket.id:04d},"
                f"{ticket.service_type or '服务'})。如需重新预约请随时告诉我。"
            )
            _record_behavior(tenant_id, session_id, "cancel", {"service_type": ticket.service_type})
            # 长期记忆:取消也算跨会话该记住的事实
            _remember(
                tenant_id,
                state.get("user_key") or session_id,
                session_id,
                f"用户取消了「{ticket.service_type or '服务'}」的预约(工单 T-{ticket.id:04d})",
                memory_type="appointment",
                importance=6,
            )
        else:
            state["reply"] = "暂时没有找到可取消的预约。您可以先告诉我需要预约什么服务。"
        state["messages"] = [AIMessage(content=state["reply"])]
        return state

    # ---- 2) 改期 ----
    if any(kw in user_input for kw in _ACTION_RESCHEDULE):
        # 历史垫底、最新覆盖:新时间以最新消息为准(可覆盖历史时间),服务/姓名从历史补全
        for t in history:
            _merge_booking_from_text(t, booking)
        _overlay_booking_from_text(user_input, booking)
        new_time = booking.get("appointment_time", "")
        ticket = _reschedule_recent(tenant_id, session_id, new_time) if new_time else None
        state["booking"] = booking
        state["awaiting_booking"] = False
        if ticket is not None:
            state["reply"] = (
                f"已为您改期:{booking.get('service_type') or '服务'}改到 {new_time}。"
                f"工单号 T-{ticket.id:04d},如需再次修改请告诉我。"
            )
            _record_behavior(
                tenant_id,
                session_id,
                "reschedule",
                {
                    "service_type": booking.get("service_type", ""),
                    "appointment_time": new_time,
                },
            )
            # 长期记忆:改期后的最终时间
            _remember(
                tenant_id,
                state.get("user_key") or session_id,
                session_id,
                f"用户把「{booking.get('service_type') or '服务'}」的预约改到 {new_time}"
                f"(工单 T-{ticket.id:04d})",
                memory_type="appointment",
                importance=6,
            )
        elif new_time:
            state["reply"] = "暂时没有找到可改期的预约,请先完成一次预约后再改期。"
        else:
            state["reply"] = "好的,您想改到什么时间呢?请告诉我具体时间。"
        state["messages"] = [AIMessage(content=state["reply"])]
        return state

    # ---- 3) 普通预约:跨轮补全(历史垫底、最新覆盖) ----
    for t in history:
        _merge_booking_from_text(t, booking)
    _overlay_booking_from_text(user_input, booking)
    state["booking"] = booking

    if booking.get("appointment_time"):
        # 优先 MCP 工具,失败本地兜底
        result = _create_booking_via_mcp(booking, tenant_id, session_id, user_input)
        if result is None:
            result = _create_booking_local(booking, tenant_id, session_id, user_input)

        ticket_id = result.get("ticket_id")
        state["awaiting_booking"] = False
        state["reply"] = (
            f"已为您预约:{booking.get('service_type') or '服务'} "
            f"在{booking.get('appointment_time')}。"
            + (f"客户:{booking.get('customer_name')}。" if booking.get("customer_name") else "")
            + f"工单号 T-{ticket_id:04d},如需修改请随时告诉我。"
        )
        # 记录预约行为(画像数据源)
        _record_behavior(
            tenant_id,
            session_id,
            "appointment",
            {
                "service_type": booking.get("service_type", ""),
                "appointment_time": booking.get("appointment_time", ""),
                "duration": _extract_duration(user_input),
            },
        )
        # 长期记忆:预约成功 → 沉淀为跨会话事实(下次换新会话也能想起)
        _remember(
            tenant_id,
            state.get("user_key") or session_id,
            session_id,
            (
                f"用户预约了「{booking.get('service_type') or '服务'}」"
                f",时间为{booking.get('appointment_time') or '待定'}"
                + (f",姓名{booking.get('customer_name')}" if booking.get("customer_name") else "")
                + f"(工单 T-{ticket_id:04d})"
            ),
            memory_type="appointment",
            importance=8,
        )
    else:
        missing = []
        if not booking.get("appointment_time"):
            missing.append("预约时间")
        if not booking.get("service_type"):
            missing.append("服务项目")
        state["awaiting_booking"] = True  # 标记等待补全,下一轮自动延续预约流程
        state["reply"] = f"好的,我来帮您预约。还需要您补充:{'、'.join(missing)}。请问方便告知吗?"

    state["messages"] = [AIMessage(content=state["reply"])]
    return state
