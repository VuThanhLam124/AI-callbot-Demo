"""Tool system for telecom callbot using in-memory sample data."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone

from app import sample_data
from app.schemas import CallbackRequest, LeadRequest, RecommendPlanRequest
from app.service import (
    create_callback,
    create_lead,
    get_plan,
    get_subscriber,
    list_plans,
    recommend_plans,
)


def _normalize(text: str) -> str:
    lowered = text.lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    norm = _normalize(text)
    return any(keyword in norm for keyword in keywords)


def _format_vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".")


def _extract_budget_vnd(text: str) -> int | None:
    norm = _normalize(text)

    match_k = re.search(r"(\d+)\s*(k|nghin|ngan)\b", norm)
    if match_k:
        return int(match_k.group(1)) * 1000

    match_m = re.search(r"(\d+)\s*(tr|trieu)\b", norm)
    if match_m:
        return int(match_m.group(1)) * 1_000_000

    match_raw = re.search(r"\b(\d{5,7})\b", norm)
    if match_raw:
        return int(match_raw.group(1))

    return None


def _extract_priority(text: str) -> str:
    norm = _normalize(text)
    if any(k in norm for k in ("thoai", "noi mang", "ngoai mang", "phut", "goi dien")):
        return "call"
    if any(k in norm for k in ("re", "tiet kiem", "gia", "chi phi", "ngan sach")):
        return "price"
    return "data"


def _extract_offer_selection(text: str, state: dict) -> str | None:
    norm = _normalize(text)
    last_offer_ids = list(state.get("last_offer_ids", []))

    if not last_offer_ids:
        return None

    idx_match = re.search(r"\b([1-3])\b", norm)
    if idx_match:
        idx = int(idx_match.group(1)) - 1
        if 0 <= idx < len(last_offer_ids):
            return last_offer_ids[idx]

    for offer_id in last_offer_ids:
        if _normalize(offer_id) in norm:
            return offer_id

    return last_offer_ids[0]


class TelecomToolSystem:
    """Rule-based tool router to make chatbot responses grounded in sample data."""

    def handle(
        self,
        user_text: str,
        phone: str,
        state: dict | None = None,
    ) -> tuple[bool, str, dict, str]:
        state = dict(state or {})
        state.setdefault("last_offer_ids", [])
        state.setdefault("last_recommend_plan_ids", [])
        state.setdefault("last_tool", "")

        if _contains_any(
            user_text,
            ("uu dai sim", "khuyen mai sim", "goi sim", "sim moi", "mua sim", "danh sach sim"),
        ):
            state["last_offer_ids"] = [offer["id"] for offer in sample_data.SIM_OFFERS[:3]]
            lines = [
                "Hiện có các ưu đãi SIM sau:",
            ]
            for idx, offer in enumerate(sample_data.SIM_OFFERS[:3], start=1):
                lines.append(
                    f"{idx}. {offer['name']} - {_format_vnd(offer['sim_price_vnd'])} VND "
                    f"({offer['bundle']})"
                )
            lines.append("Anh/chị muốn chọn gói nào? (ví dụ: gói 1)")
            reply = "\n".join(lines)
            state["last_tool"] = "list_sim_offers"
            return True, reply, state, "list_sim_offers"

        if _contains_any(
            user_text,
            ("goi lai", "hen goi lai", "ban", "de sau", "goi sau"),
        ):
            callback = create_callback(
                CallbackRequest(
                    phone=phone or "0987999999",
                    requested_time=datetime.now(timezone.utc) + timedelta(hours=1),
                    reason=f"Khách yêu cầu gọi lại: {user_text}",
                )
            )
            reply = (
                f"Em đã tạo lịch gọi lại thành công. Mã lịch: {callback.callback_id}. "
                "Dự kiến nhân viên sẽ liên hệ lại trong vòng 1 giờ."
            )
            state["last_tool"] = "create_callback"
            return True, reply, state, "create_callback"

        if _contains_any(
            user_text,
            ("dong y", "dang ky", "chot", "mua ngay", "lay goi", "chon goi"),
        ):
            selected_offer_id = _extract_offer_selection(user_text, state)
            if not selected_offer_id:
                selected_offer_id = sample_data.SIM_OFFERS[0]["id"]
            lead = create_lead(
                LeadRequest(
                    customer_name="Khách hàng demo",
                    phone=phone or "0987999999",
                    interested_offer_id=selected_offer_id,
                    note=f"Khách phản hồi: {user_text}",
                )
            )
            selected_offer = next(
                (offer for offer in sample_data.SIM_OFFERS if offer["id"] == selected_offer_id),
                sample_data.SIM_OFFERS[0],
            )
            reply = (
                f"Đã tạo yêu cầu mua {selected_offer['name']} (mã lead: {lead.lead_id}). "
                "Nhân viên VNPost Telecom sẽ gọi xác nhận trong 15 phút."
            )
            state["last_tool"] = "create_lead"
            return True, reply, state, "create_lead"

        if _contains_any(
            user_text,
            ("tra cuu", "goi dang dung", "goi hien tai", "thue bao", "tai khoan"),
        ):
            subscriber = get_subscriber(phone)
            if not subscriber:
                reply = (
                    f"Em chưa tìm thấy thuê bao {phone} trong dữ liệu mẫu. "
                    "Anh/chị kiểm tra lại số điện thoại giúp em."
                )
                state["last_tool"] = "lookup_subscriber"
                return True, reply, state, "lookup_subscriber"

            plan = get_plan(subscriber["current_plan_id"])
            if not plan:
                reply = (
                    f"Thuê bao {subscriber['phone']} đang có dữ liệu thuê bao, "
                    "nhưng chưa có thông tin gói cước hiện tại."
                )
                state["last_tool"] = "lookup_subscriber"
                return True, reply, state, "lookup_subscriber"

            reply = (
                f"Số {subscriber['phone']} ({subscriber['name']}) đang dùng gói {plan['name']} "
                f"({plan['id']}) giá {_format_vnd(plan['monthly_price_vnd'])} VND/tháng, "
                f"{plan['data_gb_per_day']}GB/ngày. "
                f"Mức dùng trung bình: {subscriber['avg_data_usage_gb_per_day']}GB/ngày, "
                f"{subscriber['avg_call_minutes_per_month']} phút/tháng."
            )
            state["last_tool"] = "lookup_subscriber"
            return True, reply, state, "lookup_subscriber"

        if _contains_any(
            user_text,
            ("goi nao phu hop", "de xuat", "tu van", "nen dung goi nao"),
        ):
            budget = _extract_budget_vnd(user_text) or 250000
            priority = _extract_priority(user_text)
            needs_5g = _contains_any(user_text, ("5g",))
            picks = recommend_plans(
                RecommendPlanRequest(
                    phone=phone or None,
                    budget_max_vnd=budget,
                    priority=priority,  # type: ignore[arg-type]
                    needs_5g=needs_5g,
                )
            )

            if not picks:
                reply = (
                    f"Em chưa tìm được gói phù hợp với ngân sách {_format_vnd(budget)} VND. "
                    "Anh/chị có thể tăng ngân sách hoặc bỏ điều kiện 5G."
                )
                state["last_tool"] = "recommend_plan"
                return True, reply, state, "recommend_plan"

            state["last_recommend_plan_ids"] = [item.plan_id for item in picks]
            lines = [f"Em đề xuất các gói phù hợp (ngân sách ≤ {_format_vnd(budget)} VND):"]
            for idx, item in enumerate(picks, start=1):
                plan = get_plan(item.plan_id)
                if not plan:
                    continue
                lines.append(
                    f"{idx}. {plan['name']} ({item.plan_id}) - "
                    f"{_format_vnd(plan['monthly_price_vnd'])} VND/tháng. "
                    f"Lý do: {item.reason}."
                )
            reply = "\n".join(lines)
            state["last_tool"] = "recommend_plan"
            return True, reply, state, "recommend_plan"

        if _contains_any(
            user_text,
            ("goi cuoc", "bang gia", "gia goi", "goi data", "goi 5g"),
        ):
            budget = _extract_budget_vnd(user_text)
            needs_5g = _contains_any(user_text, ("5g",))
            plans = list_plans(
                budget_max_vnd=budget,
                needs_5g=True if needs_5g else None,
            )
            if not plans:
                cond = []
                if budget:
                    cond.append(f"ngân sách ≤ {_format_vnd(budget)} VND")
                if needs_5g:
                    cond.append("hỗ trợ 5G")
                cond_msg = ", ".join(cond) if cond else "điều kiện đã chọn"
                reply = f"Không có gói cước phù hợp với {cond_msg}."
                state["last_tool"] = "list_plans"
                return True, reply, state, "list_plans"

            lines = ["Các gói cước hiện có trong dữ liệu mẫu:"]
            for plan in plans[:5]:
                lines.append(
                    f"- {plan['name']} ({plan['id']}): {_format_vnd(plan['monthly_price_vnd'])} VND/tháng, "
                    f"{plan['data_gb_per_day']}GB/ngày, {plan['call_minutes_onnet']} phút nội mạng."
                )
            reply = "\n".join(lines)
            state["last_tool"] = "list_plans"
            return True, reply, state, "list_plans"

        return False, "", state, ""
