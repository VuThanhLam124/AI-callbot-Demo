"""Simple rule-based callbot for telecom demo."""

from __future__ import annotations

import unicodedata
from datetime import datetime, timezone
from typing import Literal

from app import sample_data
from app.schemas import CallbackRequest, LeadRequest, RecommendPlanRequest
from app.service import create_callback, create_lead, get_subscriber, recommend_plans


def _normalize(text: str) -> str:
    lowered = text.lower().replace("đ", "d")
    normalized = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    norm = _normalize(text)
    return any(kw in norm for kw in keywords)


class TelecomCallbot:
    """Small state machine for inbound and outbound demo conversations."""

    def __init__(self, mode: Literal["inbound", "outbound"], phone: str | None = None):
        self.mode = mode
        self.phone = phone
        self.state = "start"

    def opening(self) -> str:
        if self.mode == "outbound":
            return (
                "Xin chào, em là trợ lý AI VNPost Telecom. "
                "Bên em đang có ưu đãi SIM data 5G và gói cước tiết kiệm. "
                "Anh/chị có thể dành 1 phút để em tư vấn nhanh không?"
            )
        return (
            "Xin chào quý khách, đây là tổng đài AI VNPost Telecom. "
            "Anh/chị cần tra cứu gói cước, tư vấn gói mới hay hỗ trợ khác?"
        )

    def reply(self, user_text: str) -> str:
        if self.mode == "outbound":
            return self._reply_outbound(user_text)
        return self._reply_inbound(user_text)

    def _reply_inbound(self, user_text: str) -> str:
        if _contains_any(
            user_text,
            ("gap nhan vien", "gap nguoi that", "tong dai vien", "chuyen nguoi that"),
        ):
            self.state = "handover"
            return "Em đã ghi nhận. Đang chuyển máy sang tổng đài viên hỗ trợ anh/chị."

        if _contains_any(
            user_text,
            ("tra cuu", "goi dang dung", "goi hien tai", "thue bao", "tai khoan"),
        ):
            if not self.phone:
                return "Anh/chị vui lòng cung cấp số điện thoại để em tra cứu gói đang sử dụng."
            subscriber = get_subscriber(self.phone)
            if not subscriber:
                return "Em chưa tìm thấy thuê bao trong dữ liệu demo. Anh/chị vui lòng thử lại."
            return (
                f"Số {subscriber['phone']} đang dùng gói {subscriber['current_plan_id']}. "
                f"Mức dùng data trung bình {subscriber['avg_data_usage_gb_per_day']}GB/ngày."
            )

        if _contains_any(
            user_text,
            ("tu van", "goi nao phu hop", "nen dung goi nao", "de xuat", "goi cuoc", "de nghi"),
        ):
            request = RecommendPlanRequest(
                phone=self.phone,
                budget_max_vnd=200000,
                priority="data",
                needs_5g=True,
            )
            picks = recommend_plans(request)
            if not picks:
                return "Hiện em chưa tìm được gói phù hợp với tiêu chí vừa rồi."
            best = picks[0]
            return (
                f"Em đề xuất gói {best.plan_id}. Lý do: {best.reason}. "
                "Anh/chị có muốn em đăng ký gói này và nhờ tư vấn viên xác nhận không?"
            )

        if _contains_any(user_text, ("mua sim", "sim moi", "uu dai sim", "khuyen mai sim")):
            offer = sample_data.SIM_OFFERS[0]
            return (
                f"Hiện có {offer['name']} giá {offer['sim_price_vnd']:,} VND, "
                f"ưu đãi: {offer['bundle']}. Anh/chị có quan tâm không?"
            )

        return (
            "Em chưa nhận đúng ý. Anh/chị có thể nói rõ hơn: "
            "tra cứu gói đang dùng, tư vấn gói cước, hoặc gặp tổng đài viên."
        )

    def _reply_outbound(self, user_text: str) -> str:
        if _contains_any(user_text, ("khong", "tu choi", "khong quan tam", "thoi")):
            self.state = "callback_offer"
            return "Em tôn trọng quyết định của anh/chị. Em có thể hẹn gọi lại vào khung giờ tiện hơn được không?"

        if self.state == "callback_offer":
            callback = create_callback(
                CallbackRequest(
                    phone=self.phone or "0987999999",
                    requested_time=datetime.now(timezone.utc),
                    reason=f"Khách hàng yêu cầu gọi lại: {user_text}",
                )
            )
            self.state = "done"
            return f"Em đã đặt lịch gọi lại (mã {callback.callback_id}). Cảm ơn anh/chị."

        if _contains_any(user_text, ("dong y", "co", "tu van", "nghe", "ok", "duoc")):
            self.state = "offer_sent"
            offers = sample_data.SIM_OFFERS[:2]
            return (
                "Em gửi nhanh 2 lựa chọn: "
                f"1) {offers[0]['name']} - {offers[0]['bundle']}; "
                f"2) {offers[1]['name']} - {offers[1]['bundle']}. "
                "Anh/chị muốn chọn gói nào?"
            )

        if self.state == "offer_sent":
            chosen_offer = sample_data.SIM_OFFERS[0]
            lead = create_lead(
                LeadRequest(
                    customer_name="Khách hàng demo",
                    phone=self.phone or "0987999999",
                    interested_offer_id=chosen_offer["id"],
                    note=f"Nhận phản hồi: {user_text}",
                )
            )
            self.state = "done"
            return (
                f"Em đã tạo yêu cầu tư vấn mua SIM (mã {lead.lead_id}). "
                "Nhân viên VNPost sẽ liên hệ xác nhận trong 15 phút."
            )

        return "Anh/chị có muốn nghe ưu đãi SIM mới của VNPost Telecom không?"
