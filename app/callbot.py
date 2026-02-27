"""Simple rule-based callbot for telecom demo."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from app import sample_data
from app.schemas import CallbackRequest, LeadRequest, RecommendPlanRequest
from app.service import create_callback, create_lead, get_subscriber, recommend_plans


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


class TelecomCallbot:
    """Small state machine for inbound and outbound demo conversations."""

    def __init__(self, mode: Literal["inbound", "outbound"], phone: str | None = None):
        self.mode = mode
        self.phone = phone
        self.state = "start"

    def opening(self) -> str:
        if self.mode == "outbound":
            return (
                "Xin chao, em la tro ly AI VNPost Telecom. "
                "Ben em dang co uu dai SIM data 5G va goi cuoc tiet kiem. "
                "Anh/chị co the danh 1 phut de em tu van nhanh khong?"
            )
        return (
            "Xin chao quy khach, day la tong dai AI VNPost Telecom. "
            "Anh/chị can tra cuu goi cuoc, tu van goi moi hay ho tro khac?"
        )

    def reply(self, user_text: str) -> str:
        if self.mode == "outbound":
            return self._reply_outbound(user_text)
        return self._reply_inbound(user_text)

    def _reply_inbound(self, user_text: str) -> str:
        if _contains_any(user_text, ("gap nhan vien", "gap nguoi that", "tong dai vien")):
            self.state = "handover"
            return "Em da ghi nhan. Dang chuyen may sang tong dai vien ho tro anh/chị."

        if _contains_any(user_text, ("tra cuu", "goi dang dung", "goi hien tai", "thue bao")):
            if not self.phone:
                return "Anh/chị vui long cung cap so dien thoai de em tra cuu goi dang su dung."
            subscriber = get_subscriber(self.phone)
            if not subscriber:
                return "Em chua tim thay thue bao trong du lieu demo. Anh/chị vui long thu lai."
            return (
                f"So {subscriber['phone']} dang dung goi {subscriber['current_plan_id']}. "
                f"Muc dung data trung binh {subscriber['avg_data_usage_gb_per_day']}GB/ngay."
            )

        if _contains_any(
            user_text,
            ("tu van", "goi nao phu hop", "nen dung goi nao", "de xuat", "goi cuoc"),
        ):
            request = RecommendPlanRequest(
                phone=self.phone,
                budget_max_vnd=200000,
                priority="data",
                needs_5g=True,
            )
            picks = recommend_plans(request)
            if not picks:
                return "Hien em chua tim duoc goi phu hop voi tieu chi vua roi."
            best = picks[0]
            return (
                f"Em de xuat goi {best.plan_id}. Ly do: {best.reason}. "
                "Anh/chị co muon em dang ky goi nay va nhan tu van vien xac nhan khong?"
            )

        if _contains_any(user_text, ("mua sim", "sim moi", "uu dai sim")):
            offer = sample_data.SIM_OFFERS[0]
            return (
                f"Hien co {offer['name']} gia {offer['sim_price_vnd']:,} VND, "
                f"uu dai: {offer['bundle']}. Anh/chị co quan tam khong?"
            )

        return (
            "Em chua nhan dung y. Anh/chị co the noi ro hon: "
            "tra cuu goi dang dung, tu van goi cuoc, hoac gap tong dai vien."
        )

    def _reply_outbound(self, user_text: str) -> str:
        if _contains_any(user_text, ("khong", "tu choi", "khong quan tam")):
            self.state = "callback_offer"
            return "Em ton trong quyet dinh cua anh/chị. Em co the hen goi lai vao khung gio tien hon duoc khong?"

        if self.state == "callback_offer":
            callback = create_callback(
                CallbackRequest(
                    phone=self.phone or "0987999999",
                    requested_time=datetime.now(timezone.utc),
                    reason=f"Customer requested callback: {user_text}",
                )
            )
            self.state = "done"
            return f"Em da dat lich goi lai (ma {callback.callback_id}). Cam on anh/chị."

        if _contains_any(user_text, ("dong y", "co", "tu van", "nghe")):
            self.state = "offer_sent"
            offers = sample_data.SIM_OFFERS[:2]
            return (
                "Em gui nhanh 2 lua chon: "
                f"1) {offers[0]['name']} - {offers[0]['bundle']}; "
                f"2) {offers[1]['name']} - {offers[1]['bundle']}. "
                "Anh/chị muon chon goi nao?"
            )

        if self.state == "offer_sent":
            chosen_offer = sample_data.SIM_OFFERS[0]
            lead = create_lead(
                LeadRequest(
                    customer_name="Khach hang demo",
                    phone=self.phone or "0987999999",
                    interested_offer_id=chosen_offer["id"],
                    note=f"Nhan phan hoi: {user_text}",
                )
            )
            self.state = "done"
            return (
                f"Em da tao yeu cau tu van mua SIM (ma {lead.lead_id}). "
                "Nhan vien VNPost se lien he xac nhan trong 15 phut."
            )

        return "Anh/chị co muon nghe uu dai SIM moi cua VNPost Telecom khong?"
