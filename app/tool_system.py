"""Telecom tools that expose structured data for LLM-assisted responses."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

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


def _to_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        clean = value.strip().replace(".", "").replace(",", "")
        if clean.isdigit():
            return int(clean)
    return default


def _to_bool(value: Any, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1", "yes", "co", "có"}:
            return True
        if norm in {"false", "0", "no", "khong", "không"}:
            return False
    return default


class TelecomToolSystem:
    """State-aware tool catalog for telecom domain."""

    def __init__(self) -> None:
        self._handlers = {
            "count_mobile_services": self._count_mobile_services,
            "list_mobile_service_groups": self._list_mobile_service_groups,
            "list_mobile_plans": self._list_mobile_plans,
            "recommend_mobile_plans": self._recommend_mobile_plans,
            "list_sim_offers": self._list_sim_offers,
            "lookup_subscriber": self._lookup_subscriber,
            "create_lead": self._create_lead,
            "create_callback": self._create_callback,
        }

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "count_mobile_services",
                "description": "Dem so nhom dich vu va tong so san pham trong du lieu mau.",
                "arguments": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "list_mobile_service_groups",
                "description": "Tra ve cac nhom dich vu vien thong di dong dang co.",
                "arguments": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "list_mobile_plans",
                "description": (
                    "Lay danh sach goi cuoc di dong theo bo loc gia/5G/"
                    "muc uu tien data-call-price."
                ),
                "arguments": {
                    "type": "object",
                    "properties": {
                        "budget_max_vnd": {"type": "integer"},
                        "needs_5g": {"type": "boolean"},
                        "focus": {"type": "string", "enum": ["data", "call", "price"]},
                        "limit": {"type": "integer"},
                    },
                    "required": [],
                },
            },
            {
                "name": "recommend_mobile_plans",
                "description": "De xuat goi cuoc phu hop theo nhu cau khach hang.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "budget_max_vnd": {"type": "integer"},
                        "priority": {"type": "string", "enum": ["data", "call", "price"]},
                        "needs_5g": {"type": "boolean"},
                        "segment": {"type": "string"},
                        "phone": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "list_sim_offers",
                "description": "Lay danh sach uu dai SIM moi co trong du lieu mau.",
                "arguments": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer"}},
                    "required": [],
                },
            },
            {
                "name": "lookup_subscriber",
                "description": "Tra cuu thong tin thue bao va goi dang su dung.",
                "arguments": {
                    "type": "object",
                    "properties": {"phone": {"type": "string"}},
                    "required": [],
                },
            },
            {
                "name": "create_lead",
                "description": "Tao lead tu van mua SIM/goi cuoc.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "customer_name": {"type": "string"},
                        "phone": {"type": "string"},
                        "interested_offer_id": {"type": "string"},
                        "offer_index": {"type": "integer"},
                        "note": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "create_callback",
                "description": "Tao lich hen goi lai cho khach hang.",
                "arguments": {
                    "type": "object",
                    "properties": {
                        "phone": {"type": "string"},
                        "reason": {"type": "string"},
                        "hours_from_now": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        ]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        phone: str,
        state: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        safe_state = dict(state or {})
        safe_state.setdefault("last_offer_ids", [])
        safe_state.setdefault("last_plan_ids", [])
        safe_state.setdefault("last_tool", "")

        handler = self._handlers.get(tool_name)
        if handler is None:
            safe_state["last_tool"] = "unknown"
            return (
                {
                    "ok": False,
                    "tool_name": tool_name,
                    "error": f"Unknown tool: {tool_name}",
                },
                safe_state,
            )

        try:
            result = handler(arguments or {}, phone, safe_state)
            safe_state["last_tool"] = tool_name
            return {"ok": True, "tool_name": tool_name, "result": result}, safe_state
        except Exception as exc:
            safe_state["last_tool"] = f"{tool_name}_error"
            return (
                {
                    "ok": False,
                    "tool_name": tool_name,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                safe_state,
            )

    def _count_mobile_services(
        self,
        _arguments: dict[str, Any],
        _phone: str,
        _state: dict[str, Any],
    ) -> dict[str, Any]:
        mobile_plan_count = len(sample_data.PLANS)
        sim_offer_count = len(sample_data.SIM_OFFERS)
        return {
            "service_groups": [
                {"group": "mobile_plans", "count": mobile_plan_count},
                {"group": "sim_offers", "count": sim_offer_count},
            ],
            "total_products": mobile_plan_count + sim_offer_count,
        }

    def _list_mobile_service_groups(
        self,
        _arguments: dict[str, Any],
        _phone: str,
        _state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "groups": [
                {
                    "group": "mobile_plans",
                    "label_vi": "Gói cước di động",
                    "description_vi": "Gói data/gọi điện theo tháng.",
                },
                {
                    "group": "sim_offers",
                    "label_vi": "Ưu đãi SIM mới",
                    "description_vi": "SIM mới kèm quà tặng data/gọi nội mạng.",
                },
            ]
        }

    def _list_mobile_plans(
        self,
        arguments: dict[str, Any],
        _phone: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        budget = _to_int(arguments.get("budget_max_vnd"))
        needs_5g = _to_bool(arguments.get("needs_5g"))
        focus = str(arguments.get("focus", "price")).strip().lower()
        if focus not in {"data", "call", "price"}:
            focus = "price"
        limit = max(1, min(_to_int(arguments.get("limit"), default=5) or 5, 10))

        plans = list_plans(
            budget_max_vnd=budget,
            needs_5g=needs_5g if needs_5g is not None else None,
        )

        if focus == "call":
            plans = sorted(
                plans,
                key=lambda p: (p["call_minutes_onnet"], -p["monthly_price_vnd"]),
                reverse=True,
            )
        elif focus == "data":
            plans = sorted(
                plans,
                key=lambda p: (p["data_gb_per_day"], -p["monthly_price_vnd"]),
                reverse=True,
            )
        else:
            plans = sorted(plans, key=lambda p: p["monthly_price_vnd"])

        short_list = [
            {
                "id": plan["id"],
                "name": plan["name"],
                "monthly_price_vnd": plan["monthly_price_vnd"],
                "data_gb_per_day": plan["data_gb_per_day"],
                "call_minutes_onnet": plan["call_minutes_onnet"],
                "is_5g": plan["is_5g"],
            }
            for plan in plans[:limit]
        ]
        state["last_plan_ids"] = [plan["id"] for plan in short_list]
        return {
            "filters": {
                "budget_max_vnd": budget,
                "needs_5g": needs_5g,
                "focus": focus,
                "limit": limit,
            },
            "total": len(plans),
            "plans": short_list,
        }

    def _recommend_mobile_plans(
        self,
        arguments: dict[str, Any],
        phone: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        budget = _to_int(arguments.get("budget_max_vnd"), default=250000) or 250000
        priority = str(arguments.get("priority", "data")).strip().lower()
        if priority not in {"data", "call", "price"}:
            priority = "data"

        req = RecommendPlanRequest(
            phone=str(arguments.get("phone", "")).strip() or phone,
            budget_max_vnd=max(50000, min(budget, 1_000_000)),
            priority=priority,  # type: ignore[arg-type]
            segment=str(arguments.get("segment", "")).strip() or None,
            needs_5g=_to_bool(arguments.get("needs_5g"), default=False) or False,
        )
        recs = recommend_plans(req)

        payload: list[dict[str, Any]] = []
        for rec in recs:
            plan = get_plan(rec.plan_id)
            if not plan:
                continue
            payload.append(
                {
                    "plan_id": rec.plan_id,
                    "score": rec.score,
                    "reason": rec.reason,
                    "plan": {
                        "name": plan["name"],
                        "monthly_price_vnd": plan["monthly_price_vnd"],
                        "data_gb_per_day": plan["data_gb_per_day"],
                        "call_minutes_onnet": plan["call_minutes_onnet"],
                        "is_5g": plan["is_5g"],
                    },
                }
            )

        state["last_plan_ids"] = [item["plan_id"] for item in payload]
        return {
            "request": {
                "phone": req.phone,
                "budget_max_vnd": req.budget_max_vnd,
                "priority": req.priority,
                "segment": req.segment,
                "needs_5g": req.needs_5g,
            },
            "recommendations": payload,
        }

    def _list_sim_offers(
        self,
        arguments: dict[str, Any],
        _phone: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        limit = max(1, min(_to_int(arguments.get("limit"), default=3) or 3, 10))
        offers = sample_data.SIM_OFFERS[:limit]
        state["last_offer_ids"] = [offer["id"] for offer in offers]
        return {
            "total": len(sample_data.SIM_OFFERS),
            "offers": [
                {
                    "id": offer["id"],
                    "name": offer["name"],
                    "sim_price_vnd": offer["sim_price_vnd"],
                    "bundle": offer["bundle"],
                }
                for offer in offers
            ],
        }

    def _lookup_subscriber(
        self,
        arguments: dict[str, Any],
        phone: str,
        _state: dict[str, Any],
    ) -> dict[str, Any]:
        target_phone = str(arguments.get("phone", "")).strip() or phone
        subscriber = get_subscriber(target_phone)
        if not subscriber:
            return {"found": False, "phone": target_phone}

        plan = get_plan(subscriber["current_plan_id"])
        return {
            "found": True,
            "subscriber": {
                "phone": subscriber["phone"],
                "name": subscriber["name"],
                "avg_data_usage_gb_per_day": subscriber["avg_data_usage_gb_per_day"],
                "avg_call_minutes_per_month": subscriber["avg_call_minutes_per_month"],
            },
            "current_plan": plan,
        }

    def _create_lead(
        self,
        arguments: dict[str, Any],
        phone: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        offer_id = str(arguments.get("interested_offer_id", "")).strip()
        offer_index = _to_int(arguments.get("offer_index"))
        customer_name = str(arguments.get("customer_name", "")).strip() or "Khach hang demo"
        note = str(arguments.get("note", "")).strip() or None
        target_phone = str(arguments.get("phone", "")).strip() or phone

        if not offer_id and offer_index is not None:
            last_offer_ids = state.get("last_offer_ids", [])
            idx = max(0, offer_index - 1)
            if 0 <= idx < len(last_offer_ids):
                offer_id = str(last_offer_ids[idx])

        if not offer_id:
            offer_id = sample_data.SIM_OFFERS[0]["id"]

        lead = create_lead(
            LeadRequest(
                customer_name=customer_name,
                phone=target_phone or "0987999999",
                interested_offer_id=offer_id,
                note=note,
            )
        )
        offer = next((item for item in sample_data.SIM_OFFERS if item["id"] == offer_id), None)
        return {
            "lead_id": lead.lead_id,
            "status": lead.status,
            "created_at": lead.created_at.isoformat(),
            "offer": offer,
        }

    def _create_callback(
        self,
        arguments: dict[str, Any],
        phone: str,
        _state: dict[str, Any],
    ) -> dict[str, Any]:
        target_phone = str(arguments.get("phone", "")).strip() or phone or "0987999999"
        reason = str(arguments.get("reason", "")).strip() or "Khach yeu cau goi lai."
        hours = max(0, min(_to_int(arguments.get("hours_from_now"), default=1) or 1, 24))

        callback = create_callback(
            CallbackRequest(
                phone=target_phone,
                requested_time=datetime.now(timezone.utc) + timedelta(hours=hours),
                reason=reason,
            )
        )
        return {
            "callback_id": callback.callback_id,
            "status": callback.status,
            "created_at": callback.created_at.isoformat(),
            "scheduled_after_hours": hours,
        }
