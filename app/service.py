"""Business logic for telecom demo APIs."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app import sample_data
from app.schemas import (
    CallbackRequest,
    CallbackResponse,
    LeadRequest,
    LeadResponse,
    RecommendPlanRequest,
    RecommendPlanResult,
)


def list_plans(
    segment: str | None = None,
    budget_max_vnd: int | None = None,
    needs_5g: bool | None = None,
) -> list[dict]:
    plans = sample_data.PLANS

    if segment:
        plans = [p for p in plans if segment in p["segments"]]
    if budget_max_vnd is not None:
        plans = [p for p in plans if p["monthly_price_vnd"] <= budget_max_vnd]
    if needs_5g is not None:
        plans = [p for p in plans if p["is_5g"] is needs_5g]

    return plans


def get_plan(plan_id: str) -> dict | None:
    return next((plan for plan in sample_data.PLANS if plan["id"] == plan_id), None)


def get_subscriber(phone: str) -> dict | None:
    return sample_data.SUBSCRIBERS.get(phone)


def recommend_plans(request: RecommendPlanRequest) -> list[RecommendPlanResult]:
    candidates = list_plans(
        segment=request.segment,
        budget_max_vnd=request.budget_max_vnd,
        needs_5g=request.needs_5g if request.needs_5g else None,
    )
    if not candidates:
        return []

    subscriber = get_subscriber(request.phone) if request.phone else None
    scored = []

    for plan in candidates:
        score = 0.0
        reasons: list[str] = []

        if request.priority == "data":
            score += plan["data_gb_per_day"] * 2
            reasons.append(f"{plan['data_gb_per_day']}GB/ngay")
        elif request.priority == "call":
            score += plan["call_minutes_onnet"] / 100
            reasons.append(f"{plan['call_minutes_onnet']} phut noi mang")
        else:
            score += (request.budget_max_vnd - plan["monthly_price_vnd"]) / 100000
            reasons.append(f"Gia {plan['monthly_price_vnd']:,} VND/thang")

        if subscriber:
            if plan["data_gb_per_day"] >= subscriber["avg_data_usage_gb_per_day"]:
                score += 2
                reasons.append("Phu hop muc dung data hien tai")
            if (
                plan["call_minutes_onnet"]
                >= subscriber["avg_call_minutes_per_month"] * 0.5
            ):
                score += 1
                reasons.append("Du phut goi noi mang")

        if plan["is_5g"]:
            score += 0.5

        scored.append(
            RecommendPlanResult(
                plan_id=plan["id"],
                score=round(score, 2),
                reason=", ".join(reasons),
            )
        )

    return sorted(scored, key=lambda item: item.score, reverse=True)[:3]


def create_lead(payload: LeadRequest) -> LeadResponse:
    if not any(offer["id"] == payload.interested_offer_id for offer in sample_data.SIM_OFFERS):
        raise ValueError("interested_offer_id is not valid")

    lead = {
        "lead_id": f"lead_{uuid4().hex[:8]}",
        "customer_name": payload.customer_name,
        "phone": payload.phone,
        "interested_offer_id": payload.interested_offer_id,
        "note": payload.note,
        "created_at": datetime.now(timezone.utc),
        "status": "new",
    }
    sample_data.LEADS.append(lead)
    return LeadResponse(
        lead_id=lead["lead_id"],
        created_at=lead["created_at"],
        status=lead["status"],
    )


def create_callback(payload: CallbackRequest) -> CallbackResponse:
    callback = {
        "callback_id": f"cb_{uuid4().hex[:8]}",
        "phone": payload.phone,
        "requested_time": payload.requested_time,
        "reason": payload.reason,
        "created_at": datetime.now(timezone.utc),
        "status": "scheduled",
    }
    sample_data.CALLBACKS.append(callback)
    return CallbackResponse(
        callback_id=callback["callback_id"],
        created_at=callback["created_at"],
        status=callback["status"],
    )
