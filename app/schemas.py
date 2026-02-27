"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    id: str
    name: str
    monthly_price_vnd: int
    data_gb_per_day: float
    call_minutes_onnet: int
    is_5g: bool
    segments: list[str]


class SimOffer(BaseModel):
    id: str
    name: str
    sim_price_vnd: int
    bundle: str
    target_segments: list[str]


class Subscriber(BaseModel):
    phone: str
    name: str
    current_plan_id: str
    avg_data_usage_gb_per_day: float
    avg_call_minutes_per_month: int


class RecommendPlanRequest(BaseModel):
    phone: str | None = Field(default=None, description="Subscriber phone number")
    budget_max_vnd: int = Field(default=200000, ge=50000, le=1000000)
    priority: Literal["data", "call", "price"] = "data"
    segment: str | None = None
    needs_5g: bool = False


class RecommendPlanResult(BaseModel):
    plan_id: str
    score: float
    reason: str


class RecommendPlanResponse(BaseModel):
    request: RecommendPlanRequest
    recommendations: list[RecommendPlanResult]


class LeadRequest(BaseModel):
    customer_name: str
    phone: str = Field(min_length=10, max_length=11)
    interested_offer_id: str
    note: str | None = None


class LeadResponse(BaseModel):
    lead_id: str
    created_at: datetime
    status: str


class CallbackRequest(BaseModel):
    phone: str = Field(min_length=10, max_length=11)
    requested_time: datetime
    reason: str


class CallbackResponse(BaseModel):
    callback_id: str
    created_at: datetime
    status: str
