"""FastAPI entrypoint for VNPost telecom callbot demo."""

from fastapi import FastAPI, HTTPException

from app import sample_data
from app.schemas import (
    CallbackRequest,
    CallbackResponse,
    LeadRequest,
    LeadResponse,
    Plan,
    RecommendPlanRequest,
    RecommendPlanResponse,
    SimOffer,
    Subscriber,
)
from app.service import (
    create_callback,
    create_lead,
    get_plan,
    get_subscriber,
    list_plans,
    recommend_plans,
)

app = FastAPI(
    title="VNPost Telecom Callbot Demo",
    description="Demo API for SIM sales and telecom package lookup.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "vnpost-telecom-callbot-demo"}


@app.get("/telecom/plans", response_model=list[Plan])
def telecom_plans(
    segment: str | None = None,
    budget_max_vnd: int | None = None,
    needs_5g: bool | None = None,
) -> list[Plan]:
    return [Plan(**plan) for plan in list_plans(segment, budget_max_vnd, needs_5g)]


@app.get("/telecom/plans/{plan_id}", response_model=Plan)
def telecom_plan_details(plan_id: str) -> Plan:
    plan = get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return Plan(**plan)


@app.get("/telecom/sim-offers", response_model=list[SimOffer])
def telecom_sim_offers() -> list[SimOffer]:
    return [SimOffer(**offer) for offer in sample_data.SIM_OFFERS]


@app.get("/telecom/subscribers/{phone}", response_model=Subscriber)
def telecom_subscriber(phone: str) -> Subscriber:
    subscriber = get_subscriber(phone)
    if not subscriber:
        raise HTTPException(status_code=404, detail="Subscriber not found")
    return Subscriber(**subscriber)


@app.post("/telecom/recommend-plan", response_model=RecommendPlanResponse)
def telecom_recommend_plan(payload: RecommendPlanRequest) -> RecommendPlanResponse:
    return RecommendPlanResponse(
        request=payload,
        recommendations=recommend_plans(payload),
    )


@app.post("/telecom/leads", response_model=LeadResponse)
def telecom_create_lead(payload: LeadRequest) -> LeadResponse:
    try:
        return create_lead(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/telecom/callbacks", response_model=CallbackResponse)
def telecom_create_callback(payload: CallbackRequest) -> CallbackResponse:
    return create_callback(payload)
