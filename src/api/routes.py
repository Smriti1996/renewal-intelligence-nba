# src/api/routes.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.schemas import ChatRequest, ChatResponse
from src.llm.router import answer_query
from src.common.logging import setup_logger

router = APIRouter()
logger = setup_logger("api")


# ---------- Health / readiness ----------

class HealthResponse(BaseModel):
    status: str = "ok"


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
def readiness_check() -> HealthResponse:
    # In future you can add checks for KG/index files etc.
    return HealthResponse(status="ok")


# ---------- Chat endpoint ----------

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    if not payload.user_query.strip():
        raise HTTPException(status_code=400, detail="user_query cannot be empty")

    logger.info("Incoming chat request, member=%s", payload.membership_nbr)

    result = answer_query(
        user_query=payload.user_query,
        membership_nbr=payload.membership_nbr,
    )

    return ChatResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", "unknown"),
        membership_nbr=result.get("used_member_nbr"),
    )
