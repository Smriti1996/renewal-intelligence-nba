# src/api/schemas.py
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_query: str = Field(..., description="User's natural language question")
    membership_nbr: Optional[int] = Field(
        None,
        description="Optional membership number to personalize NBA",
    )


class ChatResponse(BaseModel):
    answer: str
    intent: str
    membership_nbr: Optional[int] = None
