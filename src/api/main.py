# src/api/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as chat_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Renewal Intelligence Chat API",
        version="0.1.0",
    )

    # Basic CORS for local UI/dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router, prefix="/api")

    return app


app = create_app()
