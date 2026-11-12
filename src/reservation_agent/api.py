"""HTTP API. Run with: uvicorn reservation_agent.api:app --reload"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent
from .config import Settings
from .llm import OpenRouterClient
from .store import Store


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    reservation_id: int | None = None
    steps: int


def create_app(agent: Agent | None = None) -> FastAPI:
    app = FastAPI(title="Reservation Agent")
    state: dict[str, Agent | None] = {"agent": agent}

    def get_agent() -> Agent:
        if state["agent"] is None:
            settings = Settings.from_env()
            llm = OpenRouterClient(
                settings.api_key, settings.model, settings.fallback_model, settings.base_url
            )
            state["agent"] = Agent(llm, Store(settings.db_path), settings)
        return state["agent"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            current = get_agent()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        session_id = request.session_id or uuid.uuid4().hex
        result = current.run_turn(session_id, request.message)
        return ChatResponse(
            session_id=session_id,
            reply=result.reply,
            reservation_id=result.reservation_id,
            steps=result.steps,
        )

    @app.get("/reservations")
    def reservations(date: str | None = None) -> list[dict]:
        return get_agent().store.list_reservations(date)

    @app.get("/traces/{session_id}")
    def traces(session_id: str) -> list[dict]:
        return get_agent().store.traces(session_id)

    return app


app = create_app()
