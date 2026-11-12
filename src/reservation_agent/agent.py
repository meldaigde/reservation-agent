"""The agent loop: model call, tool execution, repeat until the model answers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .config import Settings
from .llm import LLMClient, LLMError
from .store import Store
from .tools import TOOL_SCHEMAS, ReservationTools

MAX_STEPS = 6
FALLBACK_REPLY = "Sorry, I could not complete that just now. Please try again in a moment."


def build_system_prompt(settings: Settings, now: datetime) -> str:
    return (
        f"You are the reservation assistant for {settings.business_name}.\n"
        f"Now: {now.strftime('%A %Y-%m-%d %H:%M')}. Resolve relative dates such as 'tomorrow' from this.\n"
        f"Opening hours: {settings.opening_time} to {settings.closing_time}."
        f" Largest party you can book: {settings.max_party_size}.\n"
        "To book you need: full name, phone number, date, time and party size."
        " Ask only for what is still missing, one short question at a time.\n"
        "Never invent a detail the guest did not give. Never say a table is booked unless"
        " create_reservation returned ok=true.\n"
        "When a tool returns an error, explain it plainly and offer a way forward.\n"
        "After booking, confirm name, date, time and party size in one sentence.\n"
        "Reply in the guest's language. Decline anything unrelated to reservations."
    )


@dataclass
class TurnResult:
    reply: str
    reservation_id: int | None
    steps: int
    turn: int


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        store: Store,
        settings: Settings,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.llm = llm
        self.store = store
        self.settings = settings
        self.clock = clock
        self.tools = ReservationTools(store, settings, clock)

    def run_turn(self, session_id: str, user_text: str) -> TurnResult:
        turn = self.store.turn_count(session_id) + 1
        user_message = {"role": "user", "content": user_text}
        messages = [
            {"role": "system", "content": build_system_prompt(self.settings, self.clock())},
            *self.store.history(session_id),
            user_message,
        ]
        self.store.add_message(session_id, user_message)
        reservation_id: int | None = None

        for step in range(1, MAX_STEPS + 1):
            started = time.perf_counter()
            try:
                response = self.llm.complete(messages, TOOL_SCHEMAS)
            except LLMError as exc:
                self._trace(session_id, turn, step, "llm", "error", False, started, detail={"error": str(exc)})
                return self._finish(session_id, FALLBACK_REPLY, reservation_id, step, turn)

            message = _clean(response.message)
            tool_calls = message.get("tool_calls") or []
            self._trace(
                session_id,
                turn,
                step,
                "llm",
                response.model,
                True,
                started,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                detail={"tool_calls": len(tool_calls), "failed_attempts": response.attempts},
            )
            messages.append(message)
            self.store.add_message(session_id, message)

            if not tool_calls:
                return TurnResult(message.get("content") or "", reservation_id, step, turn)

            for call in tool_calls:
                result = self._run_tool(session_id, turn, step, call)
                if result.get("reservation_id"):
                    reservation_id = result["reservation_id"]
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": json.dumps(result, ensure_ascii=False),
                }
                messages.append(tool_message)
                self.store.add_message(session_id, tool_message)

        return self._finish(session_id, FALLBACK_REPLY, reservation_id, MAX_STEPS, turn)

    def _run_tool(self, session_id: str, turn: int, step: int, call: dict[str, Any]) -> dict[str, Any]:
        function = call.get("function") or {}
        name = function.get("name", "")
        started = time.perf_counter()
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            result: dict[str, Any] = {"ok": False, "error": f"Invalid tool arguments: {exc}"}
            arguments = {}
        else:
            result = self.tools.call(name, arguments, session_id)
        self._trace(
            session_id,
            turn,
            step,
            "tool",
            name or "unknown",
            bool(result.get("ok")),
            started,
            detail={"arguments": arguments, "result": result},
        )
        return result

    def _finish(self, session_id: str, reply: str, reservation_id: int | None, steps: int, turn: int) -> TurnResult:
        self.store.add_message(session_id, {"role": "assistant", "content": reply})
        return TurnResult(reply, reservation_id, steps, turn)

    def _trace(
        self,
        session_id: str,
        turn: int,
        step: int,
        kind: str,
        name: str,
        ok: bool,
        started: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_trace(
            session_id=session_id,
            turn=turn,
            step=step,
            kind=kind,
            name=name,
            ok=ok,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            detail=detail,
            created_at=self.clock().isoformat(timespec="seconds"),
        )


def _clean(message: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields that are valid to send back to the API."""
    cleaned: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        cleaned["tool_calls"] = message["tool_calls"]
    return cleaned
