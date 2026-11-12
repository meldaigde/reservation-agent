import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from reservation_agent.agent import FALLBACK_REPLY, MAX_STEPS, Agent
from reservation_agent.api import create_app
from reservation_agent.config import Settings
from reservation_agent.llm import LLMError, OpenRouterClient, ScriptedLLM
from reservation_agent.store import Store
from reservation_agent.tools import ReservationTools, normalize_phone

NOW = datetime(2026, 11, 16, 10, 0)  # a Monday


def clock() -> datetime:
    return NOW


def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


def say(text: str) -> dict:
    return {"role": "assistant", "content": text}


BOOKING = {"name": "Ada Lovelace", "phone": "+44 20 7946 0958", "date": "2026-11-17", "time": "19:30", "party_size": 4}


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


@pytest.fixture
def tools(store: Store) -> ReservationTools:
    return ReservationTools(store, Settings(), clock)


def make_agent(store: Store, responses: list) -> tuple[Agent, ScriptedLLM]:
    llm = ScriptedLLM(responses)
    return Agent(llm, store, Settings(), clock), llm


# Tools


def test_create_reservation_saves_normalized_row(tools: ReservationTools, store: Store) -> None:
    result = tools.create_reservation(session_id="s1", **BOOKING)
    assert result["ok"] and result["reservation_id"] == 1
    row = store.list_reservations()[0]
    assert (row["name"], row["phone"], row["date"], row["time"], row["party_size"]) == (
        "Ada Lovelace",
        "+442079460958",
        "2026-11-17",
        "19:30",
        4,
    )


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"date": "17/11/2026"}, "YYYY-MM-DD"),
        ({"time": "7pm"}, "HH:MM"),
        ({"party_size": 0}, "at least 1"),
        ({"party_size": 20}, "larger than 12"),
        ({"date": "2026-11-15"}, "past"),
        ({"date": "2026-11-16", "time": "09:30"}, "past"),
        ({"time": "03:00"}, "opening hours"),
        ({"time": "23:00"}, "opening hours"),
        ({"phone": "123"}, "phone"),
        ({"name": " "}, "name"),
    ],
)
def test_create_reservation_rejects_bad_input(tools: ReservationTools, store: Store, override: dict, expected: str) -> None:
    result = tools.create_reservation(session_id="s1", **{**BOOKING, **override})
    assert result["ok"] is False
    assert expected in result["error"]
    assert store.list_reservations() == []


def test_capacity_is_shared_by_overlapping_slots(tools: ReservationTools) -> None:
    for index in range(3):
        booked = tools.create_reservation(session_id="s", **{**BOOKING, "party_size": 10, "name": f"Guest {index}"})
        assert booked["ok"]

    full = tools.check_availability("2026-11-17", "20:00", 2)
    assert full == {"ok": True, "available": False, "seats_free": 0, "date": "2026-11-17", "time": "20:00"}
    assert tools.create_reservation(session_id="s", **{**BOOKING, "time": "20:00"})["ok"] is False

    later = tools.check_availability("2026-11-17", "21:30", 2)
    assert later["available"] is True and later["seats_free"] == 30


def test_unknown_tool_and_extra_arguments(tools: ReservationTools) -> None:
    assert tools.call("delete_everything", {}, "s")["ok"] is False
    result = tools.call("check_availability", {"date": "2026-11-17", "time": "19:30", "party_size": 2, "vip": True}, "s")
    assert result["available"] is True


def test_normalize_phone() -> None:
    assert normalize_phone("+90 (546) 000-00 00") == "+905460000000"
    assert normalize_phone("0546 000 00 00") == "05460000000"


# Agent loop


def test_agent_books_through_tool_call(store: Store) -> None:
    agent, llm = make_agent(store, [tool_call("create_reservation", BOOKING), say("Booked for Ada, 4 guests.")])

    result = agent.run_turn("s1", "Table for 4 tomorrow 19:30, Ada Lovelace, +44 20 7946 0958")

    assert result.reply == "Booked for Ada, 4 guests."
    assert result.reservation_id == 1 and result.steps == 2
    # The second model call must see the tool result.
    tool_message = llm.calls[1][-1]
    assert tool_message["role"] == "tool" and json.loads(tool_message["content"])["ok"] is True
    kinds = [(trace["kind"], trace["name"], trace["ok"]) for trace in store.traces("s1")]
    assert kinds == [("llm", "scripted", True), ("tool", "create_reservation", True), ("llm", "scripted", True)]


def test_agent_keeps_memory_between_turns(store: Store) -> None:
    agent, llm = make_agent(store, [say("For how many guests?"), say("And your phone number?")])

    agent.run_turn("s1", "I'd like a table tomorrow at 8pm")
    second = agent.run_turn("s1", "Four of us")

    roles = [message["role"] for message in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert second.turn == 2
    assert "2026-11-16" in llm.calls[0][0]["content"]  # the prompt carries today's date


def test_tool_error_is_returned_to_model_not_raised(store: Store) -> None:
    agent, llm = make_agent(
        store,
        [tool_call("create_reservation", {**BOOKING, "time": "03:00"}), say("We open at 09:00. Another time?")],
    )

    result = agent.run_turn("s1", "3am please")

    assert result.reservation_id is None
    assert "opening hours" in json.loads(llm.calls[1][-1]["content"])["error"]
    assert store.traces("s1")[1]["ok"] is False


def test_malformed_tool_arguments_do_not_crash(store: Store) -> None:
    broken = tool_call("create_reservation", {})
    broken["tool_calls"][0]["function"]["arguments"] = "{not json"
    agent, llm = make_agent(store, [broken, say("Could you repeat that?")])

    result = agent.run_turn("s1", "book")

    assert result.reply == "Could you repeat that?"
    assert "Invalid tool arguments" in json.loads(llm.calls[1][-1]["content"])["error"]


def test_llm_failure_gives_fallback_reply_and_error_trace(store: Store) -> None:
    agent, _ = make_agent(store, [LLMError("boom")])

    result = agent.run_turn("s1", "hello")

    assert result.reply == FALLBACK_REPLY
    trace = store.traces("s1")[0]
    assert trace["ok"] is False and trace["detail"]["error"] == "boom"


def test_agent_stops_after_max_steps(store: Store) -> None:
    looping = [tool_call("check_availability", {"date": "2026-11-17", "time": "19:30", "party_size": 2})] * (MAX_STEPS + 2)
    agent, llm = make_agent(store, looping)

    result = agent.run_turn("s1", "loop")

    assert result.reply == FALLBACK_REPLY and len(llm.calls) == MAX_STEPS


# LLM client


class FakeHTTP:
    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.models: list[str] = []

    def post(self, url: str, headers: dict, json: dict):
        self.models.append(json["model"])
        return self.replies.pop(0)


class FakeReply:
    def __init__(self, status_code: int, body: dict, headers: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)
        self.headers = headers or {}

    def json(self) -> dict:
        return self._body


def test_client_falls_back_to_second_model() -> None:
    ok_body = {"model": "backup", "choices": [{"message": say("hi")}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
    http = FakeHTTP([FakeReply(500, {"error": "down"}), FakeReply(200, ok_body)])
    client = OpenRouterClient("key", "primary", "backup", http=http, max_retries=0)

    response = client.complete([{"role": "user", "content": "hi"}], [])

    assert http.models == ["primary", "backup"]
    assert response.model == "backup" and response.prompt_tokens == 5
    assert response.attempts and response.attempts[0].startswith("primary: HTTP 500")


def test_client_retries_rate_limit_on_same_model() -> None:
    ok_body = {"choices": [{"message": say("hi")}]}
    http = FakeHTTP(
        [FakeReply(429, {}, {"retry-after": "7"}), FakeReply(503, {}), FakeReply(200, ok_body)]
    )
    waits: list[float] = []
    client = OpenRouterClient("key", "primary", http=http, sleep=waits.append)

    response = client.complete([{"role": "user", "content": "hi"}], [])

    assert http.models == ["primary", "primary", "primary"]
    assert waits == [7.0, 4.0]  # Retry-After first, then exponential backoff
    assert response.message["content"] == "hi" and response.attempts == []


def test_client_gives_up_after_max_retries() -> None:
    http = FakeHTTP([FakeReply(429, {})] * 3)
    client = OpenRouterClient("key", "primary", http=http, max_retries=2, sleep=lambda _: None)
    with pytest.raises(LLMError, match="HTTP 429"):
        client.complete([], [])
    assert len(http.models) == 3


def test_client_does_not_retry_client_errors() -> None:
    http = FakeHTTP([FakeReply(401, {"error": "bad key"})])
    client = OpenRouterClient("key", "primary", http=http, sleep=lambda _: None)
    with pytest.raises(LLMError, match="HTTP 401"):
        client.complete([], [])
    assert len(http.models) == 1


def test_client_raises_when_every_model_fails() -> None:
    http = FakeHTTP([FakeReply(500, {}), FakeReply(200, {"unexpected": True})])
    client = OpenRouterClient("key", "primary", "backup", http=http, max_retries=0)
    with pytest.raises(LLMError, match="All models failed"):
        client.complete([], [])


def test_client_requires_key_and_model() -> None:
    with pytest.raises(LLMError):
        OpenRouterClient("", "model")
    with pytest.raises(LLMError):
        OpenRouterClient("key", "")


# API


def test_chat_endpoint_round_trip(store: Store) -> None:
    agent, _ = make_agent(store, [tool_call("create_reservation", BOOKING), say("Booked.")])
    client = TestClient(create_app(agent))

    reply = client.post("/chat", json={"message": "book it"})

    body = reply.json()
    assert reply.status_code == 200 and body["reply"] == "Booked." and body["reservation_id"] == 1
    assert len(client.get("/reservations", params={"date": "2026-11-17"}).json()) == 1
    assert len(client.get(f"/traces/{body['session_id']}").json()) == 3
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_endpoint_reports_missing_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/chat", json={"message": "hi"}).status_code == 503
