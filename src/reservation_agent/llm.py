"""LLM clients.

OpenRouterClient talks to any OpenAI-compatible chat completions endpoint and
falls back to a second model when the first one fails. ScriptedLLM replays
canned responses so the agent loop can be tested without a network call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import httpx


class LLMError(RuntimeError):
    pass


def _retry_delay(reply: Any, attempt: int, max_wait: float) -> float:
    """Honor Retry-After when the server sends it, otherwise back off 2s, 4s, 8s."""
    try:
        wait = float(reply.headers.get("retry-after", ""))
    except (AttributeError, TypeError, ValueError):
        wait = 2.0 * (2**attempt)
    return max(0.0, min(wait, max_wait))


@dataclass
class LLMResponse:
    message: dict[str, Any]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_model: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 30.0,
        http: httpx.Client | None = None,
        max_retries: int = 3,
        max_retry_wait: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait
        self.sleep = sleep
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        if not model:
            raise LLMError("MODEL is not set")
        self.models = [model] + ([fallback_model] if fallback_model else [])
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        attempts: list[str] = []
        for model in self.models:
            try:
                response = self._post(model, messages, tools)
            except (httpx.HTTPError, LLMError) as exc:
                attempts.append(f"{model}: {exc}")
                continue
            response.attempts = attempts
            return response
        raise LLMError("All models failed. " + " | ".join(attempts))

    def _post(self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }
        url = f"{self.base_url}/chat/completions"
        reply = self.http.post(url, headers=self.headers, json=payload)
        # Rate limits and server errors are usually gone a few seconds later.
        for attempt in range(self.max_retries):
            if reply.status_code != 429 and reply.status_code < 500:
                break
            self.sleep(_retry_delay(reply, attempt, self.max_retry_wait))
            reply = self.http.post(url, headers=self.headers, json=payload)
        if reply.status_code >= 400:
            raise LLMError(f"HTTP {reply.status_code}: {reply.text[:200]}")
        data = reply.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            message=message,
            model=data.get("model", model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


class ScriptedLLM:
    """Returns the given assistant messages in order. Used by tests."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        if not self._responses:
            raise LLMError("ScriptedLLM has no responses left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(message=item, model="scripted")
