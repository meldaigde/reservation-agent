# Reservation Agent

[![tests](https://github.com/meldaigde/reservation-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/meldaigde/reservation-agent/actions/workflows/tests.yml)

A small LLM agent that takes table reservations in conversation. The guest writes in any language, the agent asks for whatever is missing, checks availability and books the table through tool calls. Every model call and tool call is traced, and an eval set scores what actually ends up in the database.

I build reservation and ordering assistants for local businesses. This repository is a clean, generic version of that idea, written to show how I structure an agent: business rules in code, the model on a short leash, and numbers to tell me when a prompt or model change made things worse. It contains no client code or data.

## How it works

```
POST /chat ──> Agent.run_turn
                 │  system prompt (today's date, opening hours) + stored history + new message
                 ▼
               LLM (OpenRouter, any tool-calling model, optional fallback model)
                 │
        tool call? ── yes ──> check_availability / create_reservation ──> SQLite
                 │                     │
                 │                     └── result (or a readable error) goes back to the model
                 no
                 ▼
               reply to the guest
```

| File | What it does |
|---|---|
| `src/reservation_agent/agent.py` | The loop: call the model, run tools, feed results back, stop after 6 steps |
| `src/reservation_agent/tools.py` | Tool schemas and the business rules (opening hours, party size, capacity, past dates) |
| `src/reservation_agent/llm.py` | Client for any OpenAI-compatible endpoint, with retries and model fallback, and a scripted client for tests |
| `src/reservation_agent/store.py` | SQLite: reservations, conversation memory, traces |
| `src/reservation_agent/api.py` | FastAPI endpoints |
| `evals/` | 15 conversations with expected outcomes, and the scoring script |
| `tests/` | 33 tests that run without a network call |

## Design decisions

- **Rules live in the tools, not the prompt.** The prompt says what the opening hours are, but the tool is what refuses a 03:00 booking. A model that ignores its instructions still cannot write a bad row.
- **Tool errors are answers, not exceptions.** A wrong date format or a full time slot comes back to the model as `{"ok": false, "error": "..."}`, so it can ask the guest again. Malformed JSON arguments and unknown tool names are handled the same way.
- **The agent may not claim a booking it did not make.** The prompt forbids confirming unless `create_reservation` returned `ok: true`, and the evals check the database, not the reply text.
- **Capacity is counted across overlapping slots.** Two reservations overlap when they start less than two hours apart, and their guests share one pool of seats.
- **Memory is the stored message list.** Each session's messages, including tool calls and results, are replayed on the next turn.
- **A step limit and a fallback reply.** If the model keeps calling tools or every model fails, the guest gets a plain apology and the trace records why.

## Run it

Requires Python 3.11 or newer and an API key for [OpenRouter](https://openrouter.ai) or any other OpenAI-compatible provider. For another provider, set `LLM_BASE_URL` (for example `https://api.groq.com/openai/v1`) and put that provider's key in `OPENROUTER_API_KEY`.

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # then set OPENROUTER_API_KEY and MODEL
set -a && source .env && set +a   # PowerShell: set the variables with $env:NAME = "value"
uvicorn reservation_agent.api:app --reload
```

```bash
curl -X POST localhost:8000/chat -H "content-type: application/json" \
  -d '{"message": "Table for 4 tomorrow at 19:30, Ada Lovelace, +44 20 7946 0958"}'
```

The response includes a `session_id`. Send it back with the next message to continue the same conversation.

| Endpoint | Purpose |
|---|---|
| `POST /chat` | One guest message in, one reply out |
| `GET /reservations?date=YYYY-MM-DD` | Stored reservations |
| `GET /traces/{session_id}` | Every model and tool call of a conversation |
| `GET /health` | Liveness check |

## Observability

Each model call and tool call writes one row to the `traces` table: session, turn, step, kind (`llm` or `tool`), name, success, latency, token counts and a JSON detail field (tool arguments and result, or the error, or which models failed before the fallback answered).

```bash
curl localhost:8000/traces/<session_id>
```

This is enough to answer the usual questions: why did the agent say that, which tool rejected the request, how long did the model take, did the fallback fire.

## Evals

`evals/dataset.jsonl` holds 15 conversations: complete requests, multi-turn collection, Turkish and German, relative dates ("tomorrow", "this Friday"), 12-hour times, a mid-conversation correction, and six cases where nothing should be booked (closed hours, oversized party, past date, missing phone, off-topic, question only).

```bash
python evals/run_evals.py --min-pass 0.9
```

The script replays each conversation against a fresh database with the clock fixed to Monday 2026-11-16, then compares the stored reservation with the expected one. It prints pass or fail per case, per-field accuracy (name, phone, date, time, party size), average model calls, latency and token use, and writes `evals/results.json`.

### What the evals caught

First run, 12 Nov 2026, `openai/gpt-oss-120b` on Groq's free tier: **11 of 15 passed**. The failures were useful:

| Finding | Fix |
|---|---|
| With every detail in hand, the model checked availability and then asked "shall I go ahead?" instead of booking | One line in the prompt: call `create_reservation` as soon as all five details are known |
| Four cases hit the provider's rate limit and got the fallback apology | The client now retries 429 and 5xx responses, honoring `Retry-After` |
| Two of those outages were scored as passes, because "nothing booked" was the expected outcome | A case with a model error now always fails |

Second run, same model: **15 of 15**, about 2.5 model calls per conversation.

Read that number with care. It is one run of a small set against one model, and I changed the prompt after seeing the failures, so the set is no longer unseen. It tells me the agent handles these 15 situations; it is not a general accuracy figure. Results are not committed, because they depend on the model you choose. Run the set against two models and compare before switching.

## Tests

```bash
pytest
```

The tests cover the tool rules, capacity, the agent loop (tool call, memory, tool errors, malformed arguments, model failure, step limit), the fallback client, the eval scoring and the API. They use a scripted model, so they need no API key and run in CI.

## Limitations

- One venue, one seat pool. No table map, no cancellation or change of an existing booking.
- SQLite with a single connection: fine for a demo, not for concurrent production traffic.
- No authentication on the API. Put it behind your own backend before exposing it.
- The eval set is small and checks outcomes, not tone.

## A note on how this was built

I wrote this with an AI coding assistant (Claude), which is how I work day to day.

## License

MIT
