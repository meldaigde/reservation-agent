"""Run the eval set against a real model and score what ended up in the database.

Usage:
    python evals/run_evals.py                 # uses MODEL from the environment
    python evals/run_evals.py --min-pass 0.9  # exit code 1 below 90% of cases passing

Each case replays the guest's turns against a fresh in-memory database with the
clock fixed to Monday 2026-11-16 10:00, then compares the stored reservation with
the expected one. A case with "expect": null passes only when nothing was booked.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from reservation_agent.agent import Agent
from reservation_agent.config import Settings
from reservation_agent.llm import OpenRouterClient
from reservation_agent.store import Store

FIXED_NOW = datetime(2026, 11, 16, 10, 0)
FIELDS = ("name", "phone", "date", "time", "party_size")
HERE = Path(__file__).parent


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value))


def field_matches(field: str, expected: Any, actual: Any) -> bool:
    if field == "phone":
        return _digits(expected) == _digits(actual)
    if field == "name":
        return str(expected).casefold().strip() == str(actual).casefold().strip()
    return str(expected) == str(actual)


def score_case(expected: dict[str, Any] | None, reservations: list[dict[str, Any]]) -> dict[str, Any]:
    if expected is None:
        return {"passed": not reservations, "fields": {}, "booked": len(reservations)}
    if len(reservations) != 1:
        return {"passed": False, "fields": {field: False for field in FIELDS}, "booked": len(reservations)}
    fields = {field: field_matches(field, expected[field], reservations[0][field]) for field in FIELDS}
    return {"passed": all(fields.values()), "fields": fields, "booked": 1}


def run_case(case: dict[str, Any], llm: OpenRouterClient, settings: Settings) -> dict[str, Any]:
    store = Store(":memory:")
    agent = Agent(llm, store, settings, clock=lambda: FIXED_NOW)
    replies = [agent.run_turn(case["id"], turn).reply for turn in case["turns"]]
    reservations = store.list_reservations()
    traces = store.traces(case["id"])
    llm_traces = [trace for trace in traces if trace["kind"] == "llm"]
    llm_errors = [trace["detail"].get("error", "") for trace in llm_traces if not trace["ok"]]
    score = score_case(case["expect"], reservations)
    if llm_errors:
        # A model outage books nothing, which would look like a pass for the
        # "nothing should be booked" cases. Count it as a failure instead.
        score["passed"] = False
    return {
        "id": case["id"],
        **score,
        "llm_errors": llm_errors,
        "llm_calls": len(llm_traces),
        "tool_calls": len(traces) - len(llm_traces),
        "latency_ms": round(sum(trace["latency_ms"] for trace in llm_traces)),
        "tokens": sum(trace["prompt_tokens"] + trace["completion_tokens"] for trace in llm_traces),
        "stored": reservations,
        "last_reply": replies[-1] if replies else "",
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    booking_cases = [result for result in results if result["fields"]]
    field_accuracy = {
        field: round(sum(result["fields"][field] for result in booking_cases) / len(booking_cases), 3)
        for field in FIELDS
    } if booking_cases else {}
    return {
        "cases": len(results),
        "passed": sum(result["passed"] for result in results),
        "pass_rate": round(sum(result["passed"] for result in results) / len(results), 3),
        "field_accuracy": field_accuracy,
        "avg_llm_calls": round(sum(result["llm_calls"] for result in results) / len(results), 2),
        "avg_latency_ms": round(sum(result["latency_ms"] for result in results) / len(results)),
        "total_tokens": sum(result["tokens"] for result in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(HERE / "dataset.jsonl"))
    parser.add_argument("--out", default=str(HERE / "results.json"))
    parser.add_argument("--min-pass", type=float, default=0.0, help="fail below this pass rate (0 to 1)")
    args = parser.parse_args()

    settings = Settings.from_env()
    llm = OpenRouterClient(settings.api_key, settings.model, settings.fallback_model, settings.base_url)
    lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]

    results = []
    for case in cases:
        result = run_case(case, llm, settings)
        results.append(result)
        wrong = [field for field, ok in result["fields"].items() if not ok]
        note = f"  wrong: {', '.join(wrong)}" if wrong else ""
        if result["llm_errors"]:
            note = f"  model error: {result['llm_errors'][0][:120]}"
        print(f"{'PASS' if result['passed'] else 'FAIL'}  {result['id']:<30} {result['llm_calls']} llm calls{note}")

    summary = summarize(results)
    print("\n" + json.dumps(summary, indent=2))
    Path(args.out).write_text(
        json.dumps({"model": settings.model, "summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0 if summary["pass_rate"] >= args.min_pass else 1


if __name__ == "__main__":
    sys.exit(main())
