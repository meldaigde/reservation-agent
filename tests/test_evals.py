"""The scoring code decides what the eval numbers mean, so it gets its own tests."""

import importlib.util
import json
from pathlib import Path

EVALS = Path(__file__).parent.parent / "evals"
spec = importlib.util.spec_from_file_location("run_evals", EVALS / "run_evals.py")
run_evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_evals)

EXPECTED = {"name": "Ada Lovelace", "phone": "+442079460958", "date": "2026-11-18", "time": "19:30", "party_size": 4}


def test_exact_match_passes_and_phone_format_is_ignored() -> None:
    stored = {**EXPECTED, "name": "ada lovelace", "phone": "+44 20 7946 0958"}
    assert run_evals.score_case(EXPECTED, [stored])["passed"] is True


def test_wrong_field_fails_and_is_named() -> None:
    result = run_evals.score_case(EXPECTED, [{**EXPECTED, "time": "20:30"}])
    assert result["passed"] is False
    assert [field for field, ok in result["fields"].items() if not ok] == ["time"]


def test_missing_or_duplicate_booking_fails() -> None:
    assert run_evals.score_case(EXPECTED, [])["passed"] is False
    assert run_evals.score_case(EXPECTED, [EXPECTED, EXPECTED])["passed"] is False


def test_no_booking_expected() -> None:
    assert run_evals.score_case(None, [])["passed"] is True
    assert run_evals.score_case(None, [EXPECTED])["passed"] is False


def test_dataset_is_well_formed() -> None:
    lines = (EVALS / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    assert len({case["id"] for case in cases}) == len(cases) >= 15
    for case in cases:
        assert case["turns"] and all(isinstance(turn, str) for turn in case["turns"])
        assert case["expect"] is None or set(case["expect"]) == set(run_evals.FIELDS)
