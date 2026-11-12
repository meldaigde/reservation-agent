"""Tools the model can call. Business rules live here, not in the prompt.

Every tool returns a dict with "ok". On failure it also returns "error" with a
message the model can read and act on, so a bad argument becomes a follow-up
question to the guest and not a crash.
"""

from __future__ import annotations

import re
from datetime import date as Date
from datetime import datetime
from typing import Any, Callable

from .config import Settings
from .store import Store

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check whether a table is free for a date, time and party size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date as YYYY-MM-DD"},
                    "time": {"type": "string", "description": "24-hour time as HH:MM"},
                    "party_size": {"type": "integer", "description": "Number of guests"},
                },
                "required": ["date", "time", "party_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reservation",
            "description": (
                "Create the reservation. Call only after the guest has given name, phone,"
                " date, time and party size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Guest's full name"},
                    "phone": {"type": "string", "description": "Guest's phone number"},
                    "date": {"type": "string", "description": "Date as YYYY-MM-DD"},
                    "time": {"type": "string", "description": "24-hour time as HH:MM"},
                    "party_size": {"type": "integer", "description": "Number of guests"},
                    "notes": {"type": "string", "description": "Optional requests, allergies, occasion"},
                },
                "required": ["name", "phone", "date", "time", "party_size"],
            },
        },
    },
]


def _fail(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def normalize_phone(phone: str) -> str:
    """Keep digits and a leading plus sign."""
    cleaned = re.sub(r"[^\d+]", "", phone or "")
    return ("+" if cleaned.startswith("+") else "") + cleaned.replace("+", "")


class ReservationTools:
    def __init__(self, store: Store, settings: Settings, clock: Callable[[], datetime]) -> None:
        self.store = store
        self.settings = settings
        self.clock = clock

    def call(self, name: str, arguments: dict[str, Any], session_id: str) -> dict[str, Any]:
        if name == "check_availability":
            return self.check_availability(**_only(arguments, "date", "time", "party_size"))
        if name == "create_reservation":
            return self.create_reservation(
                session_id=session_id,
                **_only(arguments, "name", "phone", "date", "time", "party_size", "notes"),
            )
        return _fail(f"Unknown tool: {name}")

    def _validate_slot(self, date: Any, time: Any, party_size: Any) -> tuple[str, str, int] | dict[str, Any]:
        try:
            day = Date.fromisoformat(str(date))
        except ValueError:
            return _fail("date must be YYYY-MM-DD")
        try:
            parsed_time = datetime.strptime(str(time), "%H:%M")
        except ValueError:
            return _fail("time must be HH:MM in 24-hour format")
        try:
            size = int(party_size)
        except (TypeError, ValueError):
            return _fail("party_size must be a whole number")

        hhmm = parsed_time.strftime("%H:%M")
        now = self.clock()
        if size < 1:
            return _fail("party_size must be at least 1")
        if size > self.settings.max_party_size:
            return _fail(
                f"Parties larger than {self.settings.max_party_size} cannot be booked online;"
                " ask the guest to call the venue."
            )
        if day < now.date() or (day == now.date() and _minutes(hhmm) <= now.hour * 60 + now.minute):
            return _fail("That date and time is in the past.")
        if not _minutes(self.settings.opening_time) <= _minutes(hhmm) < _minutes(self.settings.closing_time):
            return _fail(
                f"Outside opening hours ({self.settings.opening_time} to {self.settings.closing_time})."
            )
        return day.isoformat(), hhmm, size

    def _seats_taken(self, date: str, time: str) -> int:
        start = _minutes(time)
        return sum(
            row["party_size"]
            for row in self.store.reservations_on(date)
            if abs(_minutes(row["time"]) - start) < self.settings.slot_minutes
        )

    def check_availability(self, date: Any = None, time: Any = None, party_size: Any = None) -> dict[str, Any]:
        slot = self._validate_slot(date, time, party_size)
        if isinstance(slot, dict):
            return slot
        day, hhmm, size = slot
        free = self.settings.seat_capacity - self._seats_taken(day, hhmm)
        if size > free:
            return {"ok": True, "available": False, "seats_free": max(free, 0), "date": day, "time": hhmm}
        return {"ok": True, "available": True, "seats_free": free, "date": day, "time": hhmm}

    def create_reservation(
        self,
        session_id: str,
        name: Any = None,
        phone: Any = None,
        date: Any = None,
        time: Any = None,
        party_size: Any = None,
        notes: Any = "",
    ) -> dict[str, Any]:
        guest = str(name or "").strip()
        if len(guest) < 2:
            return _fail("name is missing")
        number = normalize_phone(str(phone or ""))
        if len(number.lstrip("+")) < 7:
            return _fail("phone is missing or too short")

        availability = self.check_availability(date, time, party_size)
        if not availability["ok"]:
            return availability
        if not availability["available"]:
            return _fail(
                f"Not enough seats at that time ({availability['seats_free']} free). Offer another time."
            )

        reservation_id = self.store.add_reservation(
            session_id=session_id,
            name=guest,
            phone=number,
            date=availability["date"],
            time=availability["time"],
            party_size=int(party_size),
            notes=str(notes or "").strip(),
            created_at=self.clock().isoformat(timespec="seconds"),
        )
        return {
            "ok": True,
            "reservation_id": reservation_id,
            "name": guest,
            "phone": number,
            "date": availability["date"],
            "time": availability["time"],
            "party_size": int(party_size),
        }


def _only(arguments: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Drop arguments the tool does not accept, so a hallucinated key cannot raise."""
    return {key: arguments[key] for key in keys if key in arguments}
