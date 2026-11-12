"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    business_name: str = "Demo Cafe"
    opening_time: str = "09:00"
    closing_time: str = "23:00"
    max_party_size: int = 12
    # Seats that can be booked within one overlapping window.
    seat_capacity: int = 30
    # Two reservations overlap when they start less than this many minutes apart.
    slot_minutes: int = 120
    db_path: str = "reservations.db"
    api_key: str = ""
    model: str = ""
    fallback_model: str = ""
    base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            business_name=env.get("BUSINESS_NAME", cls.business_name),
            opening_time=env.get("OPENING_TIME", cls.opening_time),
            closing_time=env.get("CLOSING_TIME", cls.closing_time),
            max_party_size=int(env.get("MAX_PARTY_SIZE", cls.max_party_size)),
            seat_capacity=int(env.get("SEAT_CAPACITY", cls.seat_capacity)),
            slot_minutes=int(env.get("SLOT_MINUTES", cls.slot_minutes)),
            db_path=env.get("DB_PATH", cls.db_path),
            api_key=env.get("OPENROUTER_API_KEY", ""),
            model=env.get("MODEL", ""),
            fallback_model=env.get("FALLBACK_MODEL", ""),
            base_url=env.get("LLM_BASE_URL", cls.base_url),
        )
