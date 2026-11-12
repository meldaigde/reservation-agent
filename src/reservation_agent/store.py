"""SQLite storage: reservations, conversation memory and traces."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    party_size INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    step INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    ok INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces (session_id, id);
CREATE INDEX IF NOT EXISTS idx_reservations_date ON reservations (date);
"""


class Store:
    """One SQLite connection guarded by a lock, so FastAPI worker threads can share it."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    # Reservations

    def add_reservation(
        self,
        *,
        session_id: str,
        name: str,
        phone: str,
        date: str,
        time: str,
        party_size: int,
        notes: str,
        created_at: str,
    ) -> int:
        cur = self._execute(
            "INSERT INTO reservations (session_id, name, phone, date, time, party_size, notes, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, name, phone, date, time, party_size, notes, created_at),
        )
        return int(cur.lastrowid)

    def reservations_on(self, date: str) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM reservations WHERE date = ? ORDER BY time", (date,))

    def list_reservations(self, date: str | None = None) -> list[dict[str, Any]]:
        if date:
            return self.reservations_on(date)
        return self._query("SELECT * FROM reservations ORDER BY date, time")

    # Conversation memory

    def add_message(self, session_id: str, message: dict[str, Any]) -> None:
        self._execute(
            "INSERT INTO messages (session_id, payload) VALUES (?, ?)",
            (session_id, json.dumps(message, ensure_ascii=False)),
        )

    def history(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._query("SELECT payload FROM messages WHERE session_id = ? ORDER BY id", (session_id,))
        return [json.loads(row["payload"]) for row in rows]

    def turn_count(self, session_id: str) -> int:
        rows = self._query(
            "SELECT COALESCE(MAX(turn), 0) AS turns FROM traces WHERE session_id = ?", (session_id,)
        )
        return int(rows[0]["turns"])

    # Traces

    def add_trace(
        self,
        *,
        session_id: str,
        turn: int,
        step: int,
        kind: str,
        name: str,
        ok: bool,
        latency_ms: float,
        created_at: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            "INSERT INTO traces (session_id, turn, step, kind, name, ok, latency_ms, prompt_tokens,"
            " completion_tokens, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn,
                step,
                kind,
                name,
                int(ok),
                round(latency_ms, 2),
                prompt_tokens,
                completion_tokens,
                json.dumps(detail or {}, ensure_ascii=False),
                created_at,
            ),
        )

    def traces(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM traces WHERE session_id = ? ORDER BY id", (session_id,))
        for row in rows:
            row["ok"] = bool(row["ok"])
            row["detail"] = json.loads(row["detail"])
        return rows
