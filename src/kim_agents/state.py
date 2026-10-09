"""Explicit task state in SQLite: runs, blueprint cells, items with their status, and every message."""

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from .messages import Envelope

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, teacher_id TEXT, status TEXT NOT NULL,
    message TEXT, started_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS cells (
    run_id TEXT NOT NULL, cell_id TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL,
    PRIMARY KEY (run_id, cell_id)
);
CREATE TABLE IF NOT EXISTS items (
    run_id TEXT NOT NULL, item_id TEXT NOT NULL, cell_id TEXT NOT NULL, status TEXT NOT NULL,
    attempt INTEGER NOT NULL, body TEXT NOT NULL, verdict TEXT,
    PRIMARY KEY (run_id, item_id)
);
CREATE TABLE IF NOT EXISTS messages (
    msg_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, sender TEXT NOT NULL, recipient TEXT NOT NULL,
    type TEXT NOT NULL, attempt INTEGER NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StateStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _exec(self, sql: str, params: tuple) -> None:
        with self._lock, self._db:
            self._db.execute(sql, params)

    def start_run(self, run_id: str) -> None:
        self._exec("INSERT INTO runs (run_id, status, started_at) VALUES (?, 'running', ?)", (run_id, _now()))

    def set_teacher(self, run_id: str, teacher_id: str) -> None:
        self._exec("UPDATE runs SET teacher_id = ? WHERE run_id = ?", (teacher_id, run_id))

    def finish_run(self, run_id: str, status: str, message: str) -> None:
        self._exec(
            "UPDATE runs SET status = ?, message = ?, finished_at = ? WHERE run_id = ?",
            (status, message, _now(), run_id),
        )

    def put_cell(self, run_id: str, cell_id: str, status: str, body: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO cells (run_id, cell_id, status, body) VALUES (?, ?, ?, ?)",
            (run_id, cell_id, status, json.dumps(body, ensure_ascii=False)),
        )

    def put_item(self, run_id: str, item_id: str, cell_id: str, status: str, attempt: int, body: dict, verdict: dict | None = None) -> None:
        self._exec(
            "INSERT OR REPLACE INTO items (run_id, item_id, cell_id, status, attempt, body, verdict) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, item_id, cell_id, status, attempt, json.dumps(body, ensure_ascii=False),
             json.dumps(verdict, ensure_ascii=False) if verdict else None),
        )

    def put_message(self, env: Envelope) -> None:
        self._exec(
            "INSERT INTO messages (msg_id, run_id, sender, recipient, type, attempt, created_at, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (env.msg_id, env.run_id, env.sender, env.recipient, env.type, env.attempt, env.created_at,
             json.dumps(env.payload, ensure_ascii=False)),
        )

    def item_statuses(self, run_id: str) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute("SELECT item_id, status FROM items WHERE run_id = ?", (run_id,)).fetchall()
        return dict(rows)

    def close(self) -> None:
        self._db.close()
