"""JSONL log of every agent call. `kind` is `llm` or `tool` for calls, `event` for runtime events."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class CallLog:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def call(
        self,
        *,
        agent: str,
        kind: str,
        name: str,
        duration_ms: int,
        status: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        error: str | None = None,
        attempt: int | None = None,
    ) -> None:
        self._write(
            {
                "agent": agent,
                "kind": kind,
                "name": name,
                "duration_ms": duration_ms,
                "status": status,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "attempt": attempt,
                "error": error,
            }
        )

    def event(self, name: str, **fields) -> None:
        self._write({"agent": "runtime", "kind": "event", "name": name, **fields})

    def _write(self, record: dict) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), "run_id": self.run_id, **record}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
