from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from .models import Session
from .privacy import redact


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY, organization TEXT NOT NULL,
                    request_id TEXT NOT NULL, actor_id TEXT NOT NULL,
                    reason TEXT NOT NULL, summary TEXT NOT NULL,
                    confirmed_at TEXT NOT NULL,
                    UNIQUE(organization, request_id)
                );
                CREATE TABLE IF NOT EXISTS results (
                    organization TEXT NOT NULL, request_id TEXT NOT NULL,
                    result TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(organization, request_id)
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=3)

    def ticket(self, session: Session) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT ticket_id, reason, summary, confirmed_at FROM tickets WHERE organization=? AND request_id=?",
                (session.organization, session.request_id),
            ).fetchone()
        if not row:
            return None
        return dict(zip(("ticket_id", "reason", "summary", "confirmed_at"), row, strict=True))

    def escalate(self, session: Session, reason: str, summary: str) -> dict:
        # One current-case handoff = one server-generated idempotency key.
        digest = hashlib.sha256(
            f"{session.organization}:{session.request_id}:handoff:v1".encode()
        ).hexdigest()
        ticket_id = "review_" + digest[:16]
        with self._lock, self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ticket_id,
                    session.organization,
                    session.request_id,
                    session.actor_id,
                    reason,
                    redact(summary),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return self.ticket(session)

    def save_result(self, session: Session, result: dict):
        with self._lock, self.connect() as db:
            db.execute(
                "INSERT INTO results VALUES (?, ?, ?, ?) ON CONFLICT(organization, request_id) "
                "DO UPDATE SET result=excluded.result, updated_at=excluded.updated_at",
                (
                    session.organization,
                    session.request_id,
                    json.dumps(redact(result), ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def queue(self, organization: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT result FROM results WHERE organization=? ORDER BY updated_at DESC", (organization,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
