from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .privacy import redact


class Trace:
    def __init__(self, directory: str | Path, run_id: str):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{run_id}.jsonl"
        self.events: list[dict] = []

    def emit(self, event: str, **data):
        entry = redact({"ts": datetime.now(UTC).isoformat(), "event": event, **data})
        self.events.append(entry)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
