from __future__ import annotations

import os
import re
from typing import Any

PATTERNS = [
    (r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]"),
    (r"(?<!\w)(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)", "[PHONE]"),
    (r"\b\d{10,20}\b", "[IDENTIFIER]"),
    (r"(?i)(?:паспорт|снилс|инн)\s*[:№]?\s*[\d -]{6,25}", "[DOCUMENT]"),
    (r"(?i)(?:Bearer|Basic)\s+[A-Za-z0-9+/=_.-]+", "[AUTH]"),
]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): (
                "[SECRET]"
                if re.search(r"(?i)(credential|secret|token|password|authorization)", str(k))
                and not str(k).endswith("_tokens")
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if not isinstance(value, str):
        return value
    text = value
    # Values of configured secrets must never reach traces even in exception payloads.
    for name, secret in os.environ.items():
        if re.search(r"(?i)(API_KEY|CREDENTIAL|TOKEN|SECRET)", name) and len(secret) >= 8:
            text = text.replace(secret, "[SECRET]")
    for pattern, replacement in PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def injection_flags(text: str) -> bool:
    return bool(
        re.search(
            r"(?is)(игнорир\w*\s+(?:все\s+)?(?:предыдущ|инструкц|правил)|ignore\s+(?:all\s+)?previous|"
            r"(?:покажи|выведи|раскрой|пришли|напечатай).{0,50}(?:api.?ключ|секрет|system.?prompt|системн\w*\s+промпт))",
            text,
        )
    )
