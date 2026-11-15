from __future__ import annotations

import re

PERSONAL = r"\b(?:мо(?:й|я|е|ё|ю|и|его|ей|ему|им|ем|ём|их)|мне|у\s+меня)\b"
TAX_RECORD = r"\b(?:декларац\w*|вычет\w*|возврат\w*|задолжен\w*|долг\w*|налог\w*|сальдо|енс|плат[её]ж\w*|ден[еь]г\w*)\b"
LOOKUP_COMMAND = (
    r"\b(?:проверь(?:те)?|посмотри(?:те)?|узнай(?:те)?|сообщи(?:те)?|уточни(?:те)?|скаж(?:и|ите))\b"
)
INSTRUCTIONS = r"\b(?:где|как|в\s+каком\s+разделе|каким\s+способом)\b.{0,100}\b(?:самостоятельно|самому|посмотр\w*|провер\w*|узна\w*|отслед\w*|найти)\b"
PERSONAL_STATUS = r"\b(?:когда|одобр\w*|проверен\w*|зачисл\w*|перечисл\w*|вернул\w*|поступил\w*|сколько|сумм\w*|какой\s+статус|какое\s+сальдо)\b"


def requires_personal_lookup(text: str) -> bool:
    """Conservative guard for explicit private-record requests, not a general intent classifier."""
    for part in re.split(r"[.!?;\n]+", text.lower().replace("ё", "е")):
        if not re.search(PERSONAL, part) or not re.search(TAX_RECORD, part):
            continue
        for command in re.finditer(LOOKUP_COMMAND, part):
            tail = part[command.end() :]
            # 'Скажите, где посмотреть ...' asks for instructions, not private access.
            if re.match(r"\s*,?\s*(?:как|где|в\s+каком)\b", tail):
                continue
            if re.search(PERSONAL, tail[:180]):
                return True
        if re.search(INSTRUCTIONS, part):
            continue
        if re.search(PERSONAL_STATUS, part):
            return True
    return False
