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


def draft_quality_issue(draft: str, source_texts: list[str]) -> str | None:
    """Small explicit regression guards; this is not a general factual-entailment judge."""
    if re.search(r"\b(?:source_ids?\b|source_id_[\w-]*|fns_[a-z0-9_]+)", draft, re.I):
        return "В draft_answer попал служебный ID. Оставь IDs только в source_ids; в тексте нужны обычные предложения"
    if re.search(r"(?im)^\s*(?:\#{1,6}\s*)?(?:\*\*)?(?:источники?|sources?)\s*[:：]", draft):
        return (
            "Убери из draft_answer блок источников: проверенные название и ссылка уже добавляются приложением"
        )
    sources = "\n".join(source_texts).lower().replace("ё", "е")
    normalized = draft.lower().replace("ё", "е")
    if re.search(r"отдельн\w*\s+(?:официальн\w*\s+)?сервис", sources):
        shared_channel = re.search(
            r"\bличн\w*\s+кабинет\w*(?P<link>.{0,160}?)\bтам\s+же\b[^.!?]{0,120}\bзапис",
            normalized,
            re.S,
        )
        if shared_channel and not re.search(r"отдельн|онлайн-запис", shared_channel["link"]):
            return "Запись на приём относится к отдельному сервису из источника. Не приписывай её личному кабинету словами «там же»"
    if re.search(r"\bпо месту (?:учета|жительства|регистрации)\b", normalized) and not re.search(
        r"\bпо месту (?:учета|жительства|регистрации)\b", sources
    ):
        return "Уточнение о месте обращения не подтверждено найденным фрагментом. Не добавляй ограничения места от себя"
    duty = r"\b(?:обязанност\w*|обязаны|обязан[ао]?|освобожд\w*)\b"
    if re.search(duty, normalized) and not re.search(duty, sources):
        return "Правовой вывод об обязанности или освобождении от неё отсутствует в найденном фрагменте. Убери этот вывод"
    if re.search(r"отсутствие бумажн\w* письм", sources):
        for sentence in re.split(r"[.!?;\n]+", normalized):
            reversed_condition = re.search(r"\bбумажн\w* письм\w*.*\bне\b.*(?:гарантир|означ)", sentence)
            absence = re.search(r"отсутств|неполуч|\bне\s+(?:получ\w*|приш\w*)", sentence)
            if reversed_condition and not absence:
                return (
                    "Изменено условие исходного утверждения: источник говорит об отсутствии бумажного письма. "
                    "Сохрани это условие и отрицание; не делай нового вывода о наличии письма"
                )
    return None
