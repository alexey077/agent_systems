from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["account_access", "tax_notice", "property_error", "deduction", "tax_debt", "other"]
Priority = Literal["normal", "high"]
Reason = Literal[
    "ambiguous",
    "out_of_scope",
    "missing_evidence",
    "tool_failure",
    "personal_data_required",
    "security",
    "limit_reached",
    "invalid_answer",
    "provider_failure",
    "manual_review",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArgs(StrictModel):
    query: str = Field(
        min_length=3, max_length=400, description="Короткий поисковый запрос без персональных данных"
    )
    category: Category = Field(description="Тема обращения; other для общего поиска")
    limit: int = Field(default=3, ge=1, le=5, description="Максимальное число найденных фрагментов: 1–5")


class RoutingArgs(StrictModel):
    category: Category = Field(description="Одна тема из учебного классификатора")


class EscalateArgs(StrictModel):
    reason: Reason = Field(description="Причина ручного разбора из разрешённого перечня")
    summary: str = Field(
        min_length=5, max_length=600, description="Краткая сводка без ФИО, контактов и реквизитов"
    )


class Decision(StrictModel):
    category: Category
    priority: Priority
    draft_answer: str = Field(max_length=5000)
    source_ids: list[str] = Field(max_length=5)
    needs_human: bool
    handoff_reason: str = Field(max_length=600)


class Source(StrictModel):
    source_id: str
    title: str
    url: str
    category: Category
    text: str
    keywords: list[str]
    checked_at: str


class Citation(StrictModel):
    source_id: str
    title: str
    url: str


class CaseResult(StrictModel):
    request_id: str
    category: Category = "other"
    priority: Priority = "normal"
    recipient: str = "Ручной разбор (учебная очередь)"
    draft_answer: str = ""
    sources: list[Citation] = Field(default_factory=list)
    needs_human: bool = True
    handoff_reason: str = ""
    status: Literal["draft_ready", "needs_review", "escalated", "cancelled"] = "needs_review"
    ticket_id: str | None = None
    steps: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    estimated_cost: float | None = None
    provider: str = ""
    model: str = ""
    trace_path: str = ""


class ToolCall(StrictModel):
    id: str
    name: str
    arguments: Any


class Usage(StrictModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Turn(StrictModel):
    content: str = ""
    calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    state_id: str | None = None
    finish_reason: str = "stop"


class Session(StrictModel):
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    actor_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    organization: str = "course_demo"
    can_read: bool = True
    can_escalate: bool = True


def error(kind: str, message: str, hint: str, **extra: Any) -> dict:
    return {"ok": False, "error": {"type": kind, "message": message, "hint": hint, **extra}}
