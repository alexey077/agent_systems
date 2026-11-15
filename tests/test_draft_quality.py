import asyncio
import copy
import json
from pathlib import Path

import pytest

from citizen_agent.agent import Agent
from citizen_agent.evaluation import evaluate
from citizen_agent.models import Decision, Turn
from citizen_agent.policy import draft_quality_issue
from citizen_agent.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = {s["source_id"]: s for s in json.loads((ROOT / "data/knowledge.json").read_text())["sources"]}


@pytest.mark.parametrize(
    "draft",
    [
        "Сведения (SOURCE_ID_fns_tax_debt)",
        "Сведения fns_tax_debt.",
        "Сведения source_id: документ.",
        "Сведения.\n**Источник:** выдуманное название",
        "Сведения.\nИсточник: пересказ",
    ],
)
def test_technical_markers_and_duplicate_source_sections_are_rejected(draft):
    assert draft_quality_issue(draft, [KNOWLEDGE["fns_tax_debt"]["text"]])


@pytest.mark.parametrize(
    "draft",
    [
        "Бумажное письмо не всегда гарантирует факт начисления налогов.",
        "Бумажное письмо не означает, что есть начисления.",
    ],
)
def test_absence_condition_must_not_be_changed_to_presence(draft):
    reason = draft_quality_issue(draft, [KNOWLEDGE["fns_tax_notice"]["text"]])
    assert reason and "условие" in reason


@pytest.mark.parametrize(
    "draft",
    [
        "Отсутствие бумажного письма не означает отсутствия начислений.",
        "Если вы не получили бумажное письмо, это не означает отсутствия начислений.",
        "Бумажное письмо не пришло: проверьте электронные каналы.",
        "Неполучение бумажного письма само по себе не означает, что начислений нет.",
    ],
)
def test_supported_absence_paraphrases_are_allowed(draft):
    assert draft_quality_issue(draft, [KNOWLEDGE["fns_tax_notice"]["text"]]) is None


def test_new_legal_duty_claim_requires_support_in_source():
    draft = "Неполучение документа не освобождает от обязанности уплатить налог."
    reason = draft_quality_issue(draft, [KNOWLEDGE["fns_tax_notice"]["text"]])
    assert reason and "Правовой вывод" in reason
    assert draft_quality_issue(draft, [draft]) is None


def test_extra_jurisdiction_restriction_requires_support():
    draft = "Для сверки обратитесь в налоговый орган по месту учёта."
    assert draft_quality_issue(draft, [KNOWLEDGE["fns_tax_debt"]["text"]])
    assert draft_quality_issue(draft, ["Можно обратиться в налоговый орган по месту учёта."]) is None


@pytest.mark.parametrize(
    "scenario,bad_draft",
    [
        ("priority_high", "Бумажное письмо не всегда гарантирует факт начисления налогов."),
        ("debt_normal", "Проверьте сальдо в кабинете (SOURCE_ID_fns_tax_debt)."),
    ],
)
def test_observed_bad_draft_is_repaired_before_acceptance(executor_factory, config, scenario, bad_draft):
    case = next(
        c for c in json.loads((ROOT / "data/scenarios.json").read_text())["scenarios"] if c["id"] == scenario
    )
    correct = Decision.model_validate_json(case["script"][-1]["content"])
    bad = correct.model_copy(update={"draft_answer": bad_draft})
    provider = ScriptedProvider(
        [*case["script"][:-1], Turn(content=bad.model_dump_json()), case["script"][-1]]
    )
    executor = executor_factory()
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run(case["text"]))
    assert result.status == "draft_ready" and result.draft_answer == correct.draft_answer
    assert any(e["event"] == "answer_rejected" for e in executor.trace.events)


def test_priority_is_scored_and_cannot_silently_default_to_normal(config, tmp_path):
    case = copy.deepcopy(
        next(
            c
            for c in json.loads((ROOT / "data/scenarios.json").read_text())["scenarios"]
            if c["id"] == "priority_high"
        )
    )
    final = json.loads(case["script"][-1]["content"])
    final["priority"] = "normal"
    case["script"][-1]["content"] = json.dumps(final, ensure_ascii=False)
    report = asyncio.run(evaluate(config, ROOT, [case], 1, tmp_path / "report.json"))
    checks = report["runs"][0]["checks"]
    assert all(v for k, v in checks.items() if k != "priority")
    assert checks["priority"] is False and report["metrics"]["priority_accuracy"] == 0
    assert report["evaluation_schema_version"] == 2 and not report["runs"][0]["passed"]


@pytest.mark.parametrize(
    "draft,issue",
    [
        ("Проверьте сведения в личном кабинете. Там же можно задать вопрос или записаться на приём.", True),
        ("Вопрос задайте в личном кабинете. Для записи используйте отдельный сервис онлайн-записи.", False),
        (
            "Вопрос задайте в личном кабинете. Откройте отдельный сервис онлайн-записи. Там же можно записаться на приём.",
            False,
        ),
        ("Откройте сервис онлайн-записи. Там же можно записаться на приём.", False),
    ],
)
def test_separate_booking_service_is_not_silently_moved_into_account(draft, issue):
    assert bool(draft_quality_issue(draft, [KNOWLEDGE["fns_tax_debt"]["text"]])) == issue
