import asyncio
import json
from pathlib import Path

import pytest

from citizen_agent.agent import Agent
from citizen_agent.evaluation import evaluate
from citizen_agent.models import Decision, Session, ToolCall, Turn
from citizen_agent.policy import requires_personal_lookup
from citizen_agent.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text",
    [
        "Проверьте мою декларацию и скажите, одобрили ли мне вычет.",
        "Посмотрите сумму моей задолженности по ЕНС.",
        "Когда поступит возврат по моей декларации?",
        "Мой вычет уже одобрен?",
        "Какой статус у моей декларации?",
        "Сколько денег мне вернут по вычету?",
        "Сообщите мне сальдо ЕНС.",
        "Когда придут мои деньги по вычету?",
        "Проверьте мои платежи по налогам.",
    ],
)
def test_explicit_private_record_lookup_requires_human(text):
    assert requires_personal_lookup(text)


@pytest.mark.parametrize(
    "text",
    [
        "Где самостоятельно посмотреть статус проверки декларации на вычет?",
        "Как мне узнать, одобрен ли мой вычет?",
        "Скажите, где посмотреть статус моей декларации.",
        "Как самому проверить сумму моей задолженности?",
        "Где посмотреть сальдо ЕНС и как сообщить об ошибке?",
        "Забыл пароль от кабинета, как восстановить доступ?",
        "Подскажите, как отследить мою декларацию.",
    ],
)
def test_instructions_for_self_service_remain_supported(text):
    assert not requires_personal_lookup(text)


def manual(category="other", draft=""):
    return Decision(
        category=category,
        priority="normal",
        draft_answer=draft,
        source_ids=[],
        needs_human=True,
        handoff_reason="Нужен ручной разбор",
    )


def call(name, **args):
    return Turn(calls=[ToolCall(id=name, name=name, arguments=args)])


def final(d):
    return Turn(content=d.model_dump_json())


def test_premature_human_flag_cannot_skip_available_search(executor_factory, config):
    case = next(
        c
        for c in json.loads((ROOT / "data/scenarios.json").read_text())["scenarios"]
        if c["id"] == "account_normal"
    )
    provider = ScriptedProvider([final(manual("account_access", "Совет без источника")), *case["script"]])
    executor = executor_factory()
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run(case["text"]))
    assert result.status == "draft_ready" and not result.needs_human
    assert any(
        e["event"] == "answer_rejected" and "сначала проверь базу" in e["reason"]
        for e in executor.trace.events
    )


@pytest.mark.parametrize("approved,status,tickets", [(True, "escalated", True), (False, "cancelled", False)])
def test_premature_manual_final_must_reach_real_confirmation(
    executor_factory, config, approved, status, tickets
):
    executor = executor_factory(confirm=lambda _: approved)
    provider = ScriptedProvider(
        [
            final(manual()),
            call(
                "escalate_to_operator", reason="out_of_scope", summary="Нужен разбор обращения по иной теме"
            ),
            final(manual()),
        ]
    )
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run("Нужна помощь по иной теме"))
    assert result.status == status
    assert bool(executor.store.ticket(executor.session)) == tickets
    assert any(e["event"] == "answer_rejected" for e in executor.trace.events)


def test_permission_failure_allows_review_without_false_ticket(executor_factory, config):
    executor = executor_factory(
        session=Session(request_id="no_rights", actor_id="tester", can_escalate=False)
    )
    provider = ScriptedProvider(
        [
            final(manual()),
            call(
                "escalate_to_operator", reason="out_of_scope", summary="Нужен разбор обращения по иной теме"
            ),
            final(manual()),
        ]
    )
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run("Вопрос по иной теме"))
    assert result.status == "needs_review" and result.ticket_id is None
    assert any(
        e["event"] == "tool" and e["result"].get("error", {}).get("type") == "forbidden"
        for e in executor.trace.events
    )


def test_personal_request_cannot_be_answered_with_generic_source(executor_factory, config):
    case = next(
        c
        for c in json.loads((ROOT / "data/scenarios.json").read_text())["scenarios"]
        if c["id"] == "deduction_normal"
    )
    executor = executor_factory(confirm=lambda _: True)
    provider = ScriptedProvider(
        [
            *case["script"],
            call(
                "escalate_to_operator",
                reason="personal_data_required",
                summary="Нужно проверить личную запись",
            ),
            final(manual("deduction")),
        ]
    )
    result = asyncio.run(
        Agent(provider, executor, config, executor.trace).run("Проверьте мою декларацию на вычет")
    )
    assert result.status == "escalated" and result.draft_answer == ""
    assert result.recipient == "Ручной разбор (учебная очередь)"
    assert any(
        e["event"] == "answer_rejected" and "личной записи" in e["reason"] for e in executor.trace.events
    )


def test_handoff_does_not_keep_unsupported_factual_advice(executor_factory, config):
    executor = executor_factory(confirm=lambda _: True)
    provider = ScriptedProvider(
        [
            call(
                "escalate_to_operator", reason="out_of_scope", summary="Нужен разбор обращения по иной теме"
            ),
            final(manual(draft="Сочинённая рекомендация без источника")),
            final(manual()),
        ]
    )
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run("Иная тема"))
    assert result.status == "escalated" and result.draft_answer == ""
    assert any(
        e["event"] == "answer_rejected" and "должен быть пустым" in e["reason"] for e in executor.trace.events
    )


def test_source_id_repair_gets_exact_existing_ids_and_state(executor_factory, config):
    case = next(
        c
        for c in json.loads((ROOT / "data/scenarios.json").read_text())["scenarios"]
        if c["id"] == "fabricated_source_repair"
    )
    executor = executor_factory()
    provider = ScriptedProvider(case["script"])
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run(case["text"]))
    assert result.status == "draft_ready"
    correction = next(
        m["content"]
        for m in provider.requests[-1]["messages"]
        if m["role"] == "user" and "Допустимые source_ids" in m["content"]
    )
    assert "fns_account_access" in correction
    assert (
        '"source_ids_by_category": {"account_access": ["fns_account_access"]}'
        in provider.requests[-1]["messages"][0]["content"]
    )


def test_made_up_markdown_source_links_are_not_accepted():
    decision = Decision(
        category="account_access",
        priority="normal",
        draft_answer="Ответ [источник](SOURCE_ID_fns_account_access)",
        source_ids=["fns_account_access"],
        needs_human=False,
        handoff_reason="",
    )
    reason = Agent.check_decision(
        decision,
        {"fns_account_access": {"category": "account_access"}},
        {"account_access": {}},
        "Как войти в кабинет?",
    )
    assert reason and "URL" in reason


def test_invalid_action_arguments_do_not_satisfy_handoff_attempt(executor_factory, config):
    executor = executor_factory()
    config.runtime.max_steps = 2
    provider = ScriptedProvider(
        [call("escalate_to_operator", reason="invented", summary="Недопустимая причина"), final(manual())]
    )
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run("Иная тема"))
    assert result.status == "needs_review" and result.ticket_id is None
    assert any(
        e["event"] == "answer_rejected" and "не создаёт задачу" in e["reason"] for e in executor.trace.events
    )


def test_evaluation_keeps_failures_with_diagnostics_and_progress(config, tmp_path):
    cases = [
        {
            "id": "deliberately_wrong_label",
            "text": "Иная тема",
            "script": [final(manual()).model_dump()],
            "expected": {"categories": ["account_access"], "needs_human": False, "statuses": ["draft_ready"]},
        }
    ]
    updates = []
    report = asyncio.run(evaluate(config, ROOT, cases, 1, tmp_path / "report.json", progress=updates.append))
    assert report["metrics"]["scenario_pass_rate"] == 0
    assert report["run_count"] == 1 and not report["runs"][0]["passed"]
    assert report["failed_scenarios"][0]["failed_runs"] == 1
    assert report["runs"][0]["diagnostics"]["answer_rejections"]
    assert updates[0]["completed"] == updates[0]["total"] == 1


def test_safe_fallback_preserves_verified_category(executor_factory, config):
    executor = executor_factory()
    provider = ScriptedProvider([call("get_routing_rules", category="account_access")])
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run("Как войти в кабинет?"))
    assert result.category == "account_access"
    assert result.status == "needs_review" and result.needs_human
    assert result.draft_answer == "" and result.ticket_id is None
