import asyncio
import json
from pathlib import Path

from citizen_agent.agent import Agent
from citizen_agent.evaluation import evaluate
from citizen_agent.models import Decision, ToolCall, Turn
from citizen_agent.providers import ScriptedProvider

ROOT = Path(__file__).resolve().parents[1]


def read_call(name, **args):
    return Turn(calls=[ToolCall(id=name, name=name, arguments=args)])


def debt_repair_turns():
    route = read_call("get_routing_rules", category="tax_debt")
    good = Decision(
        category="tax_debt",
        priority="normal",
        draft_answer="Сальдо ЕНС можно посмотреть в личном кабинете. Вопрос по сальдо можно задать там же.",
        source_ids=["fns_tax_debt"],
        needs_human=False,
        handoff_reason="",
    )
    bad = good.model_copy(update={"draft_answer": "Сальдо ЕНС доступно в кабинете (SOURCE_ID_fns_tax_debt)."})
    return (
        [
            read_call("search_knowledge_base", category="tax_debt", query="сальдо ЕНС", limit=3),
            route,
            Turn(content="Ответ вне JSON"),
            route,
            Turn(content=bad.model_dump_json()),
            route,
        ],
        good,
        route,
    )


def test_reported_debt_repeat_failure_gets_one_repair_without_third_read(executor_factory, config):
    turns, good, _ = debt_repair_turns()
    provider = ScriptedProvider([*turns, Turn(content=good.model_dump_json())])
    executor = executor_factory()
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run("Где посмотреть сальдо ЕНС?"))
    assert result.status == "draft_ready" and result.category == "tax_debt"
    assert [s.source_id for s in result.sources] == ["fns_tax_debt"]
    assert result.steps == 7 and result.tool_calls == 4
    events = [e for e in executor.trace.events if e["event"] == "tool" and e["tool"] == "get_routing_rules"]
    assert sum(e["attempts"] for e in events) == 2
    blocked = [e for e in events if e.get("blocked")]
    assert len(blocked) == 1 and blocked[0]["attempts"] == 0
    assert blocked[0]["result"]["error"]["type"] == "repeat_limit"
    last = provider.requests[-1]["messages"]
    assert json.loads(last[-1]["content"])["error"]["type"] == "repeat_limit"


def test_ignoring_repeat_notice_still_stops_within_existing_limits(executor_factory, config):
    turns, _, route = debt_repair_turns()
    provider = ScriptedProvider([*turns, route] * 2)
    executor = executor_factory()
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run("Где посмотреть сальдо ЕНС?"))
    assert result.needs_human and "повтор" in result.handoff_reason
    assert result.steps == 7
    assert sum(e["event"] == "tool" and e.get("blocked", False) for e in executor.trace.events) == 1
    assert (
        sum(
            e.get("attempts", 0)
            for e in executor.trace.events
            if e["event"] == "tool" and e["tool"] == "get_routing_rules"
        )
        == 2
    )


def test_failed_reads_do_not_qualify_for_repeat_repair(executor_factory, config):
    route = read_call("get_routing_rules", category="tax_debt")
    executor = executor_factory(faults={"get_routing_rules": ["unavailable"] * 6})
    result = asyncio.run(
        Agent(
            ScriptedProvider([Turn(content="not json"), route, route, route]),
            executor,
            config,
            executor.trace,
        ).run("Вопрос о ЕНС")
    )
    assert result.steps == 4 and "повтор" in result.handoff_reason
    assert not any(e.get("blocked") for e in executor.trace.events)


def test_repeat_repair_never_allows_extra_action(executor_factory, config):
    action = read_call("escalate_to_operator", reason="out_of_scope", summary="Нужен ручной разбор")
    bad = Decision(
        category="other",
        priority="normal",
        draft_answer="Неподтверждённая справка",
        source_ids=[],
        needs_human=True,
        handoff_reason="Иная тема",
    )
    executor = executor_factory(confirm=lambda _: True)
    result = asyncio.run(
        Agent(
            ScriptedProvider([action, action, Turn(content=bad.model_dump_json()), action]),
            executor,
            config,
            executor.trace,
        ).run("Иная тема")
    )
    assert result.steps == 4 and result.status == "escalated"
    assert (
        sum(e["event"] == "tool" and e["tool"] == "escalate_to_operator" for e in executor.trace.events) == 2
    )
    assert not any(e.get("blocked") for e in executor.trace.events)


def test_repeat_notice_does_not_extend_step_budget(executor_factory, config):
    config.runtime.max_steps = 6
    turns, good, _ = debt_repair_turns()
    executor = executor_factory()
    provider = ScriptedProvider([*turns, Turn(content=good.model_dump_json())])
    result = asyncio.run(Agent(provider, executor, config, executor.trace).run("Где посмотреть сальдо ЕНС?"))
    assert result.steps == 6 and result.needs_human and "шагов" in result.handoff_reason
    assert len(provider.requests) == 6


def test_report_distinguishes_executed_reads_from_blocked_intent(config, tmp_path):
    turns, good, _ = debt_repair_turns()
    case = {
        "id": "bounded_repeat_repair",
        "text": "Где посмотреть сальдо ЕНС?",
        "script": [t.model_dump() for t in [*turns, Turn(content=good.model_dump_json())]],
        "expected": {
            "categories": ["tax_debt"],
            "needs_human": False,
            "statuses": ["draft_ready"],
            "tools": ["search_knowledge_base", "get_routing_rules"],
            "source_ids": ["fns_tax_debt"],
        },
    }
    report = asyncio.run(evaluate(config, ROOT, [case], 1, tmp_path / "report.json"))
    assert report["runs"][0]["passed"]
    diag = report["runs"][0]["diagnostics"]
    assert len(diag["successful_reads"]) == 3
    assert diag["successful_reads"][0]["source_ids"] == ["fns_tax_debt"]
    assert diag["blocked_reads"] == [{"tool": "get_routing_rules", "type": "repeat_limit", "attempts": 0}]
