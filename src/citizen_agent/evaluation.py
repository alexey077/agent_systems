from __future__ import annotations

import json
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from .agent import Agent
from .config import AppConfig
from .models import Session
from .providers import make_provider
from .storage import Store
from .tools import Executor
from .trace import Trace


async def run_case(
    config: AppConfig,
    root: Path,
    text: str,
    *,
    request_id: str | None = None,
    confirm: Callable[[dict], bool] | None = None,
    script: list | None = None,
    faults: dict | None = None,
    session_flags: dict | None = None,
):
    session = Session(
        request_id=request_id or "appeal_" + uuid.uuid4().hex[:12],
        actor_id="demo_operator",
        **(session_flags or {}),
    )
    trace = Trace(root / config.traces_path, session.request_id + "_" + uuid.uuid4().hex[:8])
    store = Store(root / config.database_path)
    executor = Executor(
        knowledge_path=root / config.knowledge_path,
        routing_path=root / config.routing_path,
        store=store,
        session=session,
        runtime=config.runtime,
        trace=trace,
        confirm=confirm,
        faults=faults,
    )
    provider = make_provider(config.provider, trace=trace, script=script)
    try:
        result = await Agent(provider, executor, config, trace).run(text)
        return result, trace.events
    finally:
        await provider.aclose()


async def evaluate(
    config: AppConfig,
    root: Path,
    cases: list[dict],
    repeats: int,
    output: Path,
    *,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    runs = []
    for case in cases:
        for repetition in range(repeats):
            started = time.monotonic()
            run_config = config.model_copy(deep=True)
            run_config.database_path = str(output.parent / "evaluation.sqlite3")
            run_config.traces_path = str(output.parent / "traces")
            # In this deterministic test harness ONLY, the fixture simulates the UI decision.
            approved = case.get("confirm", False)
            confirm = lambda card, approved=approved: approved  # noqa: E731
            result, events = await run_case(
                run_config,
                root,
                case["text"],
                confirm=confirm,
                script=case.get("script") if config.provider.kind == "scripted" else None,
                faults=case.get("faults"),
                session_flags=case.get("session_flags"),
            )
            expected = case["expected"]
            observed_tools = [e["tool"] for e in events if e["event"] == "tool"]
            checks = {
                "category": result.category in expected["categories"],
                "handoff": result.needs_human == expected["needs_human"],
                "status": result.status in expected.get("statuses", [result.status]),
                "tools": set(expected.get("tools", [])) <= set(observed_tools),
                "sources": set(expected.get("source_ids", [])) <= {s.source_id for s in result.sources},
            }
            runs.append(
                {
                    "scenario_id": case["id"],
                    "repetition": repetition + 1,
                    "checks": checks,
                    "passed": all(checks.values()),
                    "seconds": round(time.monotonic() - started, 3),
                    "result": result.model_dump(),
                    "tools": observed_tools,
                    "diagnostics": {
                        "answer_rejections": [e["reason"] for e in events if e["event"] == "answer_rejected"],
                        "tool_errors": [
                            {"tool": e["tool"], "type": e["result"]["error"]["type"]}
                            for e in events
                            if e["event"] == "tool" and not e["result"].get("ok")
                        ],
                        "provider_errors": [
                            e["error_type"] for e in events if e["event"] == "provider_error"
                        ],
                        "handoff_confirmed": any(
                            e["event"] == "tool" and e.get("confirmed_by") for e in events
                        ),
                    },
                }
            )
            if progress:
                progress(
                    {
                        "completed": len(runs),
                        "total": len(cases) * repeats,
                        "scenario_id": case["id"],
                        "repetition": repetition + 1,
                        "passed": runs[-1]["passed"],
                        "failed_checks": [k for k, ok in checks.items() if not ok],
                    }
                )
    n = len(runs)
    report = {
        "mode": "offline-scripted" if config.provider.kind == "scripted" else "real-provider",
        "warning": "Offline fixtures test the harness, not LLM quality"
        if config.provider.kind == "scripted"
        else "Synthetic scenario evaluation; inspect traces and failed cases",
        "provider": config.provider.kind,
        "model": config.provider.model,
        "scenario_count": len(cases),
        "repeats": repeats,
        "run_count": n,
        "metrics": {
            "scenario_pass_rate": sum(r["passed"] for r in runs) / n if n else 0,
            "category_accuracy": sum(r["checks"]["category"] for r in runs) / n if n else 0,
            "handoff_accuracy": sum(r["checks"]["handoff"] for r in runs) / n if n else 0,
            "draft_ready_share": sum(r["result"]["status"] == "draft_ready" for r in runs) / n if n else 0,
            "median_seconds": statistics.median(r["seconds"] for r in runs) if n else 0,
            "total_tokens": sum(r["result"]["total_tokens"] for r in runs),
        },
        "failed_scenarios": [
            {
                "scenario_id": sid,
                "failed_runs": sum(not r["passed"] for r in runs if r["scenario_id"] == sid),
                "failed_checks": dict(
                    Counter(
                        key
                        for r in runs
                        if r["scenario_id"] == sid
                        for key, ok in r["checks"].items()
                        if not ok
                    )
                ),
            }
            for sid in dict.fromkeys(r["scenario_id"] for r in runs if not r["passed"])
        ],
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
