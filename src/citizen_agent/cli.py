from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import load_config
from .evaluation import evaluate, run_case
from .privacy import redact
from .providers import GigaChatProvider, ProviderError
from .storage import Store


def confirm_card(card: dict) -> bool:
    print("\nПодтвердите действие для текущего обращения:", file=sys.stderr)
    print(json.dumps(card, ensure_ascii=False, indent=2), file=sys.stderr)
    if not sys.stdin.isatty():
        print("Интерактивное подтверждение недоступно. Действие отменено.", file=sys.stderr)
        return False
    try:
        return input("Введите ПОДТВЕРЖДАЮ для записи в очередь: ").strip() == "ПОДТВЕРЖДАЮ"
    except (EOFError, KeyboardInterrupt):
        return False


def read_cases(root: Path, path: str | None = None) -> list[dict]:
    return json.loads((Path(path) if path else root / "data/scenarios.json").read_text(encoding="utf-8"))[
        "scenarios"
    ]


async def async_main(args) -> int:
    config, root = load_config(args.config)
    load_dotenv(root / ".env", override=False)
    if args.command == "doctor":
        state = {
            "provider": config.provider.kind,
            "model": config.provider.model,
            "credentials_present": bool(os.getenv(config.provider.credentials_env)),
            "ca_bundle_configured": bool(os.getenv(config.provider.ca_bundle_env)),
            "knowledge_exists": (root / config.knowledge_path).exists(),
            "routing_exists": (root / config.routing_path).exists(),
            "tls_verification": True,
        }
        if args.live:
            if config.provider.kind != "gigachat":
                print(
                    "doctor --live сейчас проверяет native GigaChat. Для альтернативы используйте run/evaluate.",
                    file=sys.stderr,
                )
                return 2
            provider = GigaChatProvider(config.provider)
            try:
                token = await provider._access_token()
                models = await provider._request(
                    "GET",
                    config.provider.base_url.rstrip("/") + "/models",
                    headers={"Authorization": "Bearer " + token},
                )
                state["available_models"] = [m["id"] for m in models.get("data", [])]
                state["configured_model_available"] = config.provider.model in state["available_models"]
            finally:
                await provider.aclose()
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if all(state[k] for k in ("knowledge_exists", "routing_exists")) else 1
    if args.command == "queue":
        print(
            json.dumps(Store(root / config.database_path).queue("course_demo"), ensure_ascii=False, indent=2)
        )
        return 0
    if args.command == "evaluate":
        cases = read_cases(root, args.scenarios)
        if args.scenario:
            cases = [c for c in cases if c["id"] == args.scenario]
        if not cases:
            raise ValueError("Набор сценариев пуст")
        if config.provider.kind != "scripted" and args.repeats < 3:
            raise ValueError("Для оценки реальной модели нужно не менее 3 повторов каждого сценария")
        report = await evaluate(config, root, cases, args.repeats, Path(args.output).resolve())
        print(json.dumps({k: v for k, v in report.items() if k != "runs"}, ensure_ascii=False, indent=2))
        return 0 if report["metrics"]["scenario_pass_rate"] == 1 else 1
    if args.command in {"run", "demo"}:
        if args.command == "demo":
            case_id = "timeout_persistent" if args.fault else "account_normal"
            case = next(c for c in read_cases(root) if c["id"] == case_id)
            text, faults, script = case["text"], case.get("faults"), case["script"]
        else:
            case = (
                next((c for c in read_cases(root) if c["id"] == args.scenario), None)
                if args.scenario
                else None
            )
            text = case["text"] if case else args.text
            faults, script = (case.get("faults"), case.get("script")) if case else (None, None)
        if not text:
            raise ValueError("Укажите --text или --scenario")
        if config.provider.kind == "scripted":
            print("OFFLINE FIXTURE: это проверка кода, не реальный вызов модели.", file=sys.stderr)
        result, events = await run_case(
            config,
            root,
            text,
            request_id=getattr(args, "request_id", None),
            confirm=confirm_card,
            script=script,
            faults=faults,
        )
        if getattr(args, "trace", False) or args.command == "demo":
            for event in events:
                if event["event"] in {"tool", "tool_retry", "provider_error", "answer_rejected"}:
                    print(json.dumps(event, ensure_ascii=False), file=sys.stderr)
        print(result.model_dump_json(indent=2))
        # Save stdout as JSON. A review outcome is valid behavior, not a crash.
        return 0
    return 2


def main():
    parser = argparse.ArgumentParser(description="ДЗ 1: агент обработки синтетических обращений граждан")
    parser.add_argument("--config", default="config/gigachat.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Проверить настройки без вывода ключей")
    doctor.add_argument("--live", action="store_true", help="Проверить OAuth и список моделей")
    run = commands.add_parser("run", help="Обработать синтетическое обращение")
    run.add_argument("--text")
    run.add_argument("--scenario")
    run.add_argument("--request-id")
    run.add_argument("--trace", action="store_true")
    commands.add_parser("queue", help="Показать локальную очередь результатов")
    demo = commands.add_parser("demo", help="Демо обычного сценария или ошибки")
    demo.add_argument("--fault", action="store_true")
    evaluation = commands.add_parser("evaluate", help="Оценить сценарии, повторяя каждый независимо")
    evaluation.add_argument("--scenarios")
    evaluation.add_argument("--scenario")
    evaluation.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    evaluation.add_argument("--output", default="runtime/evaluation/report.json")
    try:
        code = asyncio.run(async_main(parser.parse_args()))
    except (ProviderError, ValueError, OSError, sqlite3.Error) as exc:
        # Do not print exception repr, HTTP request headers or environment values.
        print("Ошибка настройки/запуска: " + redact(str(exc)), file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)
