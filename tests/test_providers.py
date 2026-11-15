import asyncio
import json
import time

import httpx
import pytest

from citizen_agent.config import ProviderConfig
from citizen_agent.models import ToolCall, Turn
from citizen_agent.providers import CompatibleProvider, GigaChatProvider, ProviderError, assistant_message
from citizen_agent.tools import SPECS

SCHEMAS = [s.schema() for s in SPECS.values()]
USAGE = {"prompt_tokens": 50, "completion_tokens": 10}


def test_missing_proxy_dependency_is_configuration_error(monkeypatch):
    def missing_dependency(*args, **kwargs):
        raise ImportError("optional proxy dependency missing")

    monkeypatch.setattr(httpx, "AsyncClient", missing_dependency)
    with pytest.raises(ProviderError, match="requirements-dev.lock") as exc:
        GigaChatProvider(ProviderConfig())
    assert exc.value.kind == "configuration"


def test_native_gigachat_oauth_function_result_and_state(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "fake-test-authorization-only")
    requests = []

    def handler(request):
        requests.append(request)
        if "oauth" in str(request.url):
            assert request.headers["authorization"] == "Basic fake-test-authorization-only"
            assert request.headers["rquid"]
            assert b"scope=GIGACHAT_API_PERS" in request.content
            return httpx.Response(
                200, json={"access_token": "temporary-test-token", "expires_at": (time.time() + 1800) * 1000}
            )
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer temporary-test-token"
        assert "fake-test-authorization-only" not in request.content.decode()
        assert body["functions"] == SCHEMAS and "tools" not in body
        if len(requests) == 2:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "functions_state_id": "state-123",
                                "function_call": {
                                    "name": "get_routing_rules",
                                    "arguments": {"category": "account_access"},
                                },
                            },
                            "finish_reason": "function_call",
                        }
                    ],
                    "usage": USAGE,
                },
            )
        assistant = body["messages"][1]
        assert assistant["functions_state_id"] == "state-123"
        assert assistant["function_call"]["arguments"] == {"category": "account_access"}
        result = body["messages"][2]
        assert result["role"] == "function" and result["name"] == "get_routing_rules"
        assert "tool_call_id" not in result
        assert json.loads(result["content"]) == {"ok": True}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}], "usage": USAGE}
        )

    async def run():
        provider = GigaChatProvider(ProviderConfig(), transport=httpx.MockTransport(handler))
        try:
            messages = [{"role": "user", "content": "Вопрос"}]
            first = await provider.complete(messages, SCHEMAS)
            messages.extend(
                [
                    assistant_message(first),
                    {
                        "role": "tool",
                        "name": "get_routing_rules",
                        "tool_call_id": first.calls[0].id,
                        "content": '{"ok": true}',
                    },
                ]
            )
            second = await provider.complete(messages, SCHEMAS)
            assert first.usage.total == 60 and second.content == "done"
        finally:
            await provider.aclose()

    asyncio.run(run())
    assert len(requests) == 3  # token is cached, not fetched per step


def test_native_refreshes_expired_or_rejected_token(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "test-only")
    counter = {"auth": 0, "chat": 0}

    def handler(request):
        if "oauth" in str(request.url):
            counter["auth"] += 1
            return httpx.Response(
                200, json={"access_token": f"token-{counter['auth']}", "expires_at": time.time() + 1800}
            )
        counter["chat"] += 1
        if counter["chat"] == 1:
            return httpx.Response(401)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": USAGE})

    async def run():
        provider = GigaChatProvider(ProviderConfig(), transport=httpx.MockTransport(handler))
        try:
            assert (await provider.complete([{"role": "user", "content": "test"}], SCHEMAS)).content == "ok"
        finally:
            await provider.aclose()

    asyncio.run(run())
    assert counter == {"auth": 2, "chat": 2}


def test_compatible_adapter_round_trip_and_multiple_calls(monkeypatch):
    monkeypatch.setenv("COMPATIBLE_API_KEY", "local-placeholder")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert "functions" not in body
        assert body["tools"][0]["type"] == "function"
        if len(seen) > 1:
            assert "_state_id" not in body["messages"][1]
            assert isinstance(body["messages"][1]["tool_calls"][0]["function"]["arguments"], str)
            assert body["messages"][2]["role"] == "tool"
            assert body["messages"][2]["tool_call_id"] == "c1"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "get_routing_rules",
                                        "arguments": '{"category":"account_access"}',
                                    },
                                },
                                {
                                    "id": "c2",
                                    "function": {
                                        "name": "get_routing_rules",
                                        "arguments": '{"category":"tax_notice"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": USAGE,
            },
        )

    async def run():
        config = ProviderConfig(
            kind="openai_compatible",
            base_url="http://127.0.0.1:11434/v1",
            credentials_env="COMPATIBLE_API_KEY",
        )
        provider = CompatibleProvider(config, transport=httpx.MockTransport(handler))
        try:
            turn = await provider.complete([{"role": "user", "content": "hi"}], SCHEMAS)
            assert len(turn.calls) == 2 and turn.calls[0].arguments.startswith("{")
            history = [
                {"role": "user", "content": "hi"},
                assistant_message(
                    Turn(
                        calls=[
                            ToolCall(
                                id="c1", name="get_routing_rules", arguments={"category": "account_access"}
                            )
                        ],
                        state_id="native-state",
                    )
                ),
                {"role": "tool", "tool_call_id": "c1", "name": "get_routing_rules", "content": '{"ok":true}'},
            ]
            await provider.complete(history, SCHEMAS)
        finally:
            await provider.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "status,kind",
    [
        (401, "authentication"),
        (403, "forbidden"),
        (422, "request_rejected"),
        (429, "rate_limit"),
        (503, "service_unavailable"),
    ],
)
def test_http_failures_classified_and_bounded(monkeypatch, status, kind):
    monkeypatch.setenv("COMPATIBLE_API_KEY", "fake")
    count = 0

    def handler(_):
        nonlocal count
        count += 1
        return httpx.Response(status, text="private secret server diagnostic must not leak")

    async def run():
        provider = CompatibleProvider(
            ProviderConfig(kind="openai_compatible", credentials_env="COMPATIBLE_API_KEY", max_attempts=2),
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(ProviderError) as exc:
                await provider.complete([], SCHEMAS)
            assert exc.value.kind == kind
            assert f"HTTP {status}" in str(exc.value)
            assert "private secret" not in str(exc.value)
        finally:
            await provider.aclose()

    asyncio.run(run())
    assert count == (2 if status in {429, 503} else 1)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": []},
        {"choices": [{"message": {"content": 123}}]},
        {"choices": [{"message": []}]},
        {"choices": [{"message": {"content": "ok"}}], "usage": "broken"},
        {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": -1, "completion_tokens": 0}},
        {"choices": [{"message": {"content": "ok"}}]},
    ],
)
def test_bad_provider_payload_never_accepted(monkeypatch, payload):
    monkeypatch.setenv("COMPATIBLE_API_KEY", "fake")

    async def run():
        provider = CompatibleProvider(
            ProviderConfig(kind="openai_compatible", credentials_env="COMPATIBLE_API_KEY"),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
        )
        try:
            with pytest.raises(ProviderError):
                await provider.complete([], SCHEMAS)
        finally:
            await provider.aclose()

    asyncio.run(run())


def test_missing_credentials_fail_before_network(monkeypatch):
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)

    async def run():
        provider = GigaChatProvider(
            ProviderConfig(), transport=httpx.MockTransport(lambda _: pytest.fail("No request expected"))
        )
        try:
            with pytest.raises(ProviderError, match="GIGACHAT_CREDENTIALS"):
                await provider.complete([], SCHEMAS)
        finally:
            await provider.aclose()

    asyncio.run(run())
