from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
import time
import uuid
from collections import deque
from importlib.resources import files
from pathlib import Path
from typing import Protocol

import certifi
import httpx

from .config import ProviderConfig
from .models import ToolCall, Turn, Usage
from .trace import Trace


class ProviderError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class Provider(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict]) -> Turn: ...
    async def aclose(self): ...


def ssl_context(config: ProviderConfig) -> ssl.SSLContext:
    """Keep system trust, add public WebPKI roots and GigaChat's published CA locally."""
    try:
        context = ssl.create_default_context()
        context.load_verify_locations(cafile=certifi.where())
        if config.kind == "gigachat":
            pem = files("citizen_agent").joinpath("certs/russian_trusted_root_ca.pem").read_text("ascii")
            context.load_verify_locations(cadata=pem)
        ca_bundle = os.getenv(config.ca_bundle_env, "").strip()
        if ca_bundle:
            context.load_verify_locations(cafile=str(Path(ca_bundle).expanduser()))
        return context
    except (OSError, ValueError) as exc:
        raise ProviderError(
            "configuration",
            "Не удалось загрузить доверенные TLS-сертификаты. "
            f"Проверьте PEM-файл в {config.ca_bundle_env} и повторите установку проекта.",
        ) from exc


def transport_failure(exc: httpx.HTTPError, ca_bundle_env: str) -> tuple[str, str, bool]:
    """Classify nested transport errors without exposing URLs, proxy credentials or headers."""
    chain, seen, pending = [], set(), [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)
    if any(
        isinstance(e, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e).upper()
        for e in chain
    ):
        return (
            "tls",
            "Не удалось проверить TLS-сертификат сервера. "
            "Проверьте системную дату. Для дополнительных сертификатов или корпоративного прокси "
            f"задайте доверенный PEM в {ca_bundle_env}.",
            False,
        )
    if any(isinstance(e, ssl.SSLError) for e in chain):
        return "tls", "Не удалось установить TLS-соединение. Проверьте сертификаты и настройки сети.", False
    if isinstance(exc, httpx.ProxyError):
        return "proxy", "Ошибка прокси. Проверьте HTTPS_PROXY, HTTP_PROXY и ALL_PROXY.", True
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", "Истёк таймаут подключения или ожидания ответа. Проверьте сеть и VPN/прокси.", True
    if any(isinstance(e, socket.gaierror) for e in chain):
        return "dns", "Не удалось найти адрес сервера (DNS). Проверьте подключение и настройки DNS/VPN.", True
    return (
        "network",
        "Не удалось соединиться с сервером API. Проверьте сеть, VPN/прокси и доступ к API.",
        True,
    )


def assistant_message(turn: Turn) -> dict:
    result = {"role": "assistant", "content": turn.content}
    if turn.calls:
        result["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in turn.calls
        ]
    if turn.state_id:
        result["_state_id"] = turn.state_id
    return result


def usage_of(data: dict) -> Usage:
    usage = data.get("usage") or {}
    if not isinstance(usage, dict) or not {"prompt_tokens", "completion_tokens"} <= usage.keys():
        raise ProviderError("invalid_response", "API не вернул usage для контроля бюджета")
    return Usage(prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"])


class HTTPProvider:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        trace: Trace | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config, self.trace = config, trace
        context = ssl_context(config)
        try:
            self.client = httpx.AsyncClient(
                timeout=config.timeout_seconds, verify=context, transport=transport, follow_redirects=False
            )
        except ImportError as exc:
            raise ProviderError(
                "configuration", "Не установлена зависимость транспорта. Установите requirements-dev.lock"
            ) from exc

    def secret(self) -> str:
        value = os.getenv(self.config.credentials_env, "").strip()
        if not value:
            raise ProviderError(
                "configuration", f"Задайте {self.config.credentials_env} в локальном окружении"
            )
        return value

    async def _request(self, method: str, url: str, **kwargs) -> dict:
        stage = "авторизация GigaChat" if url == self.config.auth_url else "API модели"
        for attempt in range(1, self.config.max_attempts + 1):
            retryable = False
            kind = "network"
            message = "Не удалось соединиться с сервером API"
            try:
                async with self.client.stream(method, url, **kwargs) as response:
                    if response.status_code == 401:
                        raise ProviderError(
                            "authentication",
                            f"{stage}: HTTP 401, авторизация отклонена. Проверьте Authorization Key и scope.",
                        )
                    if response.status_code == 403:
                        raise ProviderError(
                            "forbidden", f"{stage}: HTTP 403. Проверьте доступ к API, scope и баланс."
                        )
                    if response.status_code == 429 or response.status_code >= 500:
                        retryable = True
                        kind = "rate_limit" if response.status_code == 429 else "service_unavailable"
                        message = (
                            "Превышен лимит запросов (HTTP 429). Повторите позже."
                            if response.status_code == 429
                            else f"Сервис или сетевой шлюз вернул HTTP {response.status_code}. "
                            "Проверьте доступность API из вашей сети и повторите позже."
                        )
                    elif response.status_code >= 300:
                        raise ProviderError(
                            "request_rejected", f"API отклонил запрос, HTTP {response.status_code}"
                        )
                    else:
                        raw = bytearray()
                        async for chunk in response.aiter_bytes():
                            raw.extend(chunk)
                            if len(raw) > self.config.max_response_bytes:
                                raise ProviderError("invalid_response", "Превышен лимит размера ответа API")
                        try:
                            payload = json.loads(raw)
                            if not isinstance(payload, dict):
                                raise ValueError()
                            return payload
                        except (ValueError, UnicodeError) as exc:
                            raise ProviderError("invalid_response", "API вернул некорректный JSON") from exc
            except httpx.HTTPError as exc:
                kind, message, retryable = transport_failure(exc, self.config.ca_bundle_env)
                if not retryable:
                    raise ProviderError(kind, f"{stage}: {message}") from exc
            if retryable:
                if self.trace:
                    self.trace.emit("provider_retry", attempt=attempt, error_type=kind)
                if attempt == self.config.max_attempts:
                    raise ProviderError(kind, f"{stage}: {message} Выполнено попыток: {attempt}.")
                await asyncio.sleep(min(0.25 * 2 ** (attempt - 1), 2))
        raise ProviderError("internal", "Не удалось выполнить запрос")

    async def aclose(self):
        await self.client.aclose()


class GigaChatProvider(HTTPProvider):
    """Native API adapter: functions, role=function and preserved functions_state_id."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token: str | None = None
        self._expires_at = 0.0

    async def _access_token(self) -> str:
        if self._token and time.time() < self._expires_at - 30:
            return self._token
        payload = await self._request(
            "POST",
            self.config.auth_url,
            headers={
                "Authorization": "Basic " + self.secret(),
                "RqUID": str(uuid.uuid4()),
                "Accept": "application/json",
            },
            data={"scope": os.getenv(self.config.scope_env, self.config.scope)},
        )
        try:
            token, expires = payload["access_token"], float(payload["expires_at"])
            if not isinstance(token, str) or not token:
                raise ValueError()
            self._token = token
            self._expires_at = expires / 1000 if expires > 1e11 else expires
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError("invalid_response", "OAuth вернул некорректный токен") from exc
        return self._token

    @staticmethod
    def translate(messages: list[dict]) -> list[dict]:
        native = []
        for msg in messages:
            item = {"role": msg["role"], "content": msg.get("content") or ""}
            if msg["role"] == "tool":
                item.update(role="function", name=msg["name"])
            if msg.get("tool_calls"):
                if len(msg["tool_calls"]) != 1:
                    raise ProviderError("protocol", "Native GigaChat v1 ожидает последовательные вызовы")
                call = msg["tool_calls"][0]["function"]
                args = call["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError as exc:
                        raise ProviderError("protocol", "Невалидные аргументы в истории") from exc
                item["function_call"] = {"name": call["name"], "arguments": args}
            if msg.get("_state_id"):
                item["functions_state_id"] = msg["_state_id"]
            native.append(item)
        return native

    async def complete(self, messages: list[dict], tools: list[dict]) -> Turn:
        body = {
            "model": self.config.model,
            "messages": self.translate(messages),
            "functions": tools,
            "function_call": "auto",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "stream": False,
        }
        data = None
        for refresh in range(2):
            token = await self._access_token()
            try:
                data = await self._request(
                    "POST",
                    self.config.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + token},
                    json=body,
                )
                break
            except ProviderError as exc:
                if exc.kind != "authentication" or refresh:
                    raise
                self._token, self._expires_at = None, 0
        try:
            choice = data["choices"][0]
            msg = choice["message"]
            fn = msg.get("function_call")
            calls = (
                [ToolCall(id="giga_" + uuid.uuid4().hex[:12], name=fn["name"], arguments=fn["arguments"])]
                if fn
                else []
            )
            return Turn(
                content=msg.get("content") or "",
                calls=calls,
                usage=usage_of(data),
                state_id=msg.get("functions_state_id"),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError("invalid_response", "Неполный ответ GigaChat API") from exc


class CompatibleProvider(HTTPProvider):
    """Chat Completions tool-calling adapter, e.g. a local Ollama /v1 server."""

    async def complete(self, messages: list[dict], tools: list[dict]) -> Turn:
        wire = []
        for msg in messages:
            item = {k: v for k, v in msg.items() if not k.startswith("_")}
            if item.get("tool_calls"):
                item = dict(item)
                item["tool_calls"] = []
                for c in msg["tool_calls"]:
                    args = c["function"]["arguments"]
                    item["tool_calls"].append(
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["function"]["name"],
                                "arguments": args
                                if isinstance(args, str)
                                else json.dumps(args, ensure_ascii=False),
                            },
                        }
                    )
            wire.append(item)
        body = {
            "model": self.config.model,
            "messages": wire,
            "tools": [{"type": "function", "function": t} for t in tools],
            "tool_choice": "auto",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "stream": False,
        }
        # A local Ollama server accepts a placeholder key. It still lives outside model messages.
        data = await self._request(
            "POST",
            self.config.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + self.secret()},
            json=body,
        )
        try:
            choice = data["choices"][0]
            msg = choice["message"]
            calls = [
                ToolCall(id=c["id"], name=c["function"]["name"], arguments=c["function"]["arguments"])
                for c in msg.get("tool_calls", [])
            ]
            return Turn(
                content=msg.get("content") or "",
                calls=calls,
                usage=usage_of(data),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError("invalid_response", "Неполный ответ совместимого API") from exc


class ScriptedProvider:
    """Explicit offline fixture runner. It is NOT a model or a measured LLM result."""

    def __init__(self, turns: list[Turn | dict]):
        self.turns = deque(Turn.model_validate(t) if isinstance(t, dict) else t for t in turns)
        self.requests: list[dict] = []

    async def complete(self, messages: list[dict], tools: list[dict]) -> Turn:
        self.requests.append({"messages": list(messages), "tools": tools})
        if not self.turns:
            raise ProviderError("script_exhausted", "Offline-сценарий закончился")
        return self.turns.popleft()

    async def aclose(self):
        pass


def make_provider(config: ProviderConfig, *, trace: Trace, script: list | None = None) -> Provider:
    if config.kind == "gigachat":
        return GigaChatProvider(config, trace=trace)
    if config.kind == "openai_compatible":
        return CompatibleProvider(config, trace=trace)
    if script is None:
        raise ProviderError("configuration", "scripted требует явного тестового сценария")
    return ScriptedProvider(script)
