from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from .models import StrictModel


class ProviderConfig(StrictModel):
    kind: Literal["gigachat", "openai_compatible", "scripted"] = "gigachat"
    model: str = "GigaChat-2-Pro"
    base_url: str = "https://api.giga.chat/v1"
    auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    credentials_env: str = Field(default="GIGACHAT_CREDENTIALS", pattern=r"^[A-Z][A-Z0-9_]*$")
    scope_env: str = "GIGACHAT_SCOPE"
    scope: str = "GIGACHAT_API_PERS"
    ca_bundle_env: str = "GIGACHAT_CA_BUNDLE"
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    max_attempts: int = Field(default=3, ge=1, le=4)
    max_output_tokens: int = Field(default=1800, ge=128, le=8192)
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_response_bytes: int = Field(default=262144, ge=1024, le=1048576)
    prompt_price_per_million: float | None = Field(default=None, ge=0)
    completion_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def secure_urls(self):
        for url in (self.base_url, self.auth_url):
            parsed = urlparse(url)
            local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Provider URL must not contain credentials, query or fragment")
            if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
                raise ValueError("HTTPS is required except for a loopback model server")
        if (self.prompt_price_per_million is None) != (self.completion_price_per_million is None):
            raise ValueError("Specify both token prices or neither")
        return self


class RuntimeConfig(StrictModel):
    max_steps: int = Field(default=10, ge=1, le=30)
    max_tool_calls: int = Field(default=12, ge=1, le=40)
    max_identical_calls: int = Field(default=2, ge=1, le=3)
    max_total_tokens: int = Field(default=30000, ge=1000)
    max_context_chars: int = Field(default=60000, ge=2000)
    max_input_chars: int = Field(default=6000, ge=10, le=20000)
    max_result_chars: int = Field(default=9000, ge=1000, le=30000)
    max_elapsed_seconds: float = Field(default=180.0, gt=0)
    tool_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    read_attempts: int = Field(default=3, ge=1, le=3)
    retry_delay_seconds: float = Field(default=0.1, ge=0, le=2)


class AppConfig(StrictModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    knowledge_path: str = "data/knowledge.json"
    routing_path: str = "data/routing.json"
    database_path: str = "runtime/appeals.sqlite3"
    traces_path: str = "runtime/traces"


def load_config(path: str | Path) -> tuple[AppConfig, Path]:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    root = path.parent.parent if path.parent.name == "config" else path.parent
    return AppConfig.model_validate(raw), root
