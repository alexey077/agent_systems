import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from citizen_agent.config import AppConfig, ProviderConfig, load_config
from citizen_agent.privacy import redact
from citizen_agent.trace import Trace

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,kind", [("gigachat", "gigachat"), ("ollama", "openai_compatible"), ("offline", "scripted")]
)
def test_provider_selected_by_config_without_agent_edits(name, kind):
    config, root = load_config(ROOT / f"config/{name}.toml")
    assert root == ROOT and config.provider.kind == kind


@pytest.mark.parametrize(
    "url", ["http://remote.example/v1", "https://user:password@example.com", "https://example.com?api_key=x"]
)
def test_config_rejects_insecure_or_embedded_credentials(url):
    with pytest.raises(ValidationError):
        ProviderConfig(base_url=url)


def test_redact_secret_and_contact_before_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "my-not-real-test-secret")
    value = {
        "content": "Телефон +7 (999) 123-45-67, a@b.example, ИНН 123456789012 и my-not-real-test-secret",
        "access_token": "a-token-value",
        "total_tokens": 123,
    }
    trace = Trace(tmp_path, "test")
    trace.emit("model", **value)
    saved = trace.path.read_text()
    for secret in ("my-not-real-test-secret", "123456789012", "a@b.example", "a-token-value", "999"):
        assert secret not in saved
    assert json.loads(saved)["total_tokens"] == 123
    assert redact("Bearer abcDEF123==") == "[AUTH]"


def test_unknown_config_fields_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"api_key": "do-not-hardcode"})
