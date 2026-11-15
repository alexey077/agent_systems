import argparse
import asyncio
import hashlib
import json
import socket
import ssl

import httpx
import pytest

from citizen_agent import cli
from citizen_agent.config import ProviderConfig
from citizen_agent.providers import GigaChatProvider, ProviderError, ssl_context

ROOT_FINGERPRINT = "d26d2d0231b7c39f92cc738512ba54103519e4405d68b5bd703e9788ca8ecf31"


def test_native_context_trusts_published_ca_without_disabling_verification(monkeypatch):
    monkeypatch.delenv("GIGACHAT_CA_BUNDLE", raising=False)
    # A Python install with an empty system store still gets public and GigaChat roots.
    monkeypatch.setattr(ssl, "create_default_context", lambda: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    context = ssl_context(ProviderConfig())
    roots = {hashlib.sha256(der).hexdigest() for der in context.get_ca_certs(binary_form=True)}
    assert ROOT_FINGERPRINT in roots
    assert len(roots) > 10
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    compatible = ssl_context(ProviderConfig(kind="openai_compatible"))
    assert ROOT_FINGERPRINT not in {
        hashlib.sha256(der).hexdigest() for der in compatible.get_ca_certs(binary_form=True)
    }


@pytest.mark.parametrize("contents", [None, "not a PEM certificate"])
def test_invalid_custom_ca_is_safe_configuration_error(tmp_path, monkeypatch, contents):
    pem = tmp_path / "custom.pem"
    if contents is not None:
        pem.write_text(contents)
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE", str(pem))
    with pytest.raises(ProviderError) as error:
        ssl_context(ProviderConfig())
    assert error.value.kind == "configuration"
    assert str(pem) not in str(error.value)
    assert "GIGACHAT_CA_BUNDLE" in str(error.value)


@pytest.mark.parametrize(
    "failure,kind,attempts",
    [
        ("tls", "tls", 1),
        ("dns", "dns", 2),
        ("timeout", "timeout", 2),
        ("network", "network", 2),
        ("proxy", "proxy", 2),
        ("tls_flat", "tls", 1),
    ],
)
def test_transport_failure_keeps_reason_and_never_exposes_secrets(monkeypatch, failure, kind, attempts):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "fake-authorization")
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        detail = "PRIVATE proxy user:password and Basic fake-authorization"
        if failure == "tls":
            try:
                raise ssl.SSLCertVerificationError(1, detail)
            except ssl.SSLCertVerificationError as cause:
                raise httpx.ConnectError(detail, request=request) from cause
        if failure == "tls_flat":
            raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] " + detail, request=request)
        if failure == "dns":
            try:
                raise socket.gaierror(-2, detail)
            except socket.gaierror as cause:
                raise httpx.ConnectError(detail, request=request) from cause
        errors = {"timeout": httpx.ConnectTimeout, "network": httpx.ConnectError, "proxy": httpx.ProxyError}
        raise errors[failure](detail, request=request)

    async def run():
        provider = GigaChatProvider(ProviderConfig(max_attempts=2), transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderError) as error:
                await provider._access_token()
            assert error.value.kind == kind
            assert "авторизация GigaChat" in str(error.value)
            for secret in ("PRIVATE", "user:password", "fake-authorization"):
                assert secret not in str(error.value)
        finally:
            await provider.aclose()

    asyncio.run(run())
    assert count == attempts


@pytest.mark.parametrize("stage", ["oauth", "models", "unavailable_model", "ok", "bad_models", "init"])
def test_doctor_keeps_failed_stage_and_returns_nonzero(monkeypatch, capsys, stage):
    closed = []

    class FakeProvider:
        def __init__(self, config):
            if stage == "init":
                raise ProviderError("configuration", "Не удалось загрузить TLS-сертификаты")

        async def _access_token(self):
            if stage == "oauth":
                raise ProviderError("tls", "Не удалось проверить сертификат")
            return "temporary-private-token"

        async def _request(self, *args, **kwargs):
            if stage == "models":
                raise ProviderError("service_unavailable", "Сетевой шлюз вернул HTTP 503")
            if stage == "bad_models":
                return {"data": [None]}
            return {"data": [{"id": "another-model" if stage == "unavailable_model" else "GigaChat-2-Pro"}]}

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(cli, "GigaChatProvider", FakeProvider)
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "private-test-key")
    args = argparse.Namespace(config="config/gigachat.toml", command="doctor", live=True)
    result = asyncio.run(cli.async_main(args))
    output = capsys.readouterr().out
    state = json.loads(output)
    assert state["credentials_present"] is True
    assert result == (0 if stage == "ok" else 2)
    assert state["oauth_ok"] == (stage not in {"oauth", "init"})
    assert state["models_ok"] == (stage in {"unavailable_model", "ok"})
    if stage in {"oauth", "init"}:
        assert state["live_stage"] == "oauth"
    elif stage in {"models", "bad_models"}:
        assert state["live_stage"] == "models"
    else:
        assert state["live_stage"] == "complete"
    assert closed == ([] if stage == "init" else [True])
    assert "private-test-key" not in output and "temporary-private-token" not in output
    if stage != "ok":
        assert "live_error" in state
