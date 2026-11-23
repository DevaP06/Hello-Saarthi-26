"""The client IP comes from X-Forwarded-For only when our proxy (Caddy) vouches for it."""
from types import SimpleNamespace

from saarthi_backend.utils import rate_limit


def _req(headers: dict, peer: str = "10.0.0.9"):
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=peer))


def test_forwarded_ip_trusted_only_from_proxy(monkeypatch):
    ident = rate_limit.get_identifier_from_request
    monkeypatch.setattr(rate_limit, "_PROXY_SECRET", "s3cret")
    assert ident(_req({"x-forwarded-for": "1.2.3.4", "x-saarthi-proxy": "s3cret"})) == "1.2.3.4"
    # e.g. code in the sandbox calling the backend directly with a made-up IP
    assert ident(_req({"x-forwarded-for": "1.2.3.4"})) == "10.0.0.9"
    assert ident(_req({"x-forwarded-for": "1.2.3.4", "x-saarthi-proxy": "guess"})) == "10.0.0.9"

    monkeypatch.setattr(rate_limit, "_PROXY_SECRET", "")  # local dev, no proxy: unchanged behaviour
    assert ident(_req({"x-forwarded-for": "1.2.3.4"})) == "1.2.3.4"
