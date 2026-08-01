import base64
import json

import pytest

from sigil.dpop import DPoPKey, ath_for


def _payload(jwt: str) -> dict:
    p = jwt.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(p + "=" * (-len(p) % 4)))
from sigil.errors import SigilAPIError
from sigil.mcp import (
    MCPToken,
    SigilTokenExchangeDeniedError,
    SigilTokenExchangeError,
    _MCPTokenCache,
)


def test_error_hierarchy_and_code():
    err = SigilTokenExchangeError("invalid_scope", "no overlap", status_code=400)
    assert isinstance(err, SigilAPIError)
    assert err.error_code == "invalid_scope"
    assert err.error_description == "no overlap"
    denied = SigilTokenExchangeDeniedError("access_denied", "revoked", status_code=403)
    assert isinstance(denied, SigilTokenExchangeError)


def test_error_description_is_bounded():
    err = SigilTokenExchangeError("server_error", "x" * 1000)
    assert len(err.error_description) == 256


def test_repr_never_leaks_the_token():
    tok = MCPToken(
        access_token="SECRET.JWT.VALUE",
        token_type="Bearer",
        scope=["read"],
        expires_in=300,
        expires_at=1000.0,
        resource="https://tool",
    )
    assert "SECRET" not in repr(tok)


def test_proof_for_requires_dpop_token():
    bearer = MCPToken("t", "Bearer", ["read"], 300, 1000.0, "https://tool")
    with pytest.raises(ValueError):
        bearer.proof_for("https://gw/mcp")
    dpop = MCPToken("t", "DPoP", ["read"], 300, 1000.0, "https://tool", _dpop=DPoPKey())
    proof = dpop.proof_for("https://gw/mcp", "GET")
    # ath (inside the encoded payload) binds the access token
    assert _payload(proof)["ath"] == ath_for("t")
    assert _payload(proof)["htm"] == "GET"


def test_cache_returns_fresh_and_re_mints_when_stale():
    cache = _MCPTokenCache()
    key = _MCPTokenCache.key("fp1", "https://tool", ["b", "a"], False)
    assert _MCPTokenCache.key("fp1", "https://tool", ["a", "b"], False) == key  # order-independent
    # a different biscuit fingerprint is a different key (no cross-agent token sharing)
    assert _MCPTokenCache.key("fp2", "https://tool", ["a", "b"], False) != key
    calls = {"n": 0}

    def mint() -> MCPToken:
        calls["n"] += 1
        return MCPToken("t", "Bearer", ["read"], 300, expires_at=1000.0, resource="https://tool")

    # now=700, leeway=30 → 1000-700=300 > 30 → fresh; second call is a cache hit
    cache.get_or_mint(key, now=700.0, leeway=30.0, mint=mint)
    cache.get_or_mint(key, now=700.0, leeway=30.0, mint=mint)
    assert calls["n"] == 1
    # now=990 → 1000-990=10 < 30 → stale → re-mint
    cache.get_or_mint(key, now=990.0, leeway=30.0, mint=mint)
    assert calls["n"] == 2


def test_cache_failed_mint_does_not_poison():
    cache = _MCPTokenCache()
    key = _MCPTokenCache.key("fp1", "https://tool", ["read"], False)
    stale = MCPToken("old", "Bearer", ["read"], 300, expires_at=1.0, resource="https://tool")
    cache._store[key] = stale

    def bad_mint() -> MCPToken:
        raise SigilTokenExchangeError("server_error", "boom")

    with pytest.raises(SigilTokenExchangeError):
        cache.get_or_mint(key, now=900.0, leeway=30.0, mint=bad_mint)
    assert cache._store[key] is stale  # the stale entry is untouched, not replaced with None


from unittest.mock import MagicMock, patch

from sigil.client import SigilClient
from sigil._context import _current_task
from sigil.errors import SigilTransportError


def _mock_response(status_code: int, body: dict, *, is_redirect: bool = False) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = body
    r.is_redirect = is_redirect  # MagicMock would otherwise auto-return a truthy attribute
    return r


def _client() -> SigilClient:
    return SigilClient(
        base_url="http://sigil-core:8120",
        internal_token="int-tok",
        tenant_id="ten-1",
        agent_id="agent-1",
        service_account_id="sa-1",
    )


_OK = {
    "access_token": "aaa.bbb.ccc",
    "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
    "token_type": "Bearer",
    "expires_in": 300,
    "scope": "read search",
}


def test_mcp_token_bearer_posts_form_without_internal_header():
    client = _client()
    with patch.object(client._session, "post", return_value=_mock_response(200, _OK)) as p:
        tok = client.mcp_token("https://tool.example/mcp", scope=["read", "search"], subject_token="BISCUIT")
    assert tok.token_type == "Bearer"
    assert tok.scope == ["read", "search"]
    assert tok.expires_in == 300
    url = p.call_args.args[0]
    assert url == "http://sigil-core:8120/oauth/token"
    data = p.call_args.kwargs["data"]
    assert data["grant_type"] == "urn:ietf:params:oauth:grant-type:token-exchange"
    assert data["subject_token_type"] == "urn:qwentrix:biscuit"
    assert data["subject_token"] == "BISCUIT"
    assert data["resource"] == "https://tool.example/mcp"
    assert data["scope"] == "read search"
    headers = p.call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"
    # Unauthenticated endpoint (RFC 8693) — none of the internal headers may leak.
    assert "X-Internal-Service-Token" not in headers
    assert "X-Tenant-ID" not in headers
    assert "X-Internal-Service-Account" not in headers
    assert "DPoP" not in headers
    # And redirects must not be followed (a 307/308 would resend the biscuit body elsewhere).
    assert p.call_args.kwargs["allow_redirects"] is False


def test_mcp_token_dpop_sends_proof_header():
    client = _client()
    with patch.object(client._session, "post", return_value=_mock_response(200, {**_OK, "token_type": "DPoP"})) as p:
        tok = client.mcp_token("https://tool.example/mcp", dpop=True, subject_token="BISCUIT")
    assert tok.token_type == "DPoP"
    assert "DPoP" in p.call_args.kwargs["headers"]
    assert tok.proof_for("https://gw.example/mcp").count(".") == 2


def test_mcp_token_uses_task_biscuit_when_no_override():
    client = _client()

    class _Task:
        _biscuit_token = "TASK-BISCUIT"

    reset = _current_task.set(_Task())
    try:
        with patch.object(client._session, "post", return_value=_mock_response(200, _OK)) as p:
            client.mcp_token("https://tool.example/mcp")
        assert p.call_args.kwargs["data"]["subject_token"] == "TASK-BISCUIT"
    finally:
        _current_task.reset(reset)


def test_mcp_token_raises_without_biscuit():
    client = _client()
    with pytest.raises(ValueError):
        client.mcp_token("https://tool.example/mcp")


def test_mcp_token_caches_second_call():
    client = _client()
    with patch.object(client._session, "post", return_value=_mock_response(200, _OK)) as p:
        client.mcp_token("https://tool.example/mcp", subject_token="B")
        client.mcp_token("https://tool.example/mcp", subject_token="B")
    assert p.call_count == 1


def test_mcp_token_maps_error_codes():
    client = _client()
    with patch.object(
        client._session, "post",
        return_value=_mock_response(400, {"error": "invalid_scope", "error_description": "no overlap"}),
    ):
        with pytest.raises(SigilTokenExchangeError) as ei:
            client.mcp_token("https://tool.example/mcp", subject_token="B")
    assert ei.value.error_code == "invalid_scope"

    with patch.object(
        client._session, "post",
        return_value=_mock_response(403, {"error": "access_denied", "error_description": "revoked"}),
    ):
        with pytest.raises(SigilTokenExchangeDeniedError):
            client.mcp_token("https://tool.example/mcp", subject_token="B")


def test_mcp_token_transport_error():
    client = _client()
    with patch.object(client._session, "post", side_effect=OSError("boom")):
        with pytest.raises(SigilTransportError):
            client.mcp_token("https://tool.example/mcp", subject_token="B")


def test_public_exports():
    import sigil

    assert sigil.MCPToken is MCPToken
    assert sigil.SigilTokenExchangeError is SigilTokenExchangeError
    assert sigil.SigilTokenExchangeDeniedError is SigilTokenExchangeDeniedError
    assert "MCPToken" in sigil.__all__
    assert "SigilTokenExchangeError" in sigil.__all__
    assert "SigilTokenExchangeDeniedError" in sigil.__all__


def test_mcp_token_expires_at_is_now_plus_expires_in():
    import time as _t
    client = _client()
    before = _t.time()
    with patch.object(client._session, "post", return_value=_mock_response(200, _OK)):
        tok = client.mcp_token("https://tool.example/mcp", subject_token="B")
    after = _t.time()
    assert before + 300 <= tok.expires_at <= after + 300 + 1


def test_mcp_token_refreshes_after_near_expiry():
    # A token minted with expires_in just below the leeway must trigger a second POST.
    client = SigilClient(
        base_url="http://sigil-core:8120", internal_token="int-tok",
        tenant_id="ten-1", agent_id="agent-1", service_account_id="sa-1",
        mcp_token_leeway=310.0,  # leeway > the 300s token lifetime → always "stale" → re-mint
    )
    with patch.object(client._session, "post", return_value=_mock_response(200, _OK)) as p:
        client.mcp_token("https://tool.example/mcp", subject_token="B")
        client.mcp_token("https://tool.example/mcp", subject_token="B")
    assert p.call_count == 2


def test_mcp_token_rejects_nonpositive_expires_in():
    client = _client()
    with patch.object(client._session, "post", return_value=_mock_response(200, {**_OK, "expires_in": 0})):
        with pytest.raises(SigilAPIError):
            client.mcp_token("https://tool.example/mcp", subject_token="B")


def test_mcp_token_different_biscuits_do_not_share_cache():
    client = _client()
    with patch.object(client._session, "post", return_value=_mock_response(200, _OK)) as p:
        client.mcp_token("https://tool.example/mcp", subject_token="BISCUIT-A")
        client.mcp_token("https://tool.example/mcp", subject_token="BISCUIT-B")
    assert p.call_count == 2  # different biscuits → different cache keys → two mints


def test_oauth_issuer_http_warning(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        SigilClient(
            base_url="https://sigil-core:8120", internal_token="int-tok",
            tenant_id="ten-1", agent_id="agent-1", service_account_id="sa-1",
            oauth_issuer="http://public.example:8080",
        )
    assert any("oauth_issuer" in r.message and "plaintext" in r.message for r in caplog.records)


def test_mcp_token_rejects_redirects():
    # A 307/308 would re-send the biscuit body to the redirect target; the SDK must refuse.
    client = _client()
    with patch.object(client._session, "post", return_value=_mock_response(307, {}, is_redirect=True)):
        with pytest.raises(SigilAPIError):
            client.mcp_token("https://tool.example/mcp", subject_token="B")


@pytest.mark.parametrize("code", ["invalid_grant", "invalid_target", "invalid_dpop_proof", "server_error"])
def test_mcp_token_error_codes_map_to_exchange_error(code):
    status = 500 if code == "server_error" else 400
    client = _client()
    with patch.object(
        client._session, "post",
        return_value=_mock_response(status, {"error": code, "error_description": "x"}),
    ):
        with pytest.raises(SigilTokenExchangeError) as ei:
            client.mcp_token("https://tool.example/mcp", subject_token="B")
    assert ei.value.error_code == code
    assert not isinstance(ei.value, SigilTokenExchangeDeniedError)  # only access_denied is terminal


def test_mcp_token_error_never_leaks_the_biscuit():
    client = _client()
    with patch.object(
        client._session, "post",
        return_value=_mock_response(400, {"error": "invalid_grant", "error_description": "SECRET-BISCUIT-VALUE"}),
    ):
        with pytest.raises(SigilTokenExchangeError) as ei:
            client.mcp_token("https://tool.example/mcp", subject_token="SECRET-BISCUIT-VALUE")
    assert "SECRET-BISCUIT-VALUE" not in str(ei.value)


def test_mcp_token_slots_blocks_vars_leak():
    from sigil.mcp import MCPToken as _MT
    tok = _MT("SECRET.JWT", "Bearer", ["read"], 300, 1000.0, "https://t")
    with pytest.raises(TypeError):
        vars(tok)  # slots=True → no __dict__ → cannot dump the token via vars()/generic loggers
