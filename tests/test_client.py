"""Unit tests for sigil.client — HTTP transport to sigil-core.

Uses unittest.mock to stub requests.Session.post — no live server required.
All header assertions use the exact names required by the server middleware.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from datetime import datetime, timedelta, timezone

from sigil.client import _MAX_BATCH_SIZE, SigilClient
from sigil.errors import CredentialRotatedError, SigilAPIError, SigilTransportError

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _mock_response(status_code: int, body: dict[str, Any]) -> MagicMock:
    """Build a mock requests.Response with the given status and JSON body."""
    r: MagicMock = MagicMock()
    r.status_code = status_code
    r.json.return_value = body
    return r


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> SigilClient:
    """A configured SigilClient pointing at a non-existent test server."""
    return SigilClient(
        base_url="http://sigil-test:8120",
        internal_token="int-tok-secret",
        service_account="test-sa",
        tenant_id="ten-uuid-1234",
        agent_id="agent-uuid-5678",
        service_account_id="sa-uuid-abcd",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Constructor validation
# ──────────────────────────────────────────────────────────────────────────────


class TestClientConstructor:
    def test_requires_internal_token(self) -> None:
        with pytest.raises(ValueError, match="internal_token"):
            SigilClient(internal_token="")

    def test_rejects_invalid_fail_mode(self) -> None:
        with pytest.raises(ValueError, match="fail_mode"):
            SigilClient(internal_token="tok", fail_mode="maybe")

    def test_accepts_open_fail_mode(self) -> None:
        c = SigilClient(internal_token="tok", fail_mode="open")
        assert c.fail_mode == "open"

    def test_default_base_url(self) -> None:
        c = SigilClient(internal_token="tok")
        assert c.base_url == "http://sigil-core:8120"

    def test_trailing_slash_stripped(self) -> None:
        c = SigilClient(internal_token="tok", base_url="http://host:8120/")
        assert c.base_url == "http://host:8120"

    def test_default_service_account(self) -> None:
        c = SigilClient(internal_token="tok")
        assert c.service_account == "sigil-python-sdk"

    def test_default_fail_mode_is_closed(self) -> None:
        c = SigilClient(internal_token="tok")
        assert c.fail_mode == "closed"

    def test_session_created(self) -> None:
        import requests

        c = SigilClient(internal_token="tok")
        assert isinstance(c._session, requests.Session)


# ──────────────────────────────────────────────────────────────────────────────
# issue_token
# ──────────────────────────────────────────────────────────────────────────────

_ISSUE_RESP: dict[str, Any] = {
    "grant_id": "grant-001",
    "biscuit_token": "tok.sig",
    "revocation_id": "rev-001",
    "expires_at": "2026-12-31T00:00:00Z",
}


class TestIssueToken:
    def test_posts_to_correct_url(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)
        ) as mock_post:
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
            url: str = mock_post.call_args.args[0]
        assert url == "http://sigil-test:8120/internal/v1/sigil/tokens/issue"

    def test_sends_internal_service_token_header(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)
        ) as mock_post:
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
            headers: dict[str, str] = mock_post.call_args.kwargs["headers"]
        assert headers["X-Internal-Service-Token"] == "int-tok-secret"

    def test_sends_internal_service_account_header(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)
        ) as mock_post:
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
            headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Internal-Service-Account"] == "test-sa"

    def test_sends_tenant_id_header(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)
        ) as mock_post:
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
            headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Tenant-ID"] == "ten-uuid-1234"

    def test_does_not_send_agent_id_header(self, client: SigilClient) -> None:
        """issue_token uses /tokens route (not rate-limited toolgate) — no X-Sigil-Agent-ID."""
        with patch.object(
            client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)
        ) as mock_post:
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
            headers = mock_post.call_args.kwargs["headers"]
        assert "X-Sigil-Agent-ID" not in headers

    def test_request_body_fields(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)
        ) as mock_post:
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
            body: dict[str, Any] = mock_post.call_args.kwargs["json"]
        assert body["agent_id"] == "agent-uuid"
        assert body["service_account_id"] == "sa-uuid"
        assert body["task_id"] == "task-uuid"
        assert body["tool_allowlist"] == ["zep.search"]
        assert body["ttl_seconds"] == 3600

    def test_attenuation_root_included_when_provided(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)
        ) as mock_post:
            client.issue_token(
                "agent-uuid",
                "sa-uuid",
                "task-uuid",
                ["zep.search"],
                3600,
                attenuation_root="parent.token",
            )
            body = mock_post.call_args.kwargs["json"]
        assert body["attenuation_root"] == "parent.token"

    def test_attenuation_root_omitted_when_none(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)
        ) as mock_post:
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
            body = mock_post.call_args.kwargs["json"]
        assert "attenuation_root" not in body

    def test_returns_parsed_response(self, client: SigilClient) -> None:
        with patch.object(client._session, "post", return_value=_mock_response(201, _ISSUE_RESP)):
            result = client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
        assert result["grant_id"] == "grant-001"
        assert result["biscuit_token"] == "tok.sig"

    def test_raises_api_error_on_non_201(self, client: SigilClient) -> None:
        with (
            patch.object(
                client._session,
                "post",
                return_value=_mock_response(400, {"error": "bad"}),
            ),
            pytest.raises(SigilAPIError) as exc_info,
        ):
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
        assert exc_info.value.status_code == 400

    def test_raises_transport_error_on_network_failure(self, client: SigilClient) -> None:
        with (
            patch.object(client._session, "post", side_effect=ConnectionError("refused")),
            pytest.raises(SigilTransportError) as exc_info,
        ):
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)
        assert exc_info.value.method == "POST"
        assert "tokens/issue" in exc_info.value.url

    def test_raises_transport_error_on_timeout(self, client: SigilClient) -> None:
        import requests as req_lib

        with (
            patch.object(client._session, "post", side_effect=req_lib.Timeout("timed out")),
            pytest.raises(SigilTransportError),
        ):
            client.issue_token("agent-uuid", "sa-uuid", "task-uuid", ["zep.search"], 3600)


# ──────────────────────────────────────────────────────────────────────────────
# preflight
# ──────────────────────────────────────────────────────────────────────────────

_PREFLIGHT_ALLOW: dict[str, Any] = {"verdict": "allow"}
_PREFLIGHT_DENY: dict[str, Any] = {
    "verdict": "deny",
    "denied_reason": "tool_not_in_scope",
}


class TestPreflight:
    def test_posts_to_correct_url(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_ALLOW)
        ) as mock_post:
            client.preflight("tok", "zep", "search", "hash123")
            url: str = mock_post.call_args.args[0]
        assert url == "http://sigil-test:8120/internal/v1/sigil/toolgate/preflight"

    def test_sends_x_sigil_agent_id_header(self, client: SigilClient) -> None:
        """X-Sigil-Agent-ID is REQUIRED by the rate-limited toolgate group."""
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_ALLOW)
        ) as mock_post:
            client.preflight("tok", "zep", "search", "hash123")
            headers: dict[str, str] = mock_post.call_args.kwargs["headers"]
        assert headers["X-Sigil-Agent-ID"] == "agent-uuid-5678"

    def test_sends_required_base_headers(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_ALLOW)
        ) as mock_post:
            client.preflight("tok", "zep", "search", "hash123")
            headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Internal-Service-Token"] == "int-tok-secret"
        assert headers["X-Internal-Service-Account"] == "test-sa"
        assert headers["X-Tenant-ID"] == "ten-uuid-1234"

    def test_request_body_fields(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_ALLOW)
        ) as mock_post:
            client.preflight("my.token", "zep", "search", "sha256hex")
            body: dict[str, Any] = mock_post.call_args.kwargs["json"]
        assert body["token"] == "my.token"
        assert body["tool_namespace"] == "zep"
        assert body["tool_name"] == "search"
        assert body["args_hash"] == "sha256hex"

    def test_per_call_agent_id_included_in_body(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_ALLOW)
        ) as mock_post:
            client.preflight("tok", "zep", "search", "h", agent_id="per-call-agent")
            body = mock_post.call_args.kwargs["json"]
        assert body["agent_id"] == "per-call-agent"

    def test_task_id_included_when_provided(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_ALLOW)
        ) as mock_post:
            client.preflight("tok", "zep", "search", "h", task_id="task-abc")
            body = mock_post.call_args.kwargs["json"]
        assert body["task_id"] == "task-abc"

    def test_task_id_omitted_when_none(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_ALLOW)
        ) as mock_post:
            client.preflight("tok", "zep", "search", "h")
            body = mock_post.call_args.kwargs["json"]
        assert "task_id" not in body

    def test_returns_verdict_dict(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_ALLOW)
        ):
            result = client.preflight("tok", "zep", "search", "h")
        assert result["verdict"] == "allow"

    def test_returns_deny_with_reason(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(200, _PREFLIGHT_DENY)
        ):
            result = client.preflight("tok", "zep", "search", "h")
        assert result["verdict"] == "deny"
        assert result["denied_reason"] == "tool_not_in_scope"

    def test_raises_transport_error_on_network_failure(self, client: SigilClient) -> None:
        with (
            patch.object(client._session, "post", side_effect=ConnectionError("refused")),
            pytest.raises(SigilTransportError) as exc_info,
        ):
            client.preflight("tok", "zep", "search", "h")
        assert "preflight" in exc_info.value.url

    def test_raises_api_error_on_non_200(self, client: SigilClient) -> None:
        with (
            patch.object(
                client._session,
                "post",
                return_value=_mock_response(503, {"error": "unavailable"}),
            ),
            pytest.raises(SigilAPIError) as exc_info,
        ):
            client.preflight("tok", "zep", "search", "h")
        assert exc_info.value.status_code == 503


# ──────────────────────────────────────────────────────────────────────────────
# log_batch
# ──────────────────────────────────────────────────────────────────────────────

_LOG_RESP: dict[str, Any] = {"accepted": 2}

_SAMPLE_EVENT: dict[str, Any] = {
    "agent_id": "agent-uuid-5678",
    "task_id": "task-uuid-0001",
    "tool_name": "search",
    "tool_namespace": "zep",
    "args_hash": "abc123",
    "outcome": "allowed",
    "risk_tier": "low",
}


class TestLogBatch:
    def test_rejects_over_max_events_before_request(self, client: SigilClient) -> None:
        events = [_SAMPLE_EVENT.copy() for _ in range(_MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValueError, match=str(_MAX_BATCH_SIZE)):
            client.log_batch(events)

    def test_max_events_exactly_accepted(self, client: SigilClient) -> None:
        events = [_SAMPLE_EVENT.copy() for _ in range(_MAX_BATCH_SIZE)]
        resp_body = {"accepted": _MAX_BATCH_SIZE}
        with patch.object(client._session, "post", return_value=_mock_response(202, resp_body)):
            result = client.log_batch(events)
        assert result["accepted"] == _MAX_BATCH_SIZE

    def test_posts_to_correct_url(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(202, _LOG_RESP)
        ) as mock_post:
            client.log_batch([_SAMPLE_EVENT, _SAMPLE_EVENT])
            url: str = mock_post.call_args.args[0]
        assert url == "http://sigil-test:8120/internal/v1/sigil/toolgate/log-batch"

    def test_sends_x_sigil_agent_id_header(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(202, _LOG_RESP)
        ) as mock_post:
            client.log_batch([_SAMPLE_EVENT])
            headers: dict[str, str] = mock_post.call_args.kwargs["headers"]
        assert headers["X-Sigil-Agent-ID"] == "agent-uuid-5678"

    def test_sends_required_base_headers(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(202, _LOG_RESP)
        ) as mock_post:
            client.log_batch([_SAMPLE_EVENT])
            headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Internal-Service-Token"] == "int-tok-secret"
        assert headers["X-Internal-Service-Account"] == "test-sa"
        assert headers["X-Tenant-ID"] == "ten-uuid-1234"

    def test_body_wraps_events_under_events_key(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(202, _LOG_RESP)
        ) as mock_post:
            client.log_batch([_SAMPLE_EVENT, _SAMPLE_EVENT])
            body: dict[str, Any] = mock_post.call_args.kwargs["json"]
        assert "events" in body
        assert len(body["events"]) == 2

    def test_returns_accepted_count(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "post", return_value=_mock_response(202, {"accepted": 3})
        ):
            result = client.log_batch([_SAMPLE_EVENT, _SAMPLE_EVENT, _SAMPLE_EVENT])
        assert result["accepted"] == 3

    def test_raises_transport_error_on_network_failure(self, client: SigilClient) -> None:
        with (
            patch.object(client._session, "post", side_effect=ConnectionError("refused")),
            pytest.raises(SigilTransportError) as exc_info,
        ):
            client.log_batch([_SAMPLE_EVENT])
        assert "log-batch" in exc_info.value.url

    def test_raises_api_error_on_non_202(self, client: SigilClient) -> None:
        with (
            patch.object(
                client._session,
                "post",
                return_value=_mock_response(500, {"error": "internal"}),
            ),
            pytest.raises(SigilAPIError) as exc_info,
        ):
            client.log_batch([_SAMPLE_EVENT])
        assert exc_info.value.status_code == 500

    def test_empty_events_list_not_rejected_by_client(self, client: SigilClient) -> None:
        """Client does not reject empty list — server enforces that constraint."""
        with patch.object(
            client._session, "post", return_value=_mock_response(202, {"accepted": 0})
        ):
            result = client.log_batch([])
        assert result["accepted"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# SigilTaskContext (Pass 2 — new task() signature)
# ──────────────────────────────────────────────────────────────────────────────


class TestSigilTaskContext:
    def test_task_returns_context(self, client: SigilClient) -> None:
        from sigil.client import SigilTaskContext

        ctx = client.task(["zep.search"], ttl_seconds=600)
        assert isinstance(ctx, SigilTaskContext)

    def test_task_id_none_before_enter(self, client: SigilClient) -> None:
        ctx = client.task(["zep.search"])
        assert ctx.task_id is None

    def test_task_enter_calls_issue_token(self, client: SigilClient) -> None:
        """__enter__ should call issue_token and set task_id."""
        import base64
        import json
        from unittest.mock import patch

        from nacl.signing import SigningKey

        def b64(b: bytes) -> str:
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

        sk = SigningKey.generate()
        vk_bytes = bytes(sk.verify_key)
        kid = "test-kid"

        future_ts = "2099-01-01T00:00:00Z"
        payload = {
            "v": 1,
            "blocks": [
                {
                    "facts": [
                        'tool("zep.search")',
                        'tenant("ten-uuid-1234")',
                        'agent("agent-uuid-5678")',
                        'task("task-from-server")',
                    ],
                    "checks": [f'check if time($t), $t < "{future_ts}"'],
                    "rid": "rid-0",
                    "idx": 0,
                }
            ],
            "kid": kid,
        }
        payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
        sig_bytes = sk.sign(payload_bytes).signature
        biscuit_token = f"{b64(payload_bytes)}.{b64(sig_bytes)}"

        issue_resp = {
            "grant_id": "g-001",
            "biscuit_token": biscuit_token,
            "revocation_id": "rev-001",
            "expires_at": "2099-01-01T00:00:00Z",
        }

        client_with_key = SigilClient(
            base_url="http://sigil-test:8120",
            internal_token="int-tok-secret",
            tenant_id="ten-uuid-1234",
            agent_id="agent-uuid-5678",
            biscuit_keyring={kid: vk_bytes},
        )
        try:
            with (
                patch.object(
                    client_with_key._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                client_with_key.task(["zep.search"]) as task,
            ):
                assert task.task_id is not None
                assert len(task.task_id) == 36  # UUID format
        finally:
            client_with_key.close()


# ──────────────────────────────────────────────────────────────────────────────
# ENT-81/SG-4 — approval_status client method + config validation
# ──────────────────────────────────────────────────────────────────────────────


class TestApprovalStatus:
    def test_returns_status_from_get(self, client: SigilClient) -> None:
        with patch.object(
            client._session, "get", return_value=_mock_response(200, {"status": "approved"})
        ) as g:
            assert client.approval_status("ap-1") == "approved"
        # GET the tenant-scoped status path (tenant is derived server-side).
        assert g.call_args.args[0].endswith("/internal/v1/sigil/toolgate/approval/ap-1")

    def test_missing_status_raises_api_error(self, client: SigilClient) -> None:
        # A missing/empty status must fail fast so the poll denies rather than stalling.
        with (
            patch.object(client._session, "get", return_value=_mock_response(200, {})),
            pytest.raises(SigilAPIError),
        ):
            client.approval_status("ap-1")

    def test_non_200_raises_api_error(self, client: SigilClient) -> None:
        with (
            patch.object(client._session, "get", return_value=_mock_response(404, {})),
            pytest.raises(SigilAPIError),
        ):
            client.approval_status("nope")

    def test_transport_error_raises(self, client: SigilClient) -> None:
        with (
            patch.object(client._session, "get", side_effect=ConnectionError("refused")),
            pytest.raises(SigilTransportError),
        ):
            client.approval_status("ap-1")


class TestRedeemApproval:
    def test_returns_grant_from_post(self, client: SigilClient) -> None:
        body = {"one_shot_token": "os", "revocation_id": "rev-1", "tool_name": "ns.tool"}
        with patch.object(
            client._session, "post", return_value=_mock_response(200, body)
        ) as p:
            out = client.redeem_approval("ap-1", "parent-tok")
        assert out["revocation_id"] == "rev-1"
        assert p.call_args.args[0].endswith("/internal/v1/sigil/toolgate/approval/ap-1/redeem")
        assert p.call_args.kwargs["json"] == {"token": "parent-tok"}

    def test_409_raises_api_error_with_status(self, client: SigilClient) -> None:
        with (
            patch.object(client._session, "post", return_value=_mock_response(409, {})),
            pytest.raises(SigilAPIError) as exc,
        ):
            client.redeem_approval("ap-1", "tok")
        assert exc.value.status_code == 409

    def test_missing_revocation_id_raises(self, client: SigilClient) -> None:
        with (
            patch.object(
                client._session, "post", return_value=_mock_response(200, {"one_shot_token": "os"})
            ),
            pytest.raises(SigilAPIError),
        ):
            client.redeem_approval("ap-1", "tok")

    def test_transport_error_raises(self, client: SigilClient) -> None:
        with (
            patch.object(client._session, "post", side_effect=ConnectionError("refused")),
            pytest.raises(SigilTransportError),
        ):
            client.redeem_approval("ap-1", "tok")


class TestApprovalConfigValidation:
    def test_rejects_non_finite_timeout(self) -> None:
        for bad in (float("nan"), float("inf"), 0.0, -1.0):
            with pytest.raises(ValueError, match="approval_timeout"):
                SigilClient(internal_token="tok", approval_timeout=bad)

    def test_rejects_non_finite_poll_interval(self) -> None:
        for bad in (float("nan"), float("inf"), 0.0):
            with pytest.raises(ValueError, match="approval_poll_interval"):
                SigilClient(internal_token="tok", approval_poll_interval=bad)


# ──────────────────────────────────────────────────────────────────────────────
# SG-7 (ENT-94c): credential near-expiry warning + rotated-credential handling
# ──────────────────────────────────────────────────────────────────────────────

_ISSUE_OK = {
    "grant_id": "g-1",
    "biscuit_token": "tok",
    "revocation_id": "r-1",
    "expires_at": "2099-01-01T00:00:00Z",
}


class TestCredentialLifecycle:
    def test_credential_rejected_raises_terminal(self, client: SigilClient) -> None:
        """422 with code=credential_rejected → CredentialRotatedError (a terminal SigilAPIError)."""
        resp = _mock_response(422, {"error": "…", "code": "credential_rejected"})
        with patch.object(client._session, "post", return_value=resp):
            with pytest.raises(CredentialRotatedError) as exc:
                client.issue_token("a", "sa", "t", ["zep.search"], 3600)
        assert isinstance(exc.value, SigilAPIError)  # existing handlers still catch it
        assert exc.value.status_code == 422

    def test_422_without_code_is_plain_apierror(self, client: SigilClient) -> None:
        """A generic 422 (no credential_rejected code) stays a plain SigilAPIError."""
        resp = _mock_response(422, {"error": "tool_allowlist exceeds task scope"})
        with patch.object(client._session, "post", return_value=resp):
            with pytest.raises(SigilAPIError) as exc:
                client.issue_token("a", "sa", "t", ["zep.search"], 3600)
        assert not isinstance(exc.value, CredentialRotatedError)

    def test_near_expiry_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        c = SigilClient(internal_token="tok", credential_near_expiry_seconds=7 * 86400)
        soon = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        body = dict(_ISSUE_OK, credential_expires_at=soon)
        with caplog.at_level("WARNING", logger="sigil"):
            with patch.object(c._session, "post", return_value=_mock_response(201, body)):
                c.issue_token("a", "sa", "t", ["zep.search"], 3600)
        assert any("near expiry" in r.message for r in caplog.records)

    def test_far_expiry_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        c = SigilClient(internal_token="tok", credential_near_expiry_seconds=7 * 86400)
        far = (datetime.now(timezone.utc) + timedelta(days=60)).isoformat()
        body = dict(_ISSUE_OK, credential_expires_at=far)
        with caplog.at_level("WARNING", logger="sigil"):
            with patch.object(c._session, "post", return_value=_mock_response(201, body)):
                c.issue_token("a", "sa", "t", ["zep.search"], 3600)
        assert not any("near expiry" in r.message for r in caplog.records)

    def test_near_expiry_warns_once_per_timestamp(self, caplog: pytest.LogCaptureFixture) -> None:
        c = SigilClient(internal_token="tok", credential_near_expiry_seconds=7 * 86400)
        soon = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        body = dict(_ISSUE_OK, credential_expires_at=soon)
        with caplog.at_level("WARNING", logger="sigil"):
            with patch.object(c._session, "post", return_value=_mock_response(201, body)):
                c.issue_token("a", "sa", "t", ["zep.search"], 3600)
                c.issue_token("a", "sa", "t", ["zep.search"], 3600)
        assert sum("near expiry" in r.message for r in caplog.records) == 1

    def test_no_credential_expiry_field_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Field absent (server feature dark) → no warning, no error."""
        c = SigilClient(internal_token="tok")
        with caplog.at_level("WARNING", logger="sigil"):
            with patch.object(c._session, "post", return_value=_mock_response(201, dict(_ISSUE_OK))):
                c.issue_token("a", "sa", "t", ["zep.search"], 3600)
        assert not any("near expiry" in r.message for r in caplog.records)

    def test_422_non_json_body_is_plain_apierror(self, client: SigilClient) -> None:
        """A 422 whose body is not JSON (e.g. a WAF/nginx HTML intercept) must fall through to a
        plain SigilAPIError, never crash on the parse and never be treated as terminal."""
        resp = _mock_response(422, {})
        resp.json.side_effect = ValueError("not json")
        with patch.object(client._session, "post", return_value=resp):
            with pytest.raises(SigilAPIError) as exc:
                client.issue_token("a", "sa", "t", ["zep.search"], 3600)
        assert not isinstance(exc.value, CredentialRotatedError)

    def test_already_expired_credential_warns_without_negative_days(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An already-expired credential (accepted in observe mode) warns, and the log line must
        not contain a misleading negative 'days left'."""
        c = SigilClient(internal_token="tok", credential_near_expiry_seconds=7 * 86400)
        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        body = dict(_ISSUE_OK, credential_expires_at=past)
        with caplog.at_level("WARNING", logger="sigil"):
            with patch.object(c._session, "post", return_value=_mock_response(201, body)):
                c.issue_token("a", "sa", "t", ["zep.search"], 3600)
        warnings = [r.message for r in caplog.records if "near expiry" in r.message]
        assert len(warnings) == 1
        assert "~0.0 days left" in warnings[0]
        assert "~-" not in warnings[0]

    def test_malformed_credential_expiry_safely_ignored(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed credential_expires_at (e.g. a hostile server injecting a newline) fails the
        RFC3339 parse and is silently ignored: no warning, and the governed call is never broken.
        This is the first line of defense against log injection — the string never reaches the log."""
        c = SigilClient(internal_token="tok", credential_near_expiry_seconds=7 * 86400)
        body = dict(_ISSUE_OK, credential_expires_at="2099-01-01T00:00:00Z\nFORGED: fine")
        with caplog.at_level("WARNING", logger="sigil"):
            with patch.object(c._session, "post", return_value=_mock_response(201, body)):
                result = c.issue_token("a", "sa", "t", ["zep.search"], 3600)
        assert result["grant_id"] == "g-1"  # call succeeded
        assert not any("near expiry" in r.message for r in caplog.records)

    def test_near_expiry_log_uses_normalized_timestamp(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning must interpolate the re-serialized timestamp, never the raw server string —
        defense-in-depth for the log line. Feed a parseable 'Z' form and confirm the log carries the
        normalized (+00:00) representation, not the raw 'Z' string."""
        c = SigilClient(internal_token="tok", credential_near_expiry_seconds=7 * 86400)
        soon = datetime.now(timezone.utc) + timedelta(days=3)
        raw = soon.isoformat().replace("+00:00", "Z")  # canonical 'Z' form
        normalized = datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()  # '+00:00' form
        body = dict(_ISSUE_OK, credential_expires_at=raw)
        with caplog.at_level("WARNING", logger="sigil"):
            with patch.object(c._session, "post", return_value=_mock_response(201, body)):
                c.issue_token("a", "sa", "t", ["zep.search"], 3600)
        warnings = [r.message for r in caplog.records if "near expiry" in r.message]
        assert len(warnings) == 1
        assert normalized in warnings[0]
        assert raw not in warnings[0]


class TestCredentialNearExpiryConfigValidation:
    def test_rejects_non_finite_or_negative(self) -> None:
        for bad in (float("nan"), float("inf"), -1.0):
            with pytest.raises(ValueError, match="credential_near_expiry_seconds"):
                SigilClient(internal_token="tok", credential_near_expiry_seconds=bad)

    def test_zero_is_valid(self) -> None:
        # Zero means "always warn" — a valid, if noisy, configuration; must not raise.
        c = SigilClient(internal_token="tok", credential_near_expiry_seconds=0.0)
        assert c.credential_near_expiry_seconds == 0.0

    def test_non_numeric_env_raises_named_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIGIL_CREDENTIAL_NEAR_EXPIRY_SECONDS", "7d")
        with pytest.raises(ValueError, match="SIGIL_CREDENTIAL_NEAR_EXPIRY_SECONDS"):
            SigilClient(internal_token="tok")
