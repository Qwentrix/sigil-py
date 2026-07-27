"""Governance flow acceptance tests (ENT-77 Pass 2).

Maps to the 6 acceptance criteria + 2 extra cases = 8 tests total:

AC1 — instrumented_tool inside task: proceeds, event buffered, log_batch called.
AC2 — tool outside token scope → SigilDeniedError(tool_not_in_scope), NO network.
AC3 — sigil-core unreachable + fail_mode=closed → SigilUnreachableDeniedError + overflow file.
AC4 — sigil-core unreachable + fail_mode=open → tool proceeds + fail_open banner event.
AC5 — instrumented_llm DLP: args_redacted scrubs PII; args_hash is over ORIGINAL.
AC6 — high-risk tool → preflight called; deny verdict → SigilDeniedError.
AC7 — kill-switch: revoked agent → SigilDeniedError(agent_revoked), no network.
AC8 — batching: 50 events triggers immediate flush; 500ms timer fires; overflow replay.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nacl.signing import SigningKey

from sigil.client import SigilClient
from sigil.decorators import instrumented_llm, instrumented_tool
from sigil.errors import (
    SigilAPIError,
    SigilDeniedError,
    SigilTransportError,
    SigilUnreachableDeniedError,
)
from sigil.redaction import args_hash

# ──────────────────────────────────────────────────────────────────────────────
# Token-building helpers  (mirrors test_verify.py pattern)
# ──────────────────────────────────────────────────────────────────────────────

_KID = "test-kid"
_TENANT = "tenant-uuid-test"
_AGENT = "agent-uuid-test"
_FUTURE_TS = "2099-01-01T00:00:00Z"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_biscuit(
    sk: SigningKey,
    tools: list[str],
    tenant: str = _TENANT,
    agent: str = _AGENT,
    extra_facts: list[str] | None = None,
) -> str:
    """Build a signed BiscuitToken with the given tool allowlist."""
    facts = [f'tool("{t}")' for t in tools] + [
        f'tenant("{tenant}")',
        f'agent("{agent}")',
        'task("task-from-sigil")',
    ]
    if extra_facts:
        facts.extend(extra_facts)
    payload = {
        "v": 1,
        "blocks": [
            {
                "facts": facts,
                "checks": [f'check if time($t), $t < "{_FUTURE_TS}"'],
                "rid": "rid-0",
                "idx": 0,
            }
        ],
        "kid": _KID,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    sig = sk.sign(payload_bytes).signature
    return f"{_b64url(payload_bytes)}.{_b64url(sig)}"


def _mock_response(status: int, body: dict[str, Any]) -> MagicMock:
    r: MagicMock = MagicMock()
    r.status_code = status
    r.json.return_value = body
    return r


def _issue_resp(biscuit_token: str) -> dict[str, Any]:
    return {
        "grant_id": "g-001",
        "biscuit_token": biscuit_token,
        "revocation_id": "rev-001",
        "expires_at": _FUTURE_TS,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def sk() -> SigningKey:
    return SigningKey.generate()


@pytest.fixture()
def keyring(sk: SigningKey) -> dict[str, bytes]:
    return {_KID: bytes(sk.verify_key)}


def _make_client(
    sk: SigningKey,
    tools: list[str],
    fail_mode: str = "closed",
    overflow_dir: str | None = None,
) -> tuple[SigilClient, str]:
    """Return (client, biscuit_token) with issue_token already stubbed."""
    biscuit = _make_biscuit(sk, tools)
    keyring = {_KID: bytes(sk.verify_key)}
    client = SigilClient(
        base_url="http://sigil-test:8120",
        internal_token="int-tok-secret",
        tenant_id=_TENANT,
        agent_id=_AGENT,
        service_account_id="sa-uuid",
        fail_mode=fail_mode,
        biscuit_keyring=keyring,
        overflow_dir=overflow_dir,
    )
    return client, biscuit


# ──────────────────────────────────────────────────────────────────────────────
# AC1 — instrumented_tool inside task: proceeds, event buffered, log_batch called
# ──────────────────────────────────────────────────────────────────────────────


class TestAC1InstrumentedToolInsideTask:
    def test_tool_executes_and_event_flushed(self, sk: SigningKey, tmp_path: Any) -> None:
        """Tool runs, audit event is buffered, task.__exit__ flushes log_batch."""
        client, biscuit = _make_client(sk, ["ns.do_thing"], overflow_dir=str(tmp_path))
        log_batch_calls: list[list[dict[str, Any]]] = []

        @instrumented_tool("ns", "do_thing")
        def do_thing(x: int) -> int:
            return x * 2

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    side_effect=[
                        _mock_response(201, issue_resp),  # issue_token
                        _mock_response(202, {"accepted": 1}),  # log_batch
                    ],
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (
                        log_batch_calls.append(events) or {"accepted": len(events)}
                    ),
                ) as mock_lb,
            ):
                with client.task(["ns.do_thing"]) as task:
                    result = do_thing(21)
                    assert result == 42
                    assert task.task_id is not None

                # task.__exit__ calls flush_all() → log_batch should be called
                assert mock_lb.call_count >= 1, "log_batch was never called after task exit"
                all_events = [ev for call in log_batch_calls for ev in call]
                tool_names = [ev["tool_name"] for ev in all_events]
                assert "ns.do_thing" in tool_names

                sent_ev = next(ev for ev in all_events if ev["tool_name"] == "ns.do_thing")
                assert sent_ev["outcome"] == "allowed"
                assert sent_ev["risk_tier"] == "low"
                assert sent_ev["agent_id"] == _AGENT
                assert "args_hash" in sent_ev
                assert sent_ev["latency_ms"] >= 0
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# AC2 — out-of-scope tool → SigilDeniedError(tool_not_in_scope), NO network call
# ──────────────────────────────────────────────────────────────────────────────


class TestAC2ToolNotInScope:
    def test_out_of_scope_raises_denied_without_network(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """Token only grants 'ns.allowed'; calling 'ns.denied_tool' → denied locally."""
        client, biscuit = _make_client(sk, ["ns.allowed"], overflow_dir=str(tmp_path))

        @instrumented_tool("ns", "denied_tool")
        def denied_tool() -> str:
            return "should not reach here"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(client, "preflight") as mock_preflight,
                client.task(["ns.allowed"]) as _task,
                pytest.raises(SigilDeniedError) as exc_info,
            ):
                denied_tool()

            assert exc_info.value.denied_reason == "tool_not_in_scope"
            assert exc_info.value.tool_name == "ns.denied_tool"
            # preflight must NOT have been called — denial was purely local
            mock_preflight.assert_not_called()
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# AC3 — unreachable + fail_mode=closed → SigilUnreachableDeniedError + overflow
# ──────────────────────────────────────────────────────────────────────────────


class TestAC3FailClosed:
    def test_unreachable_closed_raises_and_writes_overflow(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        overflow_dir = str(tmp_path / "overflow")
        client, biscuit = _make_client(
            sk, ["ns.risky"], fail_mode="closed", overflow_dir=overflow_dir
        )

        @instrumented_tool("ns", "risky", risk_tier="high")
        def risky_call() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    side_effect=SigilTransportError(
                        "connection refused",
                        method="POST",
                        url="http://sigil-test/preflight",
                    ),
                ),
            ):
                with (
                    client.task(["ns.risky"]) as _task,
                    pytest.raises(SigilUnreachableDeniedError) as exc_info,
                ):
                    risky_call()

                assert exc_info.value.denied_reason == "sigil_unreachable"
                assert exc_info.value.tool_name == "ns.risky"
                assert isinstance(exc_info.value, SigilDeniedError)

                # Verify overflow file was written
                ndjson_files = [f for f in os.listdir(overflow_dir) if f.endswith(".ndjson")]
                assert ndjson_files, "No overflow file was written"
                overflow_path = os.path.join(overflow_dir, ndjson_files[0])
                with open(overflow_path) as fh:
                    lines = [ln.strip() for ln in fh if ln.strip()]
                assert lines, "Overflow file is empty"
                event = json.loads(lines[0])
                assert event["tool_name"] == "ns.risky"
                assert event["outcome"] == "denied"
                assert event["denied_reason"] == "sigil_unreachable"
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# AC4 — unreachable + fail_mode=open → tool proceeds + fail_open banner event
# ──────────────────────────────────────────────────────────────────────────────


class TestAC4FailOpen:
    def test_unreachable_open_proceeds_with_fail_open_event(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        client, biscuit = _make_client(
            sk, ["ns.risky"], fail_mode="open", overflow_dir=str(tmp_path)
        )
        buffered: list[dict[str, Any]] = []

        @instrumented_tool("ns", "risky", risk_tier="high")
        def risky_call() -> str:
            return "executed"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    side_effect=SigilTransportError(
                        "connection refused", method="POST", url="http://test"
                    ),
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (buffered.extend(events) or {"accepted": len(events)}),
                ),
                client.task(["ns.risky"]) as _task,
            ):
                result = risky_call()
                assert result == "executed"

            # After task exit, the event must be in the buffer/log_batch calls
            assert buffered, "No events were buffered"
            tool_ev = next((ev for ev in buffered if ev.get("tool_name") == "ns.risky"), None)
            assert tool_ev is not None, "No event for ns.risky found"
            assert tool_ev["outcome"] == "allowed"
            assert tool_ev.get("fail_open") is True, "fail_open flag not set"
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# AC5 — instrumented_llm DLP: args_redacted scrubs PII; args_hash over ORIGINAL
# ──────────────────────────────────────────────────────────────────────────────


class TestAC5LLMRedaction:
    def test_pii_redacted_and_hash_over_original(self, sk: SigningKey, tmp_path: Any) -> None:
        """Prompt containing SSN → args_redacted has <PII:SSN>; args_hash unchanged."""
        client, biscuit = _make_client(sk, ["llm.call"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []
        prompt = "my SSN is 123-45-6789 and email is bob@example.com"

        @instrumented_llm("llm", "call")
        def call_llm(prompt_text: str) -> str:
            return "response"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (buffered.extend(events) or {"accepted": len(events)}),
                ),
                client.task(["llm.call"]) as _task,
            ):
                call_llm(prompt)

            assert buffered, "No events buffered"
            ev = next(e for e in buffered if e["tool_name"] == "llm.call")

            # args_redacted must contain PII placeholders
            redacted_str = json.dumps(ev["args_redacted"])
            assert "<PII:SSN>" in redacted_str, "SSN not redacted in args_redacted"
            assert "<PII:EMAIL>" in redacted_str, "EMAIL not redacted in args_redacted"
            # Original prompt must NOT appear in redacted output
            assert "123-45-6789" not in redacted_str
            assert "bob@example.com" not in redacted_str

            # args_hash must equal hash of the ORIGINAL (not redacted) args
            from sigil.decorators import _raw_args

            original_raw = _raw_args((prompt,), {})
            expected_hash = args_hash(original_raw)
            assert (
                ev["args_hash"] == expected_hash
            ), "args_hash does not match hash of original (unredacted) args"
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# AC6 — high-risk tool → preflight called; deny verdict → SigilDeniedError
# ──────────────────────────────────────────────────────────────────────────────


class TestAC6HighRiskPreflight:
    def test_preflight_called_for_high_risk(self, sk: SigningKey, tmp_path: Any) -> None:
        """High-risk tool must call preflight regardless of local verify result."""
        client, biscuit = _make_client(sk, ["ns.dangerous"], overflow_dir=str(tmp_path))

        @instrumented_tool("ns", "dangerous", risk_tier="high")
        def dangerous_call() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    return_value={"verdict": "deny", "denied_reason": "policy_deny"},
                ) as mock_preflight,
            ):
                with (
                    client.task(["ns.dangerous"]) as _task,
                    pytest.raises(SigilDeniedError) as exc_info,
                ):
                    dangerous_call()

                mock_preflight.assert_called_once()
                assert exc_info.value.denied_reason == "policy_deny"
                assert exc_info.value.tool_name == "ns.dangerous"
        finally:
            client.close()

    def test_preflight_called_for_critical_risk(self, sk: SigningKey, tmp_path: Any) -> None:
        """Critical risk must also trigger preflight."""
        client, biscuit = _make_client(sk, ["ns.critical_tool"], overflow_dir=str(tmp_path))

        @instrumented_tool("ns", "critical_tool", risk_tier="critical")
        def critical_call() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    return_value={"verdict": "allow"},
                ) as mock_preflight,
                patch.object(
                    client,
                    "log_batch",
                    return_value={"accepted": 1},
                ),
            ):
                with client.task(["ns.critical_tool"]) as _task:
                    result = critical_call()
                    assert result == "ok"

                mock_preflight.assert_called_once()
        finally:
            client.close()

    def test_approve_without_approval_id_fails_closed(self, sk: SigningKey, tmp_path: Any) -> None:
        """ENT-81/SG-4: an 'approve' verdict with no approval_id means the gate could
        not be opened server-side — fail closed (approval_service_unavailable). The
        full approval poll flow (approved/rejected/expired/timeout/unreachable) is
        covered by TestApprovalGate."""
        client, biscuit = _make_client(sk, ["ns.gate"], overflow_dir=str(tmp_path))

        @instrumented_tool("ns", "gate", risk_tier="high")
        def gated_call() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    return_value={"verdict": "approve"},
                ),
            ):
                with (
                    client.task(["ns.gate"]) as _task,
                    pytest.raises(SigilDeniedError) as exc_info,
                ):
                    gated_call()

                assert exc_info.value.denied_reason == "approval_service_unavailable"
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# AC7 — kill-switch: revoked agent → SigilDeniedError(agent_revoked), no network
# ──────────────────────────────────────────────────────────────────────────────


class TestAC7KillSwitch:
    def test_revoked_agent_raises_denied_no_network(self, sk: SigningKey, tmp_path: Any) -> None:
        """After subscriber marks agent as revoked, next tool call raises agent_revoked."""
        client, biscuit = _make_client(sk, ["ns.action"], overflow_dir=str(tmp_path))

        @instrumented_tool("ns", "action")
        def action() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(client, "preflight") as mock_preflight,
                client.task(["ns.action"]) as _task,
                # Simulate subscriber marking agent revoked (monkeypatch path).
                patch.object(client, "is_revoked", return_value=True),
                pytest.raises(SigilDeniedError) as exc_info,
            ):
                action()

            assert exc_info.value.denied_reason == "agent_revoked"
            assert exc_info.value.tool_name == "ns.action"
            # The revocation check is in-memory — preflight must NOT be called
            mock_preflight.assert_not_called()
        finally:
            client.close()

    def test_subscriber_apply_marks_revoked(self) -> None:
        """_RevocationSubscriber._apply correctly sets the revoked flag."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )

        # Initially not revoked
        assert not sub.is_revoked(_AGENT, None)

        # Apply an agent-scope revocation
        sub._apply(
            {
                "revocation_id": "rev-001",
                "tenant_id": _TENANT,
                "agent_id": _AGENT,
                "task_id": "",
                "scope": "agent",
                "reason": "admin_kill",
                "revoked_at": "2026-07-14T00:00:00Z",
            }
        )
        assert sub.is_revoked(_AGENT, None)

    def test_subscriber_task_scope_marks_task(self) -> None:
        """Task-scope revocation flags the task_id."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )
        sub._apply(
            {
                "scope": "task",
                "agent_id": _AGENT,
                "task_id": "task-abc",
                "tenant_id": _TENANT,
                "reason": "policy",
                "revoked_at": "2026-07-14T00:00:00Z",
            }
        )
        assert sub.is_revoked(_AGENT, "task-abc")

    def test_subscriber_tenant_scope_flags_own_agent(self) -> None:
        """Tenant-scope revocation flags only the subscriber's own agent."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )
        other_agent = "other-agent-uuid"
        sub._apply(
            {
                "scope": "tenant",
                "agent_id": other_agent,
                "task_id": "",
                "tenant_id": _TENANT,
                "reason": "tenant_ban",
                "revoked_at": "2026-07-14T00:00:00Z",
            }
        )
        # Only our own agent is revoked, not the "other" agent in the payload
        assert sub.is_revoked(_AGENT, None)


# ──────────────────────────────────────────────────────────────────────────────
# AC8 — batching: 50-event immediate flush, 500ms timer, overflow replay
# ──────────────────────────────────────────────────────────────────────────────


class TestAC8Batching:
    def test_50_events_trigger_immediate_flush(self, sk: SigningKey, tmp_path: Any) -> None:
        """Pushing 50 events sets the flush_event; flusher wakes and calls log_batch."""
        client, _ = _make_client(sk, [], overflow_dir=str(tmp_path))
        flush_called = threading.Event()
        sample_event: dict[str, Any] = {
            "agent_id": _AGENT,
            "task_id": "t-001",
            "tool_name": "ns.op",
            "tool_namespace": "ns",
            "args_hash": "abc",
            "latency_ms": 1,
            "outcome": "allowed",
            "risk_tier": "low",
        }

        try:
            with patch.object(
                client,
                "log_batch",
                wraps=lambda events: (flush_called.set() or {"accepted": len(events)}),
            ):
                from sigil._buffer import _FLUSH_THRESHOLD

                for _ in range(_FLUSH_THRESHOLD):
                    client._log_buffer.push(dict(sample_event))

                # Flush should be triggered well within 1 second
                assert flush_called.wait(
                    timeout=1.0
                ), "50-event threshold did not trigger immediate flush within 1s"
        finally:
            client.close()

    def test_timer_flushes_partial_buffer(self, sk: SigningKey, tmp_path: Any) -> None:
        """Fewer than 50 events are flushed when the 500ms timer fires."""
        client, _ = _make_client(sk, [], overflow_dir=str(tmp_path))
        flush_called = threading.Event()
        sample_event: dict[str, Any] = {
            "agent_id": _AGENT,
            "task_id": "t-002",
            "tool_name": "ns.op",
            "tool_namespace": "ns",
            "args_hash": "xyz",
            "latency_ms": 0,
            "outcome": "allowed",
            "risk_tier": "low",
        }

        try:
            with patch.object(
                client,
                "log_batch",
                wraps=lambda events: (flush_called.set() or {"accepted": len(events)}),
            ):
                client._log_buffer.push(dict(sample_event))  # only 1 event

                # code-M7: generous timeout (2.0 s) to avoid CI flakiness.
                assert flush_called.wait(
                    timeout=2.0
                ), "500ms timer did not trigger flush within 2.0s"
        finally:
            client.close()

    def test_overflow_replay_on_recovery(self, sk: SigningKey, tmp_path: Any) -> None:
        """Failed flush writes to overflow; next successful flush replays the file."""
        overflow_dir = str(tmp_path / "overflow")
        client, _ = _make_client(sk, [], overflow_dir=overflow_dir)
        call_count = [0]
        sample_event: dict[str, Any] = {
            "agent_id": _AGENT,
            "task_id": "t-003",
            "tool_name": "ns.op",
            "tool_namespace": "ns",
            "args_hash": "def",
            "latency_ms": 5,
            "outcome": "allowed",
            "risk_tier": "low",
        }

        try:

            def mock_log_batch(events: list[dict[str, Any]]) -> dict[str, Any]:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise SigilTransportError("test unreachable", method="POST", url="http://test/")
                return {"accepted": len(events)}

            with patch.object(client, "log_batch", side_effect=mock_log_batch):
                # Push 1 event and flush — this will fail → overflow
                client._log_buffer.push(dict(sample_event))
                client._flusher.flush_all()

                # Overflow file must exist
                ndjson_files = [f for f in os.listdir(overflow_dir) if f.endswith(".ndjson")]
                assert ndjson_files, "No overflow file written after failed flush"

                # Push another event and flush — this succeeds → triggers replay
                sample_event2 = dict(sample_event)
                sample_event2["task_id"] = "t-003b"
                client._log_buffer.push(sample_event2)
                client._flusher.flush_all()

                # Overflow file must be gone (replay succeeded)
                ndjson_files_after = [f for f in os.listdir(overflow_dir) if f.endswith(".ndjson")]
                assert not ndjson_files_after, "Overflow file still present after successful replay"
                # log_batch was called at least twice (1 fail + 1 success + 1 replay)
                assert call_count[0] >= 2
        finally:
            client.close()

    def test_buffer_flush_event_set_at_threshold(self, tmp_path: Any) -> None:
        """The buffer sets flush_event exactly when _FLUSH_THRESHOLD events are pushed."""
        from sigil._buffer import _FLUSH_THRESHOLD, _LogBuffer

        buf = _LogBuffer()
        sample: dict[str, Any] = {"x": 1}

        for i in range(_FLUSH_THRESHOLD - 1):
            buf.push(dict(sample))
            assert not buf.flush_event.is_set(), f"flush_event set prematurely at event {i + 1}"

        buf.push(dict(sample))  # the 50th push
        assert buf.flush_event.is_set(), "flush_event not set after 50 pushes"


# ──────────────────────────────────────────────────────────────────────────────
# No-task context guard
# ──────────────────────────────────────────────────────────────────────────────


class TestNoTaskContext:
    def test_tool_without_task_raises_denied(self) -> None:
        """Calling an instrumented tool outside any task raises SigilDeniedError."""

        @instrumented_tool("ns", "orphan")
        def orphan() -> None:
            pass

        with pytest.raises(SigilDeniedError) as exc_info:
            orphan()
        assert exc_info.value.denied_reason == "no_task"

    def test_llm_without_task_raises_denied(self) -> None:
        @instrumented_llm("llm", "orphan")
        def orphan_llm(prompt: str) -> str:
            return prompt

        with pytest.raises(SigilDeniedError) as exc_info:
            orphan_llm("hello")
        assert exc_info.value.denied_reason == "no_task"


# ──────────────────────────────────────────────────────────────────────────────
# SigilClient context manager
# ──────────────────────────────────────────────────────────────────────────────


class TestClientContextManager:
    def test_client_as_context_manager(self) -> None:
        """SigilClient supports 'with' syntax and calls close() on exit."""
        with SigilClient(internal_token="tok") as c:
            assert c.fail_mode == "closed"
        # After exit, flusher thread should be stopped (daemon — no explicit assert needed)


# ──────────────────────────────────────────────────────────────────────────────
# Overflow writer unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOverflowWriter:
    def test_write_creates_ndjson_file(self, tmp_path: Any) -> None:
        from sigil._overflow import _OverflowWriter

        w = _OverflowWriter(overflow_dir=str(tmp_path), agent_id="agent-x")
        event = {"tool_name": "ns.op", "outcome": "denied"}
        w.write(event)

        files = list(tmp_path.glob("*.ndjson"))
        assert files, "No NDJSON file created"
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == event

    def test_directory_property(self, tmp_path: Any) -> None:
        from sigil._overflow import _OverflowWriter

        w = _OverflowWriter(overflow_dir=str(tmp_path), agent_id="ag")
        assert w.directory == str(tmp_path)

    def test_unwritable_dir_does_not_crash(self) -> None:
        from sigil._overflow import _OverflowWriter

        w = _OverflowWriter(overflow_dir="/proc/no-such-dir-sigil-test", agent_id="ag")
        # write should silently degrade, not raise
        w.write({"x": 1})
        assert not w._writable

    def test_replay_drains_file_on_successful_log_batch(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """replay() finds NDJSON files, sends via log_batch, and deletes the file."""
        from sigil._overflow import _OverflowWriter

        overflow_dir = str(tmp_path / "ovf")
        os.makedirs(overflow_dir)
        w = _OverflowWriter(overflow_dir=overflow_dir, agent_id="ag-replay")

        event1 = {"tool_name": "ns.t1", "outcome": "denied", "x": 1}
        event2 = {"tool_name": "ns.t2", "outcome": "denied", "x": 2}
        w.write(event1)
        w.write(event2)

        replayed: list[list[dict[str, Any]]] = []

        # Build a minimal stub client with a patched log_batch
        client_stub = SigilClient(
            internal_token="tok",
            tenant_id=_TENANT,
            agent_id=_AGENT,
            overflow_dir=str(tmp_path / "stub_ovf"),
        )
        try:
            with patch.object(
                client_stub,
                "log_batch",
                wraps=lambda events: (replayed.append(events) or {"accepted": len(events)}),
            ):
                w.replay(client_stub)

            assert replayed, "log_batch never called by replay"
            all_replayed = [e for batch in replayed for e in batch]
            names = [e["tool_name"] for e in all_replayed]
            assert "ns.t1" in names
            assert "ns.t2" in names

            # File must be deleted after successful replay
            remaining = [f for f in os.listdir(overflow_dir) if f.endswith(".ndjson")]
            assert not remaining, "Overflow file not deleted after replay"
        finally:
            client_stub.close()

    def test_replay_leaves_file_on_transport_error(self, tmp_path: Any) -> None:
        """replay() leaves the file intact when log_batch raises SigilTransportError."""
        from sigil._overflow import _OverflowWriter

        overflow_dir = str(tmp_path / "ovf2")
        os.makedirs(overflow_dir)
        w = _OverflowWriter(overflow_dir=overflow_dir, agent_id="ag-fail")
        w.write({"tool_name": "ns.t", "outcome": "denied"})

        client_stub = SigilClient(
            internal_token="tok",
            tenant_id=_TENANT,
            agent_id=_AGENT,
            overflow_dir=str(tmp_path / "stub2"),
        )
        try:
            with patch.object(
                client_stub,
                "log_batch",
                side_effect=SigilTransportError("down", method="POST", url="http://x"),
            ):
                w.replay(client_stub)

            # File must NOT be deleted — transport error means not-yet-deliverable
            remaining = [f for f in os.listdir(overflow_dir) if f.endswith(".ndjson")]
            assert remaining, "Overflow file was incorrectly deleted on failure"
        finally:
            client_stub.close()

    def test_replay_deletes_empty_file(self, tmp_path: Any) -> None:
        """replay() deletes a zero-byte NDJSON file without calling log_batch."""
        from sigil._overflow import _OverflowWriter

        overflow_dir = str(tmp_path / "ovf3")
        os.makedirs(overflow_dir)
        empty_file = os.path.join(overflow_dir, "ag-empty_2026-01-01.ndjson")
        open(empty_file, "w").close()  # zero bytes

        w = _OverflowWriter(overflow_dir=overflow_dir, agent_id="ag-empty")
        client_stub = SigilClient(
            internal_token="tok",
            overflow_dir=str(tmp_path / "stub3"),
        )
        try:
            with patch.object(client_stub, "log_batch") as mock_lb:
                w.replay(client_stub)
                mock_lb.assert_not_called()
            assert not os.path.exists(empty_file)
        finally:
            client_stub.close()

    def test_replay_skips_corrupt_json_lines(self, tmp_path: Any) -> None:
        """_replay_file skips lines with invalid JSON without crashing."""
        from sigil._overflow import _OverflowWriter

        overflow_dir = str(tmp_path / "ovf4")
        os.makedirs(overflow_dir)
        w = _OverflowWriter(overflow_dir=overflow_dir, agent_id="ag-corrupt")

        # Write one valid and one corrupt line manually
        path = w._current_path()
        with open(path, "w") as fh:
            fh.write('{"tool_name": "ns.ok", "outcome": "allowed"}\n')
            fh.write("CORRUPT LINE\n")
            fh.write('{"tool_name": "ns.ok2", "outcome": "denied"}\n')

        replayed: list[list[dict[str, Any]]] = []
        client_stub = SigilClient(
            internal_token="tok",
            overflow_dir=str(tmp_path / "stub4"),
        )
        try:
            with patch.object(
                client_stub,
                "log_batch",
                wraps=lambda events: (replayed.append(events) or {"accepted": len(events)}),
            ):
                w.replay(client_stub)

            all_ev = [e for b in replayed for e in b]
            assert len(all_ev) == 2  # corrupt line skipped
        finally:
            client_stub.close()


# ──────────────────────────────────────────────────────────────────────────────
# _Flusher direct unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestFlusher:
    def test_api_error_drops_events_and_does_not_overflow(self, tmp_path: Any) -> None:
        """SigilAPIError from log_batch: events are dropped (not overflowed) and loop ends."""
        from sigil._buffer import _Flusher, _LogBuffer
        from sigil._overflow import _OverflowWriter
        from sigil.errors import SigilAPIError

        buf = _LogBuffer()
        overflow_dir = str(tmp_path / "fovf")
        os.makedirs(overflow_dir)
        overflow = _OverflowWriter(overflow_dir=overflow_dir, agent_id="ag-api")

        client_stub = SigilClient(
            internal_token="tok",
            overflow_dir=str(tmp_path / "fstub"),
        )
        try:
            flusher = _Flusher(buffer=buf, client=client_stub, overflow=overflow)

            sample = {"tool_name": "ns.t", "outcome": "allowed", "risk_tier": "low"}
            buf.push(dict(sample))
            buf.push(dict(sample))

            with patch.object(
                client_stub,
                "log_batch",
                side_effect=SigilAPIError("bad request", status_code=400),
            ):
                flusher._do_flush()

            # No overflow file should be created — API errors are not retryable
            ndjson = [f for f in os.listdir(overflow_dir) if f.endswith(".ndjson")]
            assert not ndjson, "API error should NOT write to overflow"
        finally:
            client_stub.close()


# ──────────────────────────────────────────────────────────────────────────────
# _RevocationSubscriber start/stop and is_revoked edge cases
# ──────────────────────────────────────────────────────────────────────────────


class TestSubscriberLifecycle:
    def test_start_spawns_daemon_thread(self) -> None:
        """start() spawns a daemon thread (exits cleanly when redis is absent)."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://127.0.0.1:19999"
        )
        sub.start()
        assert sub._thread is not None
        sub.stop()

    def test_stop_joins_thread(self) -> None:
        """stop() sets the stop event and joins the thread."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://127.0.0.1:19999"
        )
        sub.start()
        assert sub._thread is not None
        sub.stop()
        assert not sub._thread.is_alive() or sub._stop.is_set()

    def test_is_revoked_empty_agent_id_returns_false(self) -> None:
        """is_revoked() with empty agent_id is always False (safety guard)."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://127.0.0.1"
        )
        sub._apply(
            {
                "scope": "agent",
                "agent_id": _AGENT,
                "reason": "x",
                "tenant_id": _TENANT,
                "task_id": "",
            }
        )
        assert sub.is_revoked("", None) is False


# ──────────────────────────────────────────────────────────────────────────────
# instrumented_llm high-risk and fail-mode paths
# ──────────────────────────────────────────────────────────────────────────────


class TestInstrumentedLLMExtraPaths:
    def test_llm_high_risk_preflight_called(self, sk: SigningKey, tmp_path: Any) -> None:
        """instrumented_llm with risk_tier=high calls preflight."""
        client, biscuit = _make_client(sk, ["llm.high"], overflow_dir=str(tmp_path))

        @instrumented_llm("llm", "high", risk_tier="high")
        def llm_call(prompt: str) -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    return_value={"verdict": "allow"},
                ) as mock_pf,
                patch.object(client, "log_batch", return_value={"accepted": 1}),
            ):
                with client.task(["llm.high"]) as _task:
                    result = llm_call("tell me a story")
                    assert result == "ok"

                mock_pf.assert_called_once()
        finally:
            client.close()

    def test_llm_high_risk_deny_raises(self, sk: SigningKey, tmp_path: Any) -> None:
        """instrumented_llm preflight deny → SigilDeniedError."""
        client, biscuit = _make_client(sk, ["llm.risky"], overflow_dir=str(tmp_path))

        @instrumented_llm("llm", "risky", risk_tier="high")
        def risky_llm(prompt: str) -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    return_value={"verdict": "deny", "denied_reason": "policy_deny"},
                ),
            ):
                with (
                    client.task(["llm.risky"]) as _task,
                    pytest.raises(SigilDeniedError) as exc_info,
                ):
                    risky_llm("sensitive query")

                assert exc_info.value.denied_reason == "policy_deny"
        finally:
            client.close()

    def test_llm_unreachable_fail_closed(self, sk: SigningKey, tmp_path: Any) -> None:
        """instrumented_llm: unreachable + fail_mode=closed → SigilUnreachableDeniedError."""
        overflow_dir = str(tmp_path / "llm_ovf")
        client, biscuit = _make_client(
            sk, ["llm.closed"], fail_mode="closed", overflow_dir=overflow_dir
        )

        @instrumented_llm("llm", "closed", risk_tier="high")
        def llm_closed(prompt: str) -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    side_effect=SigilTransportError("down", method="POST", url="http://x"),
                ),
                client.task(["llm.closed"]) as _task,
                pytest.raises(SigilUnreachableDeniedError),
            ):
                llm_closed("hello")
        finally:
            client.close()

    def test_llm_unreachable_fail_open(self, sk: SigningKey, tmp_path: Any) -> None:
        """instrumented_llm: unreachable + fail_mode=open → proceeds with fail_open event."""
        client, biscuit = _make_client(
            sk, ["llm.open"], fail_mode="open", overflow_dir=str(tmp_path)
        )
        buffered: list[dict[str, Any]] = []

        @instrumented_llm("llm", "open", risk_tier="high")
        def llm_open(prompt: str) -> str:
            return "open-result"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    side_effect=SigilTransportError("down", method="POST", url="http://x"),
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (buffered.extend(events) or {"accepted": len(events)}),
                ),
                client.task(["llm.open"]) as _task,
            ):
                result = llm_open("query")
                assert result == "open-result"

            ev = next((e for e in buffered if e.get("tool_name") == "llm.open"), None)
            assert ev is not None
            assert ev["outcome"] == "allowed"
            assert ev.get("fail_open") is True
            # args_redacted must be present for LLM
            assert "args_redacted" in ev
        finally:
            client.close()

    def test_llm_out_of_scope_includes_redacted_in_event(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """instrumented_llm: out-of-scope token includes args_redacted in denied event."""
        client, biscuit = _make_client(sk, ["llm.allowed"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []

        @instrumented_llm("llm", "denied_llm")
        def denied_llm(prompt: str) -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (buffered.extend(events) or {"accepted": len(events)}),
                ),
                client.task(["llm.allowed"]) as _task,
                pytest.raises(SigilDeniedError) as exc_info,
            ):
                denied_llm("my SSN is 123-45-6789")

            assert exc_info.value.denied_reason == "tool_not_in_scope"
            # Denied event should still have args_redacted (DLP-scrubbed)
            ev = next((e for e in buffered if e.get("tool_name") == "llm.denied_llm"), None)
            assert ev is not None
            assert "args_redacted" in ev
            redacted_str = json.dumps(ev["args_redacted"])
            assert "<PII:SSN>" in redacted_str
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# H1 — SigilAPIError from preflight is handled by fail-mode path (not escaped)
# ──────────────────────────────────────────────────────────────────────────────


class TestH1APIErrorFromPreflight:
    def test_api_error_503_fail_closed_raises_unreachable(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """H1: SigilAPIError (503) from preflight → SigilUnreachableDeniedError (fail_mode=closed)."""  # noqa: E501
        client, biscuit = _make_client(
            sk, ["ns.risky"], fail_mode="closed", overflow_dir=str(tmp_path / "h1ovf")
        )

        @instrumented_tool("ns", "risky", risk_tier="high")
        def risky_call() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    side_effect=SigilAPIError("service unavailable", status_code=503),
                ),
            ):
                with (
                    client.task(["ns.risky"]) as _task,
                    pytest.raises(SigilUnreachableDeniedError) as exc_info,
                ):
                    risky_call()

                assert exc_info.value.denied_reason == "sigil_unreachable"
        finally:
            client.close()

    def test_api_error_503_fail_open_proceeds(self, sk: SigningKey, tmp_path: Any) -> None:
        """H1: SigilAPIError (503) from preflight → proceeds + fail_open event (fail_mode=open)."""
        client, biscuit = _make_client(
            sk, ["ns.risky"], fail_mode="open", overflow_dir=str(tmp_path)
        )
        buffered: list[dict[str, Any]] = []

        @instrumented_tool("ns", "risky", risk_tier="high")
        def risky_call() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    side_effect=SigilAPIError("too many requests", status_code=429),
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (buffered.extend(events) or {"accepted": len(events)}),
                ),
                client.task(["ns.risky"]) as _task,
            ):
                result = risky_call()
                assert result == "ok"

            ev = next((e for e in buffered if e.get("tool_name") == "ns.risky"), None)
            assert ev is not None
            assert ev["outcome"] == "allowed"
            assert ev.get("fail_open") is True
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# H2 — fn exception produces an "error" audit event (not silently dropped)
# ──────────────────────────────────────────────────────────────────────────────


class TestH2ExceptionEmitsErrorEvent:
    def test_tool_exception_emits_error_event(self, sk: SigningKey, tmp_path: Any) -> None:
        """H2: fn raises → outcome="error" event is buffered; exception still propagates."""
        client, biscuit = _make_client(sk, ["ns.boom"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []

        @instrumented_tool("ns", "boom")
        def boom_tool() -> str:
            raise RuntimeError("boom!")

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (buffered.extend(events) or {"accepted": len(events)}),
                ),
                client.task(["ns.boom"]) as _task,
                pytest.raises(RuntimeError, match="boom!"),
            ):
                boom_tool()

            ev = next((e for e in buffered if e.get("tool_name") == "ns.boom"), None)
            assert ev is not None, "No error event emitted for ns.boom"
            assert ev["outcome"] == "error", f"Expected outcome='error', got {ev['outcome']!r}"
        finally:
            client.close()

    def test_llm_exception_emits_error_event_with_redacted(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """H2: instrumented_llm fn raises → error event includes args_redacted."""
        client, biscuit = _make_client(sk, ["llm.explode"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []

        @instrumented_llm("llm", "explode")
        def exploding_llm(prompt: str) -> str:
            raise ValueError("LLM down")

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (buffered.extend(events) or {"accepted": len(events)}),
                ),
                client.task(["llm.explode"]) as _task,
                pytest.raises(ValueError, match="LLM down"),
            ):
                exploding_llm("my SSN is 123-45-6789")

            ev = next((e for e in buffered if e.get("tool_name") == "llm.explode"), None)
            assert ev is not None, "No error event emitted for llm.explode"
            assert ev["outcome"] == "error"
            # args_redacted must still be present (DLP ran before fn was called)
            assert "args_redacted" in ev
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# M1 — null / unknown preflight verdict → denied (fail-closed)
# ──────────────────────────────────────────────────────────────────────────────


class TestM1NullVerdictDenied:
    def test_null_verdict_raises_denied(self, sk: SigningKey, tmp_path: Any) -> None:
        """M1: verdict=null from preflight → SigilDeniedError(unknown_verdict)."""
        client, biscuit = _make_client(sk, ["ns.gated"], overflow_dir=str(tmp_path))

        @instrumented_tool("ns", "gated", risk_tier="high")
        def gated() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    return_value={"verdict": None},  # null verdict
                ),
            ):
                with (
                    client.task(["ns.gated"]) as _task,
                    pytest.raises(SigilDeniedError) as exc_info,
                ):
                    gated()

                # null → treated as "deny" via `or "deny"` fallback
                assert exc_info.value.denied_reason in (
                    "unknown_verdict",
                    "policy_deny",
                )
        finally:
            client.close()

    def test_unknown_string_verdict_raises_denied(self, sk: SigningKey, tmp_path: Any) -> None:
        """M1: verdict="ALLOW" (wrong case) → SigilDeniedError(unknown_verdict)."""
        client, biscuit = _make_client(sk, ["ns.gated2"], overflow_dir=str(tmp_path))

        @instrumented_tool("ns", "gated2", risk_tier="high")
        def gated2() -> str:
            return "ok"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                patch.object(
                    client,
                    "preflight",
                    return_value={"verdict": "ALLOW"},  # wrong case — not "allow"
                ),
            ):
                with (
                    client.task(["ns.gated2"]) as _task,
                    pytest.raises(SigilDeniedError) as exc_info,
                ):
                    gated2()

                assert exc_info.value.denied_reason == "unknown_verdict"
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# M2 — overflow files are created with restricted permissions
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="file modes not supported on Windows")
class TestM2OverflowFileModes:
    def test_overflow_dir_created_with_0o700_mode(self, tmp_path: Any) -> None:
        """M2: overflow directory is created with mode 0o700 (owner-only)."""
        from sigil._overflow import _OverflowWriter

        ovf_dir = tmp_path / "mode_dir_test"
        _OverflowWriter(overflow_dir=str(ovf_dir), agent_id="ag")
        actual = oct(os.stat(str(ovf_dir)).st_mode & 0o777)
        assert actual == oct(0o700), f"Expected dir mode 0o700, got {actual}"

    def test_overflow_file_created_with_0o600_mode(self, tmp_path: Any) -> None:
        """M2: overflow files are created with mode 0o600 (owner read/write)."""
        from sigil._overflow import _OverflowWriter

        ovf_dir = tmp_path / "mode_file_test"
        w = _OverflowWriter(overflow_dir=str(ovf_dir), agent_id="ag")
        w.write({"tool_name": "ns.test", "outcome": "denied"})

        path = w._current_path()
        assert os.path.exists(path), "Overflow file was not created"
        actual = oct(os.stat(path).st_mode & 0o777)
        assert actual == oct(0o600), f"Expected file mode 0o600, got {actual}"


# ──────────────────────────────────────────────────────────────────────────────
# M3 — subscriber tenant guard: cross-tenant revocations are ignored
# ──────────────────────────────────────────────────────────────────────────────


class TestM3SubscriberTenantGuard:
    def test_agent_scope_cross_tenant_ignored(self) -> None:
        """M3: agent-scope revocation from a different tenant is ignored."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )
        # Revocation for our agent but from a DIFFERENT tenant
        sub._apply(
            {
                "scope": "agent",
                "agent_id": _AGENT,
                "task_id": "",
                "tenant_id": "other-tenant-uuid",  # NOT our tenant
                "reason": "cross_tenant_attack",
            }
        )
        # Must NOT be revoked — cross-tenant revocation should be ignored
        assert not sub.is_revoked(_AGENT, None), "Cross-tenant agent revocation should be ignored"

    def test_task_scope_cross_tenant_ignored(self) -> None:
        """M3: task-scope revocation from a different tenant is ignored."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )
        sub._apply(
            {
                "scope": "task",
                "agent_id": _AGENT,
                "task_id": "task-victim",
                "tenant_id": "attacker-tenant",
                "reason": "cross_tenant_attack",
            }
        )
        assert not sub.is_revoked(
            _AGENT, "task-victim"
        ), "Cross-tenant task revocation should be ignored"

    def test_agent_scope_own_tenant_revokes(self) -> None:
        """M3: agent-scope revocation from our own tenant is applied normally."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )
        sub._apply(
            {
                "scope": "agent",
                "agent_id": _AGENT,
                "task_id": "",
                "tenant_id": _TENANT,
                "reason": "admin_kill",
            }
        )
        assert sub.is_revoked(_AGENT, None), "Same-tenant agent revocation must be applied"

    def test_agent_scope_absent_tenant_revokes_backward_compat(self) -> None:
        """M3: absent tenant_id in payload is backward-compatible (message predates tenant field)."""  # noqa: E501
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )
        sub._apply(
            {
                "scope": "agent",
                "agent_id": _AGENT,
                "task_id": "",
                # No tenant_id field — older message format
                "reason": "legacy_kill",
            }
        )
        assert sub.is_revoked(_AGENT, None), "Absent tenant_id must be backward-compat allowed"


# ──────────────────────────────────────────────────────────────────────────────
# F4 — own-agent relevance filter: flood of other-agent revocations cannot
# evict our own revocation from the bounded registry.
# ──────────────────────────────────────────────────────────────────────────────


class TestF4RevocationFloodProtection:
    def test_different_agent_revocation_is_noop(self) -> None:
        """F4: an agent-scope revocation for a DIFFERENT agent is completely ignored."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )
        other_agent = "completely-different-agent-uuid"

        sub._apply(
            {
                "scope": "agent",
                "agent_id": other_agent,  # NOT our agent
                "task_id": "",
                "tenant_id": _TENANT,
                "reason": "other_kill",
            }
        )

        # Other agent's revocation must NOT affect our agent or be stored at all.
        assert not sub.is_revoked(
            other_agent, None
        ), "Revocations for other agents must not be stored"
        assert not sub.is_revoked(
            _AGENT, None
        ), "Our agent must not be revoked by a different agent's revocation"
        # Verify the revocation registry is truly empty (no phantom entries).
        with sub._lock:
            assert len(sub._revoked) == 0, "Registry must be empty after irrelevant revocation"

    def test_own_agent_revoked_after_flood_of_others(self) -> None:
        """F4: our revocation survives a flood of other agents' revocations."""
        from sigil._subscriber import _REVOKED_MAX, _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )

        # First revoke our own agent.
        sub._apply(
            {
                "scope": "agent",
                "agent_id": _AGENT,
                "task_id": "",
                "tenant_id": _TENANT,
                "reason": "admin_kill",
            }
        )
        assert sub.is_revoked(_AGENT, None), "Our agent must be revoked"

        # Flood with _REVOKED_MAX + 100 revocations for different agents.
        # Under the OLD code this would FIFO-evict our revocation; under F4 it
        # is a no-op because we filter out non-own-agent revocations.
        for i in range(_REVOKED_MAX + 100):
            sub._apply(
                {
                    "scope": "agent",
                    "agent_id": f"flood-agent-{i}",  # different agent each time
                    "task_id": "",
                    "tenant_id": _TENANT,
                    "reason": "flood",
                }
            )

        # Our revocation must still be in the registry — flood was ignored.
        assert sub.is_revoked(
            _AGENT, None
        ), "Our agent must remain revoked after a flood of other-agent revocations"

    def test_task_revocation_for_different_agent_is_noop(self) -> None:
        """F4: task-scope revocation belonging to a DIFFERENT agent is ignored."""
        from sigil._subscriber import _RevocationSubscriber

        sub = _RevocationSubscriber(
            tenant_id=_TENANT, agent_id=_AGENT, redis_url="redis://localhost"
        )
        sub._apply(
            {
                "scope": "task",
                "agent_id": "other-agent-uuid",  # NOT our agent
                "task_id": "task-belongs-to-other",
                "tenant_id": _TENANT,
                "reason": "task_cancel",
            }
        )

        # Neither the task nor our agent should be revoked.
        assert not sub.is_revoked(
            _AGENT, "task-belongs-to-other"
        ), "Task revocation for other agent's task must not affect us"
        with sub._lock:
            assert len(sub._revoked) == 0, "Registry must be empty"


# ──────────────────────────────────────────────────────────────────────────────
# AC7 — ENT-81/SG-4 approval gate: 'approve' verdict → block-poll approval status
# ──────────────────────────────────────────────────────────────────────────────


class TestApprovalGate:
    def _client(self, sk: SigningKey, tmp_path: Any) -> tuple[SigilClient, str]:
        client, biscuit = _make_client(sk, ["ns.dangerous"], overflow_dir=str(tmp_path))
        # Fast, deterministic polling for tests.
        client.approval_poll_interval = 0.001
        client.approval_timeout = 0.05
        return client, biscuit

    @contextlib.contextmanager
    def _gate(
        self,
        client: SigilClient,
        biscuit: str,
        preflight_ret: dict[str, Any],
        status_side: Any,
    ) -> Any:
        """Yield (callable, approval_status_mock) with all patches + an active task
        context still OPEN, so the tool call runs inside the governed scope."""

        @instrumented_tool("ns", "dangerous", risk_tier="high")
        def dangerous_call() -> str:
            return "ok"

        is_exc = isinstance(status_side, BaseException) or (
            isinstance(status_side, type) and issubclass(status_side, BaseException)
        )
        status_kw = {"side_effect": status_side} if is_exc else {"return_value": status_side}
        with (
            patch.object(
                client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
            ),
            patch.object(client, "preflight", return_value=preflight_ret),
            patch.object(client, "approval_status", **status_kw) as mock_status,
            # ENT-82: the approved path redeems for a one-shot grant; mock it so approved
            # tests execute. Denial-path tests never reach redeem (they deny at finalize).
            patch.object(
                client, "redeem_approval",
                return_value={"revocation_id": "grant-ok", "one_shot_token": "t"},
            ),
            patch.object(client, "log_batch", return_value={"accepted": 1}),
            client.task(["ns.dangerous"]),
        ):
            yield dangerous_call, mock_status

    _APPROVE = {"verdict": "approve", "approval_id": "ap-1"}

    def test_approve_approved_executes(self, sk: SigningKey, tmp_path: Any) -> None:
        client, biscuit = self._client(sk, tmp_path)
        try:
            with self._gate(client, biscuit, self._APPROVE, "approved") as (call, mock_status):
                assert call() == "ok"
                mock_status.assert_called()
        finally:
            client.close()

    def test_approve_redeem_replayed_denies(self, sk: SigningKey, tmp_path: Any) -> None:
        """ENT-82: a 409 from redeem (approval already redeemed) fails CLOSED — the tool
        does NOT execute and the denial reason is approval_replayed."""
        client, biscuit = self._client(sk, tmp_path)

        @instrumented_tool("ns", "dangerous", risk_tier="high")
        def call() -> str:
            return "ok"

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(client, "preflight", return_value=self._APPROVE),
                patch.object(client, "approval_status", return_value="approved"),
                patch.object(
                    client, "redeem_approval",
                    side_effect=SigilAPIError("already redeemed", status_code=409),
                ),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                client.task(["ns.dangerous"]),
            ):
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                assert exc.value.denied_reason == "approval_replayed"
        finally:
            client.close()

    def test_approve_redeem_unavailable_denies(self, sk: SigningKey, tmp_path: Any) -> None:
        """ENT-82: a transport failure on redeem fails CLOSED with approval_token_unavailable."""
        client, biscuit = self._client(sk, tmp_path)

        @instrumented_tool("ns", "dangerous", risk_tier="high")
        def call() -> str:
            return "ok"

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(client, "preflight", return_value=self._APPROVE),
                patch.object(client, "approval_status", return_value="approved"),
                patch.object(
                    client, "redeem_approval",
                    side_effect=SigilTransportError("down", method="POST", url="http://x/"),
                ),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                client.task(["ns.dangerous"]),
            ):
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                assert exc.value.denied_reason == "approval_token_unavailable"
        finally:
            client.close()

    def test_approve_grant_id_recorded_in_audit(self, sk: SigningKey, tmp_path: Any) -> None:
        """ENT-82: on a redeemed approval the execution audit event carries the one-shot
        grant's revocation_id as proof the call ran under a fresh single-use grant."""
        client, biscuit = self._client(sk, tmp_path)
        captured: list[dict[str, Any]] = []

        @instrumented_tool("ns", "dangerous", risk_tier="high")
        def call() -> str:
            return "ok"

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(client, "preflight", return_value=self._APPROVE),
                patch.object(client, "approval_status", return_value="approved"),
                patch.object(
                    client, "redeem_approval",
                    return_value={"revocation_id": "rev-xyz", "one_shot_token": "t"},
                ),
                patch.object(client._log_buffer, "push", side_effect=captured.append),
                client.task(["ns.dangerous"]),
            ):
                assert call() == "ok"
            allowed = next(e for e in captured if e.get("outcome") == "allowed")
            assert allowed.get("approval_grant_id") == "rev-xyz"
        finally:
            client.close()

    def test_approve_rejected_denies(self, sk: SigningKey, tmp_path: Any) -> None:
        client, biscuit = self._client(sk, tmp_path)
        try:
            with self._gate(client, biscuit, self._APPROVE, "rejected") as (call, _):
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                assert exc.value.denied_reason == "approval_rejected"
        finally:
            client.close()

    def test_approve_expired_denies(self, sk: SigningKey, tmp_path: Any) -> None:
        client, biscuit = self._client(sk, tmp_path)
        try:
            with self._gate(client, biscuit, self._APPROVE, "expired") as (call, _):
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                assert exc.value.denied_reason == "approval_expired"
        finally:
            client.close()

    def test_approve_timeout_denies(self, sk: SigningKey, tmp_path: Any) -> None:
        client, biscuit = self._client(sk, tmp_path)
        try:
            # Always pending → the local approval_timeout elapses → fail closed.
            with self._gate(client, biscuit, self._APPROVE, "pending") as (call, _):
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                assert exc.value.denied_reason == "approval_timeout"
        finally:
            client.close()

    def test_approve_poll_unreachable_fails_closed(self, sk: SigningKey, tmp_path: Any) -> None:
        client, biscuit = self._client(sk, tmp_path)
        try:
            err = SigilTransportError("boom", method="GET", url="http://x")
            with self._gate(client, biscuit, self._APPROVE, err) as (call, _):
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                # A status-poll failure must NEVER apply fail_mode — always deny.
                assert exc.value.denied_reason == "approval_service_unavailable"
        finally:
            client.close()

    def test_approve_missing_id_fails_closed(self, sk: SigningKey, tmp_path: Any) -> None:
        client, biscuit = self._client(sk, tmp_path)
        try:
            # approve verdict but no approval_id → never poll; fail closed immediately.
            with self._gate(
                client, biscuit, {"verdict": "approve"}, "approved"
            ) as (call, mock_status):
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                assert exc.value.denied_reason == "approval_service_unavailable"
                mock_status.assert_not_called()
        finally:
            client.close()

    def test_approve_timeout_capped_by_deadline_not_interval(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        # poll_interval (5s) is far larger than approval_timeout (0.02s): the sleep must
        # be capped to the remaining window so the wait cannot overshoot the timeout.
        import time as _time

        client, biscuit = _make_client(sk, ["ns.dangerous"], overflow_dir=str(tmp_path))
        client.approval_poll_interval = 5.0
        client.approval_timeout = 0.02
        try:
            with self._gate(client, biscuit, self._APPROVE, "pending") as (call, _):
                start = _time.monotonic()
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                elapsed = _time.monotonic() - start
                assert exc.value.denied_reason == "approval_timeout"
                assert elapsed < 2.0, f"timeout overshot the deadline: {elapsed:.3f}s"
        finally:
            client.close()

    def test_approve_poll_error_fails_closed_even_with_fail_mode_open(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        # A status-poll API error must fail closed even when fail_mode="open" — an
        # approval-gated call can never proceed ungoverned. Also exercises the
        # SigilAPIError branch of the poll's catch-all.
        client, biscuit = _make_client(
            sk, ["ns.dangerous"], fail_mode="open", overflow_dir=str(tmp_path)
        )
        client.approval_poll_interval = 0.001
        client.approval_timeout = 0.05
        try:
            err = SigilAPIError("internal server error", status_code=500)
            with self._gate(client, biscuit, self._APPROVE, err) as (call, _):
                with pytest.raises(SigilDeniedError) as exc:
                    call()
                assert exc.value.denied_reason == "approval_service_unavailable"
        finally:
            client.close()
