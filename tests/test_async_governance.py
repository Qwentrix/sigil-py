"""Async governance tests — C1/H3: async wrappers are generated correctly.

Covers:
- C1/H3: @instrumented_tool and @instrumented_llm produce real async wrappers
  (not sync wrappers) when decorated async functions are used.
- allow path: fn proceeds, outcome="allowed" event buffered
- out-of-scope-deny path: SigilDeniedError raised, no execution
- exception→error-event path: fn raises, outcome="error" event buffered
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (mirrors test_governance.py pattern)
# ──────────────────────────────────────────────────────────────────────────────

_KID = "async-test-kid"
_TENANT = "tenant-uuid-async"
_AGENT = "agent-uuid-async"
_FUTURE_TS = "2099-01-01T00:00:00Z"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_biscuit(sk: SigningKey, tools: list[str]) -> str:
    facts = [f'tool("{t}")' for t in tools] + [
        f'tenant("{_TENANT}")',
        f'agent("{_AGENT}")',
        'task("task-async")',
    ]
    payload = {
        "v": 1,
        "blocks": [
            {
                "facts": facts,
                "checks": [f'check if time($t), $t < "{_FUTURE_TS}"'],
                "rid": "rid-async",
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
        "grant_id": "g-async-001",
        "biscuit_token": biscuit_token,
        "revocation_id": "rev-async-001",
        "expires_at": _FUTURE_TS,
    }


def _make_client(
    sk: SigningKey,
    tools: list[str],
    fail_mode: str = "closed",
    overflow_dir: str | None = None,
) -> tuple[SigilClient, str]:
    biscuit = _make_biscuit(sk, tools)
    keyring = {_KID: bytes(sk.verify_key)}
    client = SigilClient(
        base_url="http://sigil-test:8120",
        internal_token="int-tok-secret",
        tenant_id=_TENANT,
        agent_id=_AGENT,
        service_account_id="sa-uuid-async",
        fail_mode=fail_mode,
        biscuit_keyring=keyring,
        overflow_dir=overflow_dir,
    )
    return client, biscuit


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def sk() -> SigningKey:
    return SigningKey.generate()


# ──────────────────────────────────────────────────────────────────────────────
# C1/H3 — @instrumented_tool wraps async functions as real coroutines
# ──────────────────────────────────────────────────────────────────────────────


class TestAsyncInstrumentedTool:
    async def test_async_tool_allow_proceeds_and_emits_event(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """C1/H3: async fn inside scope → awaitable, outcome='allowed' event emitted."""
        client, biscuit = _make_client(sk, ["ns.async_tool"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []

        @instrumented_tool("ns", "async_tool")
        async def async_tool(x: int) -> int:
            await asyncio.sleep(0)  # yields control — proves it's a real coroutine
            return x * 2

        # Verify the wrapper is itself a coroutine function (C1/H3 key assertion)
        assert inspect.iscoroutinefunction(
            async_tool
        ), "instrumented_tool must produce a coroutine function for async fns"

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
                client.task(["ns.async_tool"]) as _task,
            ):
                result = await async_tool(21)
                assert result == 42

            ev = next((e for e in buffered if e.get("tool_name") == "ns.async_tool"), None)
            assert ev is not None, "No event buffered for async_tool"
            assert ev["outcome"] == "allowed"
        finally:
            client.close()

    async def test_async_tool_out_of_scope_raises_denied(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """C1/H3: async fn outside token scope → SigilDeniedError, fn not called.

        I-4: also asserts that the local deny makes NO network call (preflight
        must not be invoked when local verify already denies).
        """
        client, biscuit = _make_client(sk, ["ns.other_tool"], overflow_dir=str(tmp_path))
        called = False

        @instrumented_tool("ns", "missing_tool")
        async def missing_tool() -> str:
            nonlocal called
            called = True
            return "should not run"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                # I-4: patch preflight to prove it is never called on a local deny.
                patch.object(client, "preflight") as mock_preflight,
                client.task(["ns.other_tool"]) as _task,
                pytest.raises(SigilDeniedError) as exc_info,
            ):
                await missing_tool()

            assert exc_info.value.denied_reason == "tool_not_in_scope"
            assert not called, "Async fn was called despite being out of scope"
            # I-4: local verify denied first — preflight must never have been called.
            mock_preflight.assert_not_called()
        finally:
            client.close()

    async def test_async_tool_exception_emits_error_event(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """H2 (async path): async fn raises → outcome='error' event buffered; exception re-raised."""  # noqa: E501
        client, biscuit = _make_client(sk, ["ns.async_boom"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []

        @instrumented_tool("ns", "async_boom")
        async def async_boom() -> str:
            await asyncio.sleep(0)
            raise RuntimeError("async boom!")

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
                client.task(["ns.async_boom"]) as _task,
                pytest.raises(RuntimeError, match="async boom!"),
            ):
                await async_boom()

            ev = next((e for e in buffered if e.get("tool_name") == "ns.async_boom"), None)
            assert ev is not None, "No error event emitted for async_boom"
            assert ev["outcome"] == "error"
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# C1/H3 — @instrumented_llm wraps async functions as real coroutines
# ──────────────────────────────────────────────────────────────────────────────


class TestAsyncInstrumentedLLM:
    async def test_async_llm_allow_with_dlp_redaction(self, sk: SigningKey, tmp_path: Any) -> None:
        """C1/H3 + I4: async LLM wrapper runs DLP (redact_safe), outcome='allowed', event emitted."""  # noqa: E501
        client, biscuit = _make_client(sk, ["llm.async_chat"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []

        @instrumented_llm("llm", "async_chat")
        async def async_chat(prompt: str) -> str:
            await asyncio.sleep(0)
            return "response"

        assert inspect.iscoroutinefunction(
            async_chat
        ), "instrumented_llm must produce a coroutine function for async fns"

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
                client.task(["llm.async_chat"]) as _task,
            ):
                result = await async_chat("my SSN is 123-45-6789")
                assert result == "response"

            ev = next((e for e in buffered if e.get("tool_name") == "llm.async_chat"), None)
            assert ev is not None, "No event buffered for async_chat"
            assert ev["outcome"] == "allowed"
            # DLP must have scrubbed the SSN in args_redacted
            assert "args_redacted" in ev
            redacted_str = json.dumps(ev["args_redacted"])
            assert "<PII:SSN>" in redacted_str, "SSN must be redacted in args_redacted"
            assert "123-45-6789" not in redacted_str, "Raw SSN must not appear in event"
        finally:
            client.close()

    async def test_async_llm_out_of_scope_raises_denied_with_redacted(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """C1/H3: async LLM fn outside scope → SigilDeniedError; event still has args_redacted."""
        client, biscuit = _make_client(sk, ["llm.other_model"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []

        @instrumented_llm("llm", "denied_async_llm")
        async def denied_async_llm(prompt: str) -> str:
            return "never"

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
                client.task(["llm.other_model"]) as _task,
                pytest.raises(SigilDeniedError) as exc_info,
            ):
                await denied_async_llm("secret SSN 987-65-4321")

            assert exc_info.value.denied_reason == "tool_not_in_scope"
            ev = next(
                (e for e in buffered if e.get("tool_name") == "llm.denied_async_llm"),
                None,
            )
            assert ev is not None
            assert "args_redacted" in ev
            redacted_str = json.dumps(ev["args_redacted"])
            assert "<PII:SSN>" in redacted_str
        finally:
            client.close()

    async def test_async_llm_exception_emits_error_event(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """H2 (async LLM path): fn raises → error event with args_redacted; exception re-raised."""
        client, biscuit = _make_client(sk, ["llm.async_explode"], overflow_dir=str(tmp_path))
        buffered: list[dict[str, Any]] = []

        @instrumented_llm("llm", "async_explode")
        async def async_explode(prompt: str) -> str:
            await asyncio.sleep(0)
            raise ValueError("async LLM down")

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
                client.task(["llm.async_explode"]) as _task,
                pytest.raises(ValueError, match="async LLM down"),
            ):
                await async_explode("user input")

            ev = next(
                (e for e in buffered if e.get("tool_name") == "llm.async_explode"),
                None,
            )
            assert ev is not None, "No error event emitted for async_explode"
            assert ev["outcome"] == "error"
            assert "args_redacted" in ev
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# I-5 — async preflight paths: M1 (unknown verdict) and H1 (unreachable)
# ──────────────────────────────────────────────────────────────────────────────


class TestAsyncPreflightPaths:
    """I-5: M1 (null/unknown verdict) and H1 (unreachable) via the async path."""

    async def test_async_unknown_verdict_raises_denied(self, sk: SigningKey, tmp_path: Any) -> None:
        """I-5 / M1 (async): preflight returns an unknown/empty verdict → SigilDeniedError
        with denied_reason='unknown_verdict'.  Fail-closed — tool is NOT executed."""
        client, biscuit = _make_client(sk, ["ns.high_tool"], overflow_dir=str(tmp_path))
        executed = False

        @instrumented_tool("ns", "high_tool", risk_tier="high")
        async def high_tool() -> str:
            nonlocal executed
            executed = True
            return "should not run"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                # M1: preflight returns an unrecognised verdict string.
                patch.object(
                    client,
                    "preflight",
                    return_value={"verdict": "UNKNOWN_VERDICT"},
                ),
                client.task(["ns.high_tool"]) as _task,
                pytest.raises(SigilDeniedError) as exc_info,
            ):
                await high_tool()

            assert exc_info.value.denied_reason == "unknown_verdict"
            assert not executed, "Tool must not execute on unknown verdict"
        finally:
            client.close()

    async def test_async_high_risk_unreachable_fail_closed(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """I-5 / H1 (async, closed): SigilAPIError 503 from preflight → fail-closed
        SigilUnreachableDeniedError + denial event written to overflow."""
        client, biscuit = _make_client(
            sk, ["ns.critical_tool"], overflow_dir=str(tmp_path), fail_mode="closed"
        )
        executed = False

        @instrumented_tool("ns", "critical_tool", risk_tier="critical")
        async def critical_tool() -> str:
            nonlocal executed
            executed = True
            return "should not run"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                # H1: preflight returns a 503 — treated as unreachable.
                patch.object(
                    client,
                    "preflight",
                    side_effect=SigilAPIError("503 service unavailable", status_code=503),
                ),
                client.task(["ns.critical_tool"]) as _task,
                pytest.raises(SigilUnreachableDeniedError) as exc_info,
            ):
                await critical_tool()

            assert exc_info.value.denied_reason == "sigil_unreachable"
            assert not executed, "Tool must not execute when fail-closed"

            # Denial event must have been written to the overflow directory.
            import os

            ndjson_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".ndjson")]
            assert ndjson_files, "Overflow file must be created on fail-closed denial"
        finally:
            client.close()

    async def test_async_high_risk_unreachable_fail_open(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """I-5 / H1 (async, open): SigilTransportError from preflight + fail_mode='open'
        → tool proceeds, audit event has fail_open=True."""
        client, biscuit = _make_client(
            sk, ["ns.open_tool"], overflow_dir=str(tmp_path), fail_mode="open"
        )
        buffered: list[dict[str, Any]] = []

        @instrumented_tool("ns", "open_tool", risk_tier="high")
        async def open_tool() -> str:
            return "allowed in fail-open mode"

        issue_resp = _issue_resp(biscuit)

        try:
            with (
                patch.object(
                    client._session,
                    "post",
                    return_value=_mock_response(201, issue_resp),
                ),
                # H1: network failure during preflight.
                patch.object(
                    client,
                    "preflight",
                    side_effect=SigilTransportError(
                        "connection refused", method="POST", url="http://sigil-test:8120/preflight"
                    ),
                ),
                patch.object(
                    client,
                    "log_batch",
                    wraps=lambda events: (buffered.extend(events) or {"accepted": len(events)}),
                ),
                client.task(["ns.open_tool"]) as _task,
            ):
                result = await open_tool()
                assert result == "allowed in fail-open mode"

            ev = next((e for e in buffered if e.get("tool_name") == "ns.open_tool"), None)
            assert ev is not None, "Audit event must be emitted in fail-open path"
            assert ev["outcome"] == "allowed"
            assert ev.get("fail_open") is True, "fail_open flag must be set in event"
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# ENT-81/SG-4 — approval gate on the async path (offloaded via run_in_executor)
# ──────────────────────────────────────────────────────────────────────────────


class TestAsyncApprovalGate:
    def _client(self, sk: SigningKey, tmp_path: Any) -> tuple[SigilClient, str]:
        client, biscuit = _make_client(sk, ["ns.danger"], overflow_dir=str(tmp_path))
        client.approval_poll_interval = 0.001
        client.approval_timeout = 0.05
        return client, biscuit

    async def test_async_approve_approved_executes(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        client, biscuit = self._client(sk, tmp_path)

        @instrumented_tool("ns", "danger", risk_tier="high")
        async def danger() -> str:
            await asyncio.sleep(0)
            return "ok"

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(
                    client, "preflight", return_value={"verdict": "approve", "approval_id": "ap-1"}
                ),
                patch.object(client, "approval_status", return_value="approved"),
                patch.object(
                    client, "redeem_approval",
                    return_value={"revocation_id": "grant-ok", "one_shot_token": "t"},
                ),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                client.task(["ns.danger"]),
            ):
                assert await danger() == "ok"
        finally:
            client.close()

    async def test_async_approve_rejected_denies(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        client, biscuit = self._client(sk, tmp_path)

        @instrumented_tool("ns", "danger", risk_tier="high")
        async def danger() -> str:
            await asyncio.sleep(0)
            return "ok"

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(
                    client, "preflight", return_value={"verdict": "approve", "approval_id": "ap-1"}
                ),
                patch.object(client, "approval_status", return_value="rejected"),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                client.task(["ns.danger"]),
            ):
                with pytest.raises(SigilDeniedError) as exc:
                    await danger()
                assert exc.value.denied_reason == "approval_rejected"
        finally:
            client.close()

    async def test_async_approve_redeem_replayed_denies(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """ENT-82 (async): a 409 from redeem fails closed with approval_replayed."""
        client, biscuit = self._client(sk, tmp_path)

        @instrumented_tool("ns", "danger", risk_tier="high")
        async def danger() -> str:
            return "ok"

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(
                    client, "preflight", return_value={"verdict": "approve", "approval_id": "ap-1"}
                ),
                patch.object(client, "approval_status", return_value="approved"),
                patch.object(
                    client, "redeem_approval",
                    side_effect=SigilAPIError("already redeemed", status_code=409),
                ),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                client.task(["ns.danger"]),
            ):
                with pytest.raises(SigilDeniedError) as exc:
                    await danger()
                assert exc.value.denied_reason == "approval_replayed"
        finally:
            client.close()

    async def test_async_approve_redeem_unavailable_denies(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """ENT-82 (async): a transport failure on redeem fails closed with
        approval_token_unavailable — the propagation through run_in_executor still denies."""
        client, biscuit = self._client(sk, tmp_path)

        @instrumented_tool("ns", "danger", risk_tier="high")
        async def danger() -> str:
            return "ok"

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(
                    client, "preflight", return_value={"verdict": "approve", "approval_id": "ap-1"}
                ),
                patch.object(client, "approval_status", return_value="approved"),
                patch.object(
                    client, "redeem_approval",
                    side_effect=SigilTransportError("down", method="POST", url="http://x/"),
                ),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                client.task(["ns.danger"]),
            ):
                with pytest.raises(SigilDeniedError) as exc:
                    await danger()
                assert exc.value.denied_reason == "approval_token_unavailable"
        finally:
            client.close()

    async def test_async_approve_polls_natively_via_asyncio_sleep(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """#311: the async approve path waits with ``asyncio.sleep`` on the event loop
        (native async) between polls and never calls the blocking ``time.sleep`` — so
        the (up to approval_timeout) idle wait cannot pin an executor-pool thread.
        """
        client, biscuit = self._client(sk, tmp_path)

        @instrumented_tool("ns", "danger", risk_tier="high")
        async def danger() -> str:
            return "ok"

        # pending first, approved second → forces exactly one inter-poll wait.
        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(
                    client, "preflight", return_value={"verdict": "approve", "approval_id": "ap-1"}
                ),
                patch.object(client, "approval_status", side_effect=["pending", "approved"]),
                patch.object(
                    client, "redeem_approval",
                    return_value={"revocation_id": "grant-ok", "one_shot_token": "t"},
                ),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                patch("sigil.decorators.asyncio.sleep", new=AsyncMock()) as mock_asleep,
                patch("sigil.decorators.time.sleep") as mock_tsleep,
                client.task(["ns.danger"]),
            ):
                assert await danger() == "ok"

            # The inter-poll wait was awaited on the loop at the configured interval …
            assert any(
                call.args == (client.approval_poll_interval,)
                for call in mock_asleep.await_args_list
            ), (
                f"expected an awaited asyncio.sleep({client.approval_poll_interval}); "
                f"got {mock_asleep.await_args_list}"
            )
            # … and the blocking time.sleep was never used on the async path (#311).
            mock_tsleep.assert_not_called()
        finally:
            client.close()

    async def test_async_approve_timeout_denies_via_native_poll(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        """#311: a never-resolving approval on the async path fails closed with
        ``approval_timeout`` once the local ``approval_timeout`` deadline elapses,
        driven entirely by the native-async poller."""
        client, biscuit = self._client(sk, tmp_path)

        @instrumented_tool("ns", "danger", risk_tier="high")
        async def danger() -> str:
            return "ok"

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(
                    client, "preflight", return_value={"verdict": "approve", "approval_id": "ap-1"}
                ),
                patch.object(client, "approval_status", return_value="pending"),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                client.task(["ns.danger"]),
            ):
                with pytest.raises(SigilDeniedError) as exc:
                    await danger()
                assert exc.value.denied_reason == "approval_timeout"
        finally:
            client.close()
