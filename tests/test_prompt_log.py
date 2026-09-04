"""SG-9 SP-1 (ENT-91) — Task-Replay prompt-log lane.

Covers the additive prompt-log producer added to the SDK:

- client.log_prompt transport (success / transport error / non-202 / non-JSON).
- _buffer routing: _partition_events, _send_prompt_log, and the _Flusher splitting a
  mixed page into log_batch (tool) + log_prompt (prompt), with overflow on transport failure.
- _overflow replay routing: a file holding both kinds replays each to the right endpoint.
- @instrumented_llm decorator emit: buffers a prompt-log event on success (correct hashes,
  model resolution, token counts, latency, DLP-redacted samples); @instrumented_tool does NOT;
  the error path emits no prompt-log; async variant emits; model-resolution precedence.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from nacl.signing import SigningKey

from sigil._buffer import (
    _KIND_KEY,
    _PROMPT_LOG_KIND,
    _Flusher,
    _LogBuffer,
    _partition_events,
    _send_prompt_log,
)
from sigil._overflow import _OverflowWriter
from sigil.client import SigilClient
from sigil.decorators import instrumented_llm, instrumented_tool
from sigil.errors import SigilAPIError, SigilTransportError

# ──────────────────────────────────────────────────────────────────────────────
# Token / client helpers (mirror tests/test_governance.py)
# ──────────────────────────────────────────────────────────────────────────────

_KID = "test-kid"
_TENANT = "tenant-uuid-test"
_AGENT = "agent-uuid-test"
_FUTURE_TS = "2099-01-01T00:00:00Z"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_biscuit(sk: SigningKey, tools: list[str]) -> str:
    facts = [f'tool("{t}")' for t in tools] + [
        f'tenant("{_TENANT}")',
        f'agent("{_AGENT}")',
        'task("task-from-sigil")',
    ]
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


def _mock_response(status: int, body: Any) -> MagicMock:
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


def _make_client(sk: SigningKey, overflow_dir: str) -> SigilClient:
    return SigilClient(
        base_url="http://sigil-test:8120",
        internal_token="int-tok-secret",
        tenant_id=_TENANT,
        agent_id=_AGENT,
        service_account_id="sa-uuid",
        fail_mode="closed",
        biscuit_keyring={_KID: bytes(sk.verify_key)},
        overflow_dir=overflow_dir,
    )


# A representative OpenAI-style result: text under choices[0].message.content, usage, model.
_LLM_RESULT = {
    "choices": [{"message": {"content": "hi there, here is the summary"}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    "model": "gpt-4o-mini",
}
_RESPONSE_TEXT = "hi there, here is the summary"


@pytest.fixture()
def sk() -> SigningKey:
    return SigningKey.generate()


# ──────────────────────────────────────────────────────────────────────────────
# client.log_prompt transport
# ──────────────────────────────────────────────────────────────────────────────


class TestClientLogPrompt:
    def _client(self, sk: SigningKey, tmp_path: Any) -> SigilClient:
        return _make_client(sk, str(tmp_path))

    def test_success_posts_to_per_task_endpoint(self, sk: SigningKey, tmp_path: Any) -> None:
        client = self._client(sk, tmp_path)
        try:
            with patch.object(
                client._session, "post", return_value=_mock_response(202, {"status": "accepted"})
            ) as mock_post:
                out = client.log_prompt(
                    "task-123",
                    "agent-456",
                    prompt_hash="a" * 64,
                    prompt_redacted={"args": ["x"]},
                    response_hash="b" * 64,
                    response_sampled={"text": "hi"},
                    model="gpt-4o",
                    model_provider="openai",
                    token_count_input=3,
                    token_count_output=4,
                    latency_ms=12,
                )
            assert out == {"status": "accepted"}
            url = mock_post.call_args.args[0]
            assert url.endswith("/internal/v1/sigil/tasks/task-123/log-prompt")
            body = mock_post.call_args.kwargs["json"]
            assert body["agent_id"] == "agent-456"
            assert body["model"] == "gpt-4o" and body["model_provider"] == "openai"
            assert body["token_count_input"] == 3 and body["latency_ms"] == 12
        finally:
            client.close()

    def test_transport_error_raises(self, sk: SigningKey, tmp_path: Any) -> None:
        client = self._client(sk, tmp_path)
        try:
            with (
                patch.object(client._session, "post", side_effect=OSError("boom")),
                pytest.raises(SigilTransportError),
            ):
                client.log_prompt(
                    "t",
                    "a",
                    prompt_hash="x",
                    prompt_redacted={},
                    response_hash="y",
                    response_sampled=None,
                    model="m",
                    model_provider="p",
                )
        finally:
            client.close()

    def test_non_202_raises_api_error(self, sk: SigningKey, tmp_path: Any) -> None:
        client = self._client(sk, tmp_path)
        try:
            with (
                patch.object(client._session, "post", return_value=_mock_response(500, {})),
                pytest.raises(SigilAPIError),
            ):
                client.log_prompt(
                    "t",
                    "a",
                    prompt_hash="x",
                    prompt_redacted={},
                    response_hash="y",
                    response_sampled=None,
                    model="m",
                    model_provider="p",
                )
        finally:
            client.close()

    def test_non_json_body_raises_api_error(self, sk: SigningKey, tmp_path: Any) -> None:
        client = self._client(sk, tmp_path)
        try:
            resp = _mock_response(202, None)
            resp.json.side_effect = ValueError("not json")
            with (
                patch.object(client._session, "post", return_value=resp),
                pytest.raises(SigilAPIError),
            ):
                client.log_prompt(
                    "t",
                    "a",
                    prompt_hash="x",
                    prompt_redacted={},
                    response_hash="y",
                    response_sampled=None,
                    model="m",
                    model_provider="p",
                )
        finally:
            client.close()


# ──────────────────────────────────────────────────────────────────────────────
# _buffer routing
# ──────────────────────────────────────────────────────────────────────────────


def _prompt_event(task_id: str = "t1") -> dict[str, Any]:
    return {
        _KIND_KEY: _PROMPT_LOG_KIND,
        "task_id": task_id,
        "agent_id": "a1",
        "prompt_hash": "p" * 64,
        "prompt_redacted": {"args": ["q"]},
        "response_hash": "r" * 64,
        "response_sampled": {"text": "resp"},
        "model": "gpt-4o",
        "model_provider": "openai",
        "token_count_input": 5,
        "token_count_output": 6,
        "latency_ms": 9,
    }


class TestBufferRouting:
    def test_partition_splits_by_kind(self) -> None:
        tool_ev = {"tool_name": "ns.x", "outcome": "allowed"}
        tool, prompts = _partition_events([tool_ev, _prompt_event(), tool_ev])
        assert tool == [tool_ev, tool_ev]
        assert len(prompts) == 1 and prompts[0][_KIND_KEY] == _PROMPT_LOG_KIND

    def test_send_prompt_log_maps_fields(self) -> None:
        client = MagicMock()
        _send_prompt_log(client, _prompt_event("task-9"))
        client.log_prompt.assert_called_once()
        args, kwargs = client.log_prompt.call_args
        assert args == ("task-9", "a1")
        assert kwargs["prompt_hash"] == "p" * 64
        assert kwargs["model"] == "gpt-4o" and kwargs["model_provider"] == "openai"
        assert kwargs["token_count_input"] == 5 and kwargs["latency_ms"] == 9

    def test_flusher_routes_each_kind(self) -> None:
        buf = _LogBuffer()
        buf.push({"tool_name": "ns.x", "outcome": "allowed"})
        buf.push(_prompt_event())
        client = MagicMock()
        overflow = MagicMock()
        _Flusher(buf, client, overflow)._do_flush()
        # tool event → log_batch (exactly the one tool event); prompt → log_prompt
        client.log_batch.assert_called_once()
        assert client.log_batch.call_args.args[0] == [{"tool_name": "ns.x", "outcome": "allowed"}]
        client.log_prompt.assert_called_once()
        assert client.log_prompt.call_args.args[0] == "t1"
        overflow.write.assert_not_called()

    def test_flusher_prompt_transport_failure_persists_to_overflow(self) -> None:
        buf = _LogBuffer()
        pe = _prompt_event()
        buf.push(pe)
        client = MagicMock()
        client.log_prompt.side_effect = SigilTransportError("down", method="POST", url="u")
        overflow = MagicMock()
        _Flusher(buf, client, overflow)._do_flush()
        overflow.write.assert_called_once_with(pe)
        # transport-down page must NOT trigger an opportunistic replay
        overflow.replay.assert_not_called()

    def test_flusher_prompt_api_error_is_dropped_not_persisted(self) -> None:
        buf = _LogBuffer()
        buf.push(_prompt_event())
        client = MagicMock()
        client.log_prompt.side_effect = SigilAPIError("bad", status_code=400)
        overflow = MagicMock()
        _Flusher(buf, client, overflow)._do_flush()
        overflow.write.assert_not_called()  # non-retryable → dropped

    def test_flusher_stops_network_after_prompt_transport_down(self) -> None:
        # A page of 3 prompt-logs where the first hits a transport error: the remaining two
        # must be persisted to overflow WITHOUT further network attempts (no per-event stall).
        buf = _LogBuffer()
        pe1, pe2, pe3 = _prompt_event("t1"), _prompt_event("t2"), _prompt_event("t3")
        for pe in (pe1, pe2, pe3):
            buf.push(pe)
        client = MagicMock()
        client.log_prompt.side_effect = SigilTransportError("down", method="POST", url="u")
        overflow = MagicMock()
        _Flusher(buf, client, overflow)._do_flush()
        assert client.log_prompt.call_count == 1  # only the first attempts the network
        assert overflow.write.call_count == 3  # all three persisted for later replay
        overflow.replay.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# _overflow replay routing
# ──────────────────────────────────────────────────────────────────────────────


class TestOverflowReplayRouting:
    def test_replay_routes_both_kinds_and_deletes_file(self, tmp_path: Any) -> None:
        w = _OverflowWriter(str(tmp_path), _AGENT)
        w.write({"tool_name": "ns.x", "outcome": "allowed"})
        w.write(_prompt_event("task-of"))
        client = MagicMock()
        w.replay(client)
        client.log_batch.assert_called_once()
        assert client.log_batch.call_args.args[0][0]["tool_name"] == "ns.x"
        client.log_prompt.assert_called_once()
        assert client.log_prompt.call_args.args[0] == "task-of"
        # file drained + removed
        assert not list(tmp_path.glob("*.ndjson"))

    def test_replay_leaves_file_when_prompt_send_fails(self, tmp_path: Any) -> None:
        w = _OverflowWriter(str(tmp_path), _AGENT)
        w.write(_prompt_event())
        client = MagicMock()
        client.log_prompt.side_effect = SigilTransportError("down", method="POST", url="u")
        w.replay(client)
        assert list(tmp_path.glob("*.ndjson"))  # retained for next attempt


# ──────────────────────────────────────────────────────────────────────────────
# @instrumented_llm decorator emit
# ──────────────────────────────────────────────────────────────────────────────


def _captured_pushes(client: SigilClient) -> list[dict[str, Any]]:
    """Install a spy on the ring buffer's push and return the list it fills."""
    seen: list[dict[str, Any]] = []
    orig = client._log_buffer.push

    def _spy(ev: dict[str, Any]) -> None:
        seen.append(ev)
        orig(ev)

    client._log_buffer.push = _spy  # type: ignore[method-assign]
    return seen


def _prompt_logs(seen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in seen if e.get(_KIND_KEY) == _PROMPT_LOG_KIND]


class TestInstrumentedLlmEmit:
    def _run(self, client: SigilClient, biscuit: str, fn: Any, *call: Any, **kw: Any) -> Any:
        with (
            patch.object(
                client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
            ),
            patch.object(client, "log_batch", return_value={"accepted": 1}),
            patch.object(client, "log_prompt", return_value={"status": "accepted"}),
            client.task(["openai.chat"]) as _task,
        ):
            return fn(*call, **kw)

    def test_llm_success_buffers_prompt_log(self, sk: SigningKey, tmp_path: Any) -> None:
        client = _make_client(sk, str(tmp_path))
        biscuit = _make_biscuit(sk, ["openai.chat"])
        seen = _captured_pushes(client)

        @instrumented_llm("openai", "chat")
        def call_llm(prompt: str) -> dict[str, Any]:
            return _LLM_RESULT

        try:
            self._run(client, biscuit, call_llm, "summarize the Q4 report")
        finally:
            client.close()

        logs = _prompt_logs(seen)
        assert len(logs) == 1, "exactly one prompt-log event should be buffered"
        ev = logs[0]
        assert ev["task_id"] and ev["agent_id"] == _AGENT
        assert ev["prompt_hash"] == hashlib.sha256(b"summarize the Q4 report").hexdigest()
        assert ev["response_hash"] == hashlib.sha256(_RESPONSE_TEXT.encode()).hexdigest()
        assert ev["model"] == "gpt-4o-mini"  # from result.model (no explicit/kwarg)
        assert ev["model_provider"] == "openai"
        assert ev["token_count_input"] == 11 and ev["token_count_output"] == 7
        assert ev["latency_ms"] >= 0
        assert ev["response_sampled"]["text"] == _RESPONSE_TEXT  # no PII → unchanged

    def test_explicit_model_wins(self, sk: SigningKey, tmp_path: Any) -> None:
        client = _make_client(sk, str(tmp_path))
        biscuit = _make_biscuit(sk, ["openai.chat"])
        seen = _captured_pushes(client)

        @instrumented_llm("openai", "chat", model="claude-sonnet-x")
        def call_llm(prompt: str) -> dict[str, Any]:
            return _LLM_RESULT

        try:
            self._run(client, biscuit, call_llm, "hello")
        finally:
            client.close()
        assert _prompt_logs(seen)[0]["model"] == "claude-sonnet-x"

    def test_kwarg_model_used_when_no_explicit(self, sk: SigningKey, tmp_path: Any) -> None:
        client = _make_client(sk, str(tmp_path))
        biscuit = _make_biscuit(sk, ["openai.chat"])
        seen = _captured_pushes(client)

        @instrumented_llm("openai", "chat")
        def call_llm(prompt: str, model: str) -> str:
            return "plain string reply"

        try:
            self._run(client, biscuit, call_llm, "hi", model="gpt-4o-from-kwarg")
        finally:
            client.close()
        ev = _prompt_logs(seen)[0]
        assert ev["model"] == "gpt-4o-from-kwarg"
        # bare-str result path still hashes the response
        assert ev["response_hash"] == hashlib.sha256(b"plain string reply").hexdigest()

    def test_pii_in_response_is_redacted_in_sample(self, sk: SigningKey, tmp_path: Any) -> None:
        client = _make_client(sk, str(tmp_path))
        biscuit = _make_biscuit(sk, ["openai.chat"])
        seen = _captured_pushes(client)

        @instrumented_llm("openai", "chat")
        def call_llm(prompt: str) -> str:
            return "contact me at jane.doe@example.com"

        try:
            self._run(client, biscuit, call_llm, "hi")
        finally:
            client.close()
        sample = _prompt_logs(seen)[0]["response_sampled"]["text"]
        assert "jane.doe@example.com" not in sample and "<PII:EMAIL>" in sample

    def test_error_path_emits_no_prompt_log(self, sk: SigningKey, tmp_path: Any) -> None:
        client = _make_client(sk, str(tmp_path))
        biscuit = _make_biscuit(sk, ["openai.chat"])
        seen = _captured_pushes(client)

        @instrumented_llm("openai", "chat")
        def call_llm(prompt: str) -> str:
            raise RuntimeError("model down")

        try:
            with pytest.raises(RuntimeError):
                self._run(client, biscuit, call_llm, "hi")
        finally:
            client.close()
        # no prompt-log (needs a real response), but the tool-invocation error event IS present
        assert _prompt_logs(seen) == []
        assert any(e.get("outcome") == "error" for e in seen)

    def test_instrumented_tool_emits_no_prompt_log(self, sk: SigningKey, tmp_path: Any) -> None:
        client = _make_client(sk, str(tmp_path))
        biscuit = _make_biscuit(sk, ["zep.search"])
        seen = _captured_pushes(client)

        @instrumented_tool("zep", "search")
        def search(q: str) -> list[str]:
            return ["a"]

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                client.task(["zep.search"]) as _task,
            ):
                search("q")
        finally:
            client.close()
        assert _prompt_logs(seen) == []


class TestInstrumentedLlmEmitAsync:
    @pytest.mark.asyncio
    async def test_async_llm_success_buffers_prompt_log(
        self, sk: SigningKey, tmp_path: Any
    ) -> None:
        client = _make_client(sk, str(tmp_path))
        biscuit = _make_biscuit(sk, ["openai.chat"])
        seen = _captured_pushes(client)

        @instrumented_llm("openai", "chat")
        async def call_llm(prompt: str) -> dict[str, Any]:
            return _LLM_RESULT

        try:
            with (
                patch.object(
                    client._session, "post", return_value=_mock_response(201, _issue_resp(biscuit))
                ),
                patch.object(client, "log_batch", return_value={"accepted": 1}),
                patch.object(client, "log_prompt", return_value={"status": "accepted"}),
                client.task(["openai.chat"]) as _task,
            ):
                await call_llm("summarize the Q4 report")
        finally:
            client.close()
        logs = _prompt_logs(seen)
        assert len(logs) == 1
        assert logs[0]["prompt_hash"] == hashlib.sha256(b"summarize the Q4 report").hexdigest()
