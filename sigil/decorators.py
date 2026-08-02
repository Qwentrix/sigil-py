"""Decorator helpers for instrumenting AI agent tools and LLM calls.

Each call through an instrumented function runs the full governance flow:

1. Require an active ``SigilTaskContext`` (via contextvar).
2. **Revocation check** — consult the in-memory kill-switch registry; no network.
3. **Local verify** — ``verify_local(biscuit_token, keyring, required_tool, ...)``
   in <1 ms with no network.  Denial → ``SigilDeniedError`` with the verify
   reason (e.g. ``tool_not_in_scope``).
4. **Preflight** (``risk_tier`` in ``{"high", "critical"}`` only) — HTTP call to
   sigil-core.  Unreachable → fail-mode logic (closed → ``SigilUnreachableDeniedError``
   + overflow write; open → allow + ``fail_open=True`` banner event).
5. Execute the wrapped function (sync or async — detected at decoration time via
   ``inspect.iscoroutinefunction``).
6. Buffer an audit event (``args_hash`` over UNREDACTED args; ``args_redacted``
   for ``@instrumented_llm`` calls).  Emitted in a ``try/finally`` so that
   exceptions from ``fn`` produce an ``outcome="error"`` event instead of
   being silently dropped.

Shared governance logic (steps 2–4) is factored into ``_governance_check`` so
that sync and async wrappers do not duplicate it.

``@instrumented_llm`` is identical to ``@instrumented_tool`` with the addition
of DLP redaction (``redact_safe(args)``) included in the buffered event.  The
``args_hash`` is always computed over the ORIGINAL, unredacted args.

Example::

    client = SigilClient(...)

    @instrumented_tool("zep", "search", risk_tier="low")
    def search_kb(query: str) -> list[dict]:
        ...

    @instrumented_llm("openai", "chat", risk_tier="low")
    def call_llm(prompt: str) -> str:
        ...

    with client.task(["zep.search", "openai.chat"]) as task:
        results = search_kb("Q4 revenue")
        answer = call_llm("Summarise the following: ...")
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import logging
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from sigil._buffer import _KIND_KEY, _PROMPT_LOG_KIND
from sigil._context import _current_task
from sigil.errors import (
    TERMINAL_QUARANTINE_REASONS,
    AgentQuarantinedError,
    SigilAPIError,
    SigilDeniedError,
    SigilTransportError,
    SigilUnreachableDeniedError,
)
from sigil.redaction import args_hash, canonical_json, redact_safe
from sigil.verify import verify_local

if TYPE_CHECKING:
    from sigil.client import SigilClient, SigilTaskContext

F = TypeVar("F", bound=Callable[..., Any])

_log = logging.getLogger("sigil")

_HIGH_RISK_TIERS: frozenset[str] = frozenset({"high", "critical"})

# F6: valid risk_tier values — validated at decoration time so a typo ("medium",
# "HIGH") fails loudly at import rather than silently skipping preflight.
_VALID_RISK_TIERS: frozenset[str] = frozenset({"low", "med", "high", "critical"})


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _raw_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Canonical args representation used for hashing and redaction."""
    return {"args": list(args), "kwargs": kwargs}


def _extract_prompt(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Extract the prompt text from the call arguments of an instrumented LLM call.

    Handles four common LLM SDK calling conventions without ever raising:

    (a) Positional string — ``fn("hello")`` → ``args[0]`` is a ``str``.
    (b) Positional messages list — ``fn([{"role": "user", "content": "..."}])``
        → join the ``"content"`` fields of all dicts that have one.
    (c) Named scalar kwarg — ``fn(prompt="…")`` / ``fn(input="…")`` /
        ``fn(content="…")`` → return the first matching str kwarg.
    (d) Named messages kwarg — ``fn(messages=[{"role": "user", "content": "…"}])``
        → join the ``"content"`` fields (same logic as (b)).

    Returns ``""`` for any unrecognised shape so entropy gracefully falls back to
    0.0 rather than crashing the governed call.

    Fully typed and ruff/mypy-strict compatible.
    """
    try:
        # (a) positional string
        if args and isinstance(args[0], str):
            return args[0]
        # (b) positional messages list
        if args and isinstance(args[0], list):
            parts: list[str] = []
            for msg in args[0]:
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str):
                        parts.append(c)
            return "\n".join(parts)
        # (c) named scalar kwarg: prompt | input | content
        for key in ("prompt", "input", "content"):
            val = kwargs.get(key)
            if isinstance(val, str):
                return val
        # (d) named messages kwarg
        msgs = kwargs.get("messages")
        if isinstance(msgs, list):
            parts = []
            for msg in msgs:
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str):
                        parts.append(c)
            return "\n".join(parts)
    except Exception:  # noqa: BLE001
        pass
    return ""


def _shannon_entropy(text: str) -> float:
    """Return base-2 Shannon entropy over the character frequency of *text*.

    Empty or all-whitespace input returns 0.0.  The result is in bits (log base
    2), which is the conventional unit for prompt-entropy signalling.

    Example::

        _shannon_entropy("aaaa")   # → 0.0  (single symbol)
        _shannon_entropy("abcd")   # → 2.0  (four equally-likely symbols)
    """
    stripped = text.strip()
    if not stripped:
        return 0.0
    length = len(stripped)
    freq: dict[str, int] = {}
    for ch in stripped:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _make_event(
    agent_id: str,
    task_id: str,
    tool_fqn: str,
    namespace: str,
    ah: str,
    latency_ms: int,
    outcome: str,
    risk_tier: str,
    *,
    denied_reason: str | None = None,
    args_redacted: Any = None,
    fail_open: bool = False,
    approval_grant_id: str | None = None,
    prompt_entropy: float = 0.0,
) -> dict[str, Any]:
    """Build an audit event dict conforming to docs/protocol.md §4.1."""
    ev: dict[str, Any] = {
        "agent_id": agent_id,
        "task_id": task_id,
        "tool_name": tool_fqn,
        "tool_namespace": namespace,
        "args_hash": ah,
        "latency_ms": latency_ms,
        "outcome": outcome,
        "risk_tier": risk_tier,
        "prompt_entropy": prompt_entropy,
    }
    if denied_reason is not None:
        ev["denied_reason"] = denied_reason
    if args_redacted is not None:
        ev["args_redacted"] = args_redacted
    if fail_open:
        ev["fail_open"] = True
    if approval_grant_id:
        # ENT-82: the revocation_id of the one-shot single-use grant minted when the
        # human approval was redeemed — cryptographic proof this call ran under a fresh,
        # tool-scoped, human-approved grant (not the broad task token).
        ev["approval_grant_id"] = approval_grant_id
    return ev


def _require_task(tool_fqn: str) -> SigilTaskContext:
    """Return the active SigilTaskContext or raise a clear SDK usage error.

    Absence of a task context is an SDK usage error (not an agent governance
    denial), but we raise ``SigilDeniedError`` with ``denied_reason="no_task"``
    so that callers have a uniform error type to handle.
    """
    task: SigilTaskContext | None = _current_task.get()
    if task is None:
        raise SigilDeniedError(
            f"No active SigilTaskContext — '{tool_fqn}' must be called inside "
            "'with client.task([...]) as task:'",
            denied_reason="no_task",
            tool_name=tool_fqn,
            task_id="",
        )
    return task


def _handle_preflight_unreachable(
    task: SigilTaskContext,
    tool_fqn: str,
    namespace: str,
    ah: str,
    risk_tier: str,
    agent_id: str,
    task_id: str,
    exc: Exception,
    args_redacted: Any,
) -> bool:
    """Handle a transport or API error from preflight according to fail-mode.

    Both ``SigilTransportError`` (network failure) and ``SigilAPIError``
    (non-200 response such as 503/429/400/500) are treated identically:
    apply the configured fail-mode.

    Returns:
        True  — fail_mode="open": caller should proceed with execution.
        Never returns False — raises instead if fail_mode="closed".

    Raises:
        SigilUnreachableDeniedError: when fail_mode="closed".
    """
    client: SigilClient = task._client

    if client.fail_mode == "closed":
        ev = _make_event(
            agent_id,
            task_id,
            tool_fqn,
            namespace,
            ah,
            0,
            "denied",
            risk_tier,
            denied_reason="sigil_unreachable",
            args_redacted=args_redacted,
        )
        client._overflow.write(ev)
        raise SigilUnreachableDeniedError(
            f"sigil-core unreachable during preflight for '{tool_fqn}' " f"(fail_mode=closed)",
            denied_reason="sigil_unreachable",
            tool_name=tool_fqn,
            task_id=task_id,
        ) from exc

    # fail_mode="open" — must never be default; warn loudly.
    _log.warning(
        "sigil FAIL-OPEN: sigil-core preflight error for '%s' "
        "(task_id=%s, error=%s). Tool call is being ALLOWED. "
        "Do NOT use fail_mode='open' in production.",
        tool_fqn,
        task_id,
        exc,
    )
    return True


# Terminal approval statuses returned by intelligent-automation (via sigil-core's
# status proxy). Anything else (e.g. "pending") means keep polling.
_APPROVAL_TERMINAL: frozenset[str] = frozenset({"approved", "rejected", "expired"})


def _approval_poll_once(
    client: SigilClient, approval_id: str, deadline: float
) -> tuple[str | None, float]:
    """One iteration of the approval poll, performing NO sleeping.

    Returns ``(result, sleep_seconds)``:

    * ``result`` — a terminal outcome (``"approved"`` / ``"rejected"`` /
      ``"expired"`` / ``"timeout"`` / ``"unavailable"``) when polling should stop,
      otherwise ``None`` to keep waiting.
    * ``sleep_seconds`` — how long to wait before the next iteration (only
      meaningful when ``result is None``). Capped to the time remaining before
      *deadline* so a ``poll_interval`` larger than the remaining window cannot push
      the actual wait past ``approval_timeout`` (#307).

    Isolating a single blocking ``approval_status`` call lets the native-async
    poller offload *only that call* to a thread and ``await asyncio.sleep`` between
    iterations, instead of pinning an executor thread for the whole approval window
    (#311). ``deadline`` uses ``time.monotonic()`` (process-wide), so it stays
    consistent whether this runs on the event loop or in an executor thread.
    """
    try:
        status = client.approval_status(approval_id)
    except Exception:  # noqa: BLE001
        # Do NOT apply fail_mode here — an approval-gated call must never proceed
        # ungoverned when the decision cannot be confirmed. ANY error (transport,
        # API, or unexpected) fails closed.
        return "unavailable", 0.0
    if status in _APPROVAL_TERMINAL:
        return status, 0.0
    # "pending" (or an unrecognised non-terminal value) — keep waiting.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "timeout", 0.0
    return None, min(client.approval_poll_interval, remaining)


def _poll_approval(client: SigilClient, approval_id: str) -> str:
    """Block-poll a pending ENT-81 approval until it resolves, times out, or the
    status endpoint becomes unreachable. Synchronous — used by the blocking
    wrappers; see :func:`_poll_approval_async` for the native-async variant.

    Returns one of:
      - ``"approved"``    — a sigil_approver approved; the caller may proceed.
      - ``"rejected"``    — a sigil_approver rejected.
      - ``"expired"``     — the server marked the approval expired.
      - ``"timeout"``     — the local ``approval_timeout`` elapsed with no decision.
      - ``"unavailable"`` — a status poll failed (transport / API error).

    Every non-``approved`` outcome is a FAIL-CLOSED denial at the call site. The
    local ``approval_timeout`` is the authoritative bound; the server-sent
    ``expires_at`` is advisory (#307), so clock skew between the SDK host and
    sigil-core cannot extend the wait beyond the configured timeout.
    """
    deadline = time.monotonic() + client.approval_timeout
    while True:
        result, sleep_for = _approval_poll_once(client, approval_id, deadline)
        if result is not None:
            return result
        time.sleep(sleep_for)


async def _poll_approval_async(client: SigilClient, approval_id: str) -> str:
    """Native-async variant of :func:`_poll_approval` for ``async def`` tools (#311).

    Offloads only the individual blocking ``approval_status`` HTTP call to the
    default executor (each is a single short round-trip) and ``await``s
    ``asyncio.sleep`` between polls. The executor thread is therefore held only for
    the brief status call, never for the (up to ``approval_timeout``, default 300s)
    idle wait — so a burst of concurrently-awaiting approvals cannot exhaust the
    thread pool the way offloading the whole blocking poll loop to a thread did.

    Returns the same outcome vocabulary as :func:`_poll_approval`; every
    non-``approved`` outcome is a FAIL-CLOSED denial at the call site.
    """
    deadline = time.monotonic() + client.approval_timeout
    loop = asyncio.get_running_loop()
    while True:
        result, sleep_for = await loop.run_in_executor(
            None, _approval_poll_once, client, approval_id, deadline
        )
        if result is not None:
            return result
        await asyncio.sleep(sleep_for)


def _governance_check(
    task: SigilTaskContext,
    client: SigilClient,
    tool_fqn: str,
    namespace: str,
    name: str,
    ah: str,
    risk_tier: str,
    agent_id: str,
    task_id: str,
    args_redacted: Any,
) -> tuple[bool, str | None]:
    """Run governance steps 2-4 shared by sync and async wrappers.

    Steps executed:

    2. **Revocation check** — in-memory, no network.
    3. **Local verify** — ed25519 biscuit verification, <1 ms.
    4. **Preflight** — HTTP call to sigil-core for high/critical risk only.

    Returns:
        ``(fail_open, approval_id)``:

        * ``fail_open`` — ``True`` only when sigil-core is unreachable and
          ``fail_mode="open"``.  Always ``False`` for low/med risk.
        * ``approval_id`` — non-``None`` only when preflight returned an ``approve``
          verdict (ENT-81): the caller MUST poll that approval (``_poll_approval`` /
          ``_poll_approval_async``) and pass the outcome to
          :func:`_finalize_approval`.  The poll is deliberately NOT done here so the
          async path does not pin an executor thread for the approval window (#311).

    Raises:
        SigilDeniedError: On revocation, local verify denial, preflight deny,
            or unknown preflight verdict.
        SigilUnreachableDeniedError: On preflight transport/API failure when
            ``fail_mode="closed"``.
    """
    # ── Step 2: revocation check (in-memory, no network) ─────────────────────
    if client.is_revoked(agent_id, task_id):
        ev = _make_event(
            agent_id,
            task_id,
            tool_fqn,
            namespace,
            ah,
            0,
            "denied",
            risk_tier,
            denied_reason="agent_revoked",
            args_redacted=args_redacted,
        )
        client._log_buffer.push(ev)
        raise SigilDeniedError(
            f"Agent or task is revoked; call to '{tool_fqn}' denied",
            denied_reason="agent_revoked",
            tool_name=tool_fqn,
            task_id=task_id,
        )

    # ── Step 2b: kill-switch health gate (fail-closed option) ─────────────────
    # If the operator set kill_switch_fail_mode="closed", a DEGRADED subscriber
    # (configured but not connected → revocations may be silently missed) must DENY
    # rather than allow. Default "open" preserves prior behavior; the subscriber's own
    # throttled WARNING + client.is_kill_switch_healthy() give visibility either way.
    # is_kill_switch_healthy() returns True when the kill-switch is intentionally
    # disabled, so this never fires for deployments that don't use the Redis kill-switch.
    if client.kill_switch_fail_mode == "closed" and not client.is_kill_switch_healthy():
        ev = _make_event(
            agent_id,
            task_id,
            tool_fqn,
            namespace,
            ah,
            0,
            "denied",
            risk_tier,
            denied_reason="kill_switch_degraded",
            args_redacted=args_redacted,
        )
        client._log_buffer.push(ev)
        raise SigilDeniedError(
            f"Kill-switch degraded; call to '{tool_fqn}' denied "
            "(kill_switch_fail_mode=closed)",
            denied_reason="kill_switch_degraded",
            tool_name=tool_fqn,
            task_id=task_id,
        )

    # ── Step 3: local verify (<1 ms, no network) ──────────────────────────────
    biscuit_token = task._biscuit_token
    if biscuit_token is None:
        raise SigilDeniedError(
            "Task context has no biscuit token (was __enter__ called?)",
            denied_reason="token_invalid",
            tool_name=tool_fqn,
            task_id=task_id,
        )
    vr = verify_local(
        biscuit_token,
        client.biscuit_keyring,
        active_kid=client.active_kid or None,
        required_tool=tool_fqn,
        expected_tenant=client.tenant_id if client.tenant_id else None,
    )
    if not vr.ok:
        ev = _make_event(
            agent_id,
            task_id,
            tool_fqn,
            namespace,
            ah,
            0,
            "denied",
            risk_tier,
            denied_reason=vr.reason,
            args_redacted=args_redacted,
        )
        client._log_buffer.push(ev)
        raise SigilDeniedError(
            f"Local verify denied '{tool_fqn}': {vr.reason}",
            denied_reason=vr.reason or "token_invalid",
            tool_name=tool_fqn,
            task_id=task_id,
        )

    # ── Step 4: preflight (high / critical only) ──────────────────────────────
    fail_open = False
    if risk_tier in _HIGH_RISK_TIERS:
        try:
            verdict = client.preflight(
                biscuit_token,
                namespace,
                name,
                ah,
                agent_id=agent_id,
                task_id=task_id,
            )
            # M1: null / empty / unknown verdict → fail closed (deny).
            v: str = verdict.get("verdict") or "deny"
            if v == "deny":
                reason: str = verdict.get("denied_reason") or "policy_deny"
                ev = _make_event(
                    agent_id,
                    task_id,
                    tool_fqn,
                    namespace,
                    ah,
                    0,
                    "denied",
                    risk_tier,
                    denied_reason=reason,
                    args_redacted=args_redacted,
                )
                client._log_buffer.push(ev)
                # SG-6 (ENT-86c): a cascade-revocation containment surfaced on the preflight
                # deny path is TERMINAL — raise the distinct AgentQuarantinedError so a caller's
                # retry/backoff wrapper stops rather than hammering a permanently-denying
                # endpoint. Subclass of SigilDeniedError, so existing handlers still catch it.
                # Scope: this is the preflight deny path only. A generic kill-switch revocation
                # (Step 2 in-memory subscriber) is a separate signal that stays "agent_revoked".
                # Trim+lower the server reason before matching so a stray-cased/whitespaced value
                # still classifies terminal (fail-safe: a miss only downgrades to the retryable
                # base class — it never turns a deny into an allow).
                err_cls = (
                    AgentQuarantinedError
                    if reason.strip().lower() in TERMINAL_QUARANTINE_REASONS
                    else SigilDeniedError
                )
                raise err_cls(
                    f"Preflight denied '{tool_fqn}': {reason}",
                    denied_reason=reason,
                    tool_name=tool_fqn,
                    task_id=task_id,
                )
            elif v == "approve":
                # ENT-81/SG-4: a HIGH/CRITICAL tool call needs human approval. sigil-core
                # has opened an intelligent-automation approval; block here polling its
                # status until a sigil_approver resolves it (or we time out / it becomes
                # unreachable). Every non-approved outcome is FAIL-CLOSED (deny).
                approval_id = verdict.get("approval_id")
                if not approval_id:
                    # approve verdict with no approval id → the gate could not be opened
                    # server-side; fail closed.
                    ev = _make_event(
                        agent_id,
                        task_id,
                        tool_fqn,
                        namespace,
                        ah,
                        0,
                        "denied",
                        risk_tier,
                        denied_reason="approval_service_unavailable",
                        args_redacted=args_redacted,
                    )
                    client._log_buffer.push(ev)
                    raise SigilDeniedError(
                        f"Preflight returned 'approve' for '{tool_fqn}' without an "
                        "approval_id; failing closed",
                        denied_reason="approval_service_unavailable",
                        tool_name=tool_fqn,
                        task_id=task_id,
                    )
                # Do NOT poll here: on the async path this function runs inside an
                # executor thread (I-1), so a blocking poll of up to approval_timeout
                # (300s) would pin that thread (#311). Signal "approval pending" to the
                # caller, which polls with the correct primitive — blocking
                # _poll_approval or native-async _poll_approval_async — and turns the
                # outcome into a fall-through or a fail-closed denial via
                # _finalize_approval.
                return False, str(approval_id)
            elif v != "allow":
                # Unknown verdict (e.g. null, "", "ALLOW") — fail closed.
                ev = _make_event(
                    agent_id,
                    task_id,
                    tool_fqn,
                    namespace,
                    ah,
                    0,
                    "denied",
                    risk_tier,
                    denied_reason="unknown_verdict",
                    args_redacted=args_redacted,
                )
                client._log_buffer.push(ev)
                raise SigilDeniedError(
                    f"Preflight returned unknown verdict {v!r} for '{tool_fqn}'; " "failing closed",
                    denied_reason="unknown_verdict",
                    tool_name=tool_fqn,
                    task_id=task_id,
                )
            # v == "allow": fall through to execute.
        except (SigilTransportError, SigilAPIError) as exc:
            # H1: widen beyond SigilTransportError — SigilAPIError (503/429/500)
            # is also treated as unreachable and routed through the fail-mode path.
            fail_open = _handle_preflight_unreachable(
                task,
                tool_fqn,
                namespace,
                ah,
                risk_tier,
                agent_id,
                task_id,
                exc,
                args_redacted=args_redacted,
            )
            # fail_open=True → continue to execute

    return fail_open, None


def _finalize_approval(
    client: SigilClient,
    outcome: str,
    *,
    agent_id: str,
    task_id: str,
    tool_fqn: str,
    namespace: str,
    ah: str,
    risk_tier: str,
    args_redacted: Any,
) -> None:
    """Turn an approval poll *outcome* into a fall-through or a FAIL-CLOSED denial.

    Extracted from the preflight approve-branch so the sync and async wrappers can
    each supply the poll result from their own poll implementation (blocking
    :func:`_poll_approval` vs. native-async :func:`_poll_approval_async`) while
    sharing identical denial-event + exception handling (#311).

    Returns normally on ``"approved"`` (the caller proceeds to execute). Every other
    outcome pushes a ``denied`` audit event and raises :class:`SigilDeniedError`.
    """
    if outcome == "approved":
        return
    # Any non-approved outcome denies. .get() with a fail-closed default guards against
    # an unexpected outcome string ever surfacing a raw KeyError instead of a clean
    # SigilDeniedError (defensive — the poll helpers only emit the four keys below).
    reason = {
        "rejected": "approval_rejected",
        "expired": "approval_expired",
        "timeout": "approval_timeout",
        "unavailable": "approval_service_unavailable",
    }.get(outcome, "approval_service_unavailable")
    ev = _make_event(
        agent_id,
        task_id,
        tool_fqn,
        namespace,
        ah,
        0,
        "denied",
        risk_tier,
        denied_reason=reason,
        args_redacted=args_redacted,
    )
    client._log_buffer.push(ev)
    raise SigilDeniedError(
        f"Approval for '{tool_fqn}' resolved to {reason}",
        denied_reason=reason,
        tool_name=tool_fqn,
        task_id=task_id,
    )


def _redeem_denied_reason(exc: Exception) -> str:
    """Map a redeem failure to a bounded denial reason. HTTP 409 = the approval was
    already redeemed (single-use exhausted); anything else = the one-shot could not be
    minted / confirmed."""
    if getattr(exc, "status_code", None) == 409:
        return "approval_replayed"
    return "approval_token_unavailable"


def _raise_redeem_denied(
    client: SigilClient,
    reason: str,
    *,
    agent_id: str,
    task_id: str,
    tool_fqn: str,
    namespace: str,
    ah: str,
    risk_tier: str,
    args_redacted: Any,
) -> NoReturn:
    """Push a denied audit event and raise SigilDeniedError for a failed redemption.
    A governed call must NEVER proceed without a confirmed single-use redemption (#ENT-82)."""
    ev = _make_event(
        agent_id, task_id, tool_fqn, namespace, ah, 0, "denied", risk_tier,
        denied_reason=reason, args_redacted=args_redacted,
    )
    client._log_buffer.push(ev)
    raise SigilDeniedError(
        f"Approval redemption for '{tool_fqn}' failed: {reason}",
        denied_reason=reason,
        tool_name=tool_fqn,
        task_id=task_id,
    )


def _redeem_approval(
    client: SigilClient,
    approval_id: str,
    token: str,
    *,
    agent_id: str,
    task_id: str,
    tool_fqn: str,
    namespace: str,
    ah: str,
    risk_tier: str,
    args_redacted: Any,
) -> str:
    """Redeem an APPROVED gate for a one-shot single-use grant; return its revocation_id.

    Called once after :func:`_finalize_approval` confirms ``"approved"``. ANY failure
    (already redeemed, unreachable, malformed) fails CLOSED: a denied event is emitted and
    :class:`SigilDeniedError` raised — the governed call does not run. The returned grant id
    is recorded on the execution audit event as proof of a fresh, human-approved grant."""
    try:
        result = client.redeem_approval(approval_id, token)
    except Exception as exc:  # noqa: BLE001
        _raise_redeem_denied(
            client, _redeem_denied_reason(exc),
            agent_id=agent_id, task_id=task_id, tool_fqn=tool_fqn,
            namespace=namespace, ah=ah, risk_tier=risk_tier, args_redacted=args_redacted,
        )
    else:
        # `result` is provably bound here (the except branch is NoReturn). Using `else`
        # makes that unambiguous and robust against future refactors of the raise helper.
        revocation_id = str(result.get("revocation_id") or "")
        if not revocation_id:
            # Defence-in-depth: client.redeem_approval already raises on a missing grant id,
            # but a governed call must never proceed without proof of a single-use redemption.
            _raise_redeem_denied(
                client, "approval_token_unavailable",
                agent_id=agent_id, task_id=task_id, tool_fqn=tool_fqn,
                namespace=namespace, ah=ah, risk_tier=risk_tier, args_redacted=args_redacted,
            )
        return revocation_id


async def _redeem_approval_async(
    client: SigilClient,
    approval_id: str,
    token: str,
    *,
    agent_id: str,
    task_id: str,
    tool_fqn: str,
    namespace: str,
    ah: str,
    risk_tier: str,
    args_redacted: Any,
) -> str:
    """Native-async variant of :func:`_redeem_approval` (#311 pattern): offload only the
    single blocking redeem HTTP call to a thread so the event loop is not stalled."""
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, client.redeem_approval, approval_id, token)
    except Exception as exc:  # noqa: BLE001
        _raise_redeem_denied(
            client, _redeem_denied_reason(exc),
            agent_id=agent_id, task_id=task_id, tool_fqn=tool_fqn,
            namespace=namespace, ah=ah, risk_tier=risk_tier, args_redacted=args_redacted,
        )
    else:
        # `result` is provably bound here (the except branch is NoReturn). Using `else`
        # makes that unambiguous and robust against future refactors of the raise helper.
        revocation_id = str(result.get("revocation_id") or "")
        if not revocation_id:
            # Defence-in-depth: client.redeem_approval already raises on a missing grant id,
            # but a governed call must never proceed without proof of a single-use redemption.
            _raise_redeem_denied(
                client, "approval_token_unavailable",
                agent_id=agent_id, task_id=task_id, tool_fqn=tool_fqn,
                namespace=namespace, ah=ah, risk_tier=risk_tier, args_redacted=args_redacted,
            )
        return revocation_id


# ──────────────────────────────────────────────────────────────────────────────
# SG-9 SP-1 (ENT-91): Task-Replay prompt-log lane (@instrumented_llm enrichment)
# ──────────────────────────────────────────────────────────────────────────────

# Chars of the response kept in the DLP-scrubbed sample. Bounds the buffered/overflowed
# event size; the full response is never stored (only its hash + this redacted head).
_RESPONSE_SAMPLE_MAX: int = 2000
# Chars of serialized prompt_redacted kept — bounds prompt_redacted the same way
# _RESPONSE_SAMPLE_MAX bounds the response sample (a single log-prompt is one DB row per
# LLM call, so an unbounded redacted-args payload would inflate buffer/overflow/DB storage).
_PROMPT_REDACTED_MAX: int = 4000


def _extract_response_text(result: Any) -> str:
    """Best-effort string form of an LLM result for hashing + sampling. Never raises.

    Handles the common return shapes (OpenAI-style ``choices[0].message.content`` /
    ``choices[0].text``, a bare ``str``, dicts) and falls back to ``str(result)`` so the
    prompt lane still gets a stable hash even for an unrecognised SDK response object.
    """
    if result is None:
        return ""  # a legit None result → empty response, not the literal string "None"
    try:
        choices = getattr(result, "choices", None)
        if choices is None and isinstance(result, dict):
            choices = result.get("choices")
        if choices:
            first = choices[0]
            msg = getattr(first, "message", None)
            if msg is None and isinstance(first, dict):
                msg = first.get("message")
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if isinstance(content, str):
                return content
            text = getattr(first, "text", None)
            if text is None and isinstance(first, dict):
                text = first.get("text")
            if isinstance(text, str):
                return text
        if isinstance(result, str):
            return result
    except Exception:  # noqa: BLE001
        pass
    try:
        return str(result)
    except Exception:  # noqa: BLE001
        return ""


def _extract_tokens(result: Any) -> tuple[int, int]:
    """Best-effort ``(prompt_tokens, completion_tokens)`` from an LLM result. Never raises.

    Reads the OpenAI-style ``usage`` object/dict when present; returns ``(0, 0)`` when the
    result carries no recognisable usage (the columns are nullable-defaulting-to-0, so 0
    honestly signals "not reported by the provider").
    """
    try:
        usage = getattr(result, "usage", None)
        if usage is None and isinstance(result, dict):
            usage = result.get("usage")
        if usage is not None:

            def _g(obj: Any, key: str) -> int:
                v = getattr(obj, key, None)
                if v is None and isinstance(obj, dict):
                    v = obj.get(key)
                return int(v) if isinstance(v, (int, float)) else 0

            return _g(usage, "prompt_tokens"), _g(usage, "completion_tokens")
    except Exception:  # noqa: BLE001
        pass
    return 0, 0


def _resolve_model(explicit: str | None, name: str, kwargs: dict[str, Any], result: Any) -> str:
    """Resolve the concrete model id for the prompt-log row (never empty — the column is NOT NULL).

    Precedence: an explicit ``model=`` on ``@instrumented_llm`` > the call's ``model=`` kwarg
    (the near-universal LLM SDK convention) > the result object's ``.model`` > the decorator
    tool ``name`` as a coarse fallback. Best-effort; never raises.
    """
    if explicit:
        return explicit
    m = kwargs.get("model")
    if isinstance(m, str) and m:
        return m
    try:
        rm = getattr(result, "model", None)
        if rm is None and isinstance(result, dict):
            rm = result.get("model")
        if isinstance(rm, str) and rm:
            return rm
    except Exception:  # noqa: BLE001
        pass
    return name


def _build_prompt_log_event(
    *,
    task_id: str,
    agent_id: str,
    namespace: str,
    tool_name: str,
    explicit_model: str | None,
    prompt_text: str,
    prompt_redacted: Any,
    result: Any,
    latency_ms: int,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the buffered prompt-log event for the Task-Replay prompt lane.

    Carries hashes + DLP-redacted samples ONLY (never raw prompt/response text). Tagged with
    the buffer discriminator so :class:`~sigil._buffer._Flusher` routes it to ``log_prompt``.
    """
    response_text = _extract_response_text(result)
    token_in, token_out = _extract_tokens(result)
    # Redact a margin window BEFORE truncating so a DLP pattern straddling the cut is still
    # scrubbed (the response_hash is over the full text; only this sample is bounded).
    redacted_sample = redact_safe({"text": response_text[: _RESPONSE_SAMPLE_MAX * 2]})
    sample_text = redacted_sample.get("text", "")[:_RESPONSE_SAMPLE_MAX]
    # Bound prompt_redacted's serialized size (already DLP-scrubbed; just size-capped).
    pr: Any = prompt_redacted if isinstance(prompt_redacted, dict) else {"args": prompt_redacted}
    pr_json = canonical_json(pr)
    if len(pr_json) > _PROMPT_REDACTED_MAX:
        pr = {"_truncated": True, "preview": pr_json[:_PROMPT_REDACTED_MAX]}
    return {
        _KIND_KEY: _PROMPT_LOG_KIND,
        "task_id": task_id,
        "agent_id": agent_id,
        "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "prompt_redacted": pr,
        "response_hash": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "response_sampled": {"text": sample_text},
        "model": _resolve_model(explicit_model, tool_name, kwargs, result),
        "model_provider": namespace,
        "token_count_input": token_in,
        "token_count_output": token_out,
        "latency_ms": latency_ms,
    }


def _enqueue_prompt_log(
    client: SigilClient,
    prompt_log_ctx: dict[str, Any],
    *,
    task_id: str,
    agent_id: str,
    result: Any,
    latency_ms: int,
) -> None:
    """Build + buffer one prompt-log event. Best-effort: never raises into the governed call.

    Pushed onto the SAME ring buffer as tool events, so it is non-blocking on both the sync and
    async paths and inherits the buffer's durability (overflow-backed + drained on close).
    """
    try:
        ev = _build_prompt_log_event(
            task_id=task_id,
            agent_id=agent_id,
            result=result,
            latency_ms=latency_ms,
            **prompt_log_ctx,
        )
        client._log_buffer.push(ev)
    except Exception:  # noqa: BLE001 — enrichment lane; a failure here must not affect the call
        _log.debug("sigil: prompt-log build/enqueue failed (task_id=%s)", task_id, exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# I-3: shared execute+audit helpers (extracted from the 4-way inline shell)
# ──────────────────────────────────────────────────────────────────────────────


def _execute_audit(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    client: SigilClient,
    agent_id: str,
    task_id: str,
    tool_fqn: str,
    namespace: str,
    ah: str,
    risk_tier: str,
    fail_open: bool,
    args_redacted: Any,
    approval_grant_id: str | None = None,
    prompt_entropy: float = 0.0,
    prompt_log_ctx: dict[str, Any] | None = None,
) -> Any:
    """Execute *fn* synchronously, time it, and emit one audit event in all cases.

    The ``try/finally`` guarantees the audit event is emitted even when *fn*
    raises, producing ``outcome="error"`` instead of ``outcome="allowed"``.

    When *prompt_log_ctx* is supplied (``@instrumented_llm`` only) and the call
    succeeded, an additive prompt-log event is also buffered (SG-9 SP-1). It is
    enrichment on top of the tool-invocation event above and is emitted only on
    success — a prompt-log needs a real response.
    """
    t0 = time.monotonic()
    outcome = "allowed"
    result: Any = None
    try:
        result = fn(*args, **kwargs)
        return result
    except BaseException:
        outcome = "error"
        raise
    finally:
        latency_ms = int((time.monotonic() - t0) * 1000)
        ev = _make_event(
            agent_id,
            task_id,
            tool_fqn,
            namespace,
            ah,
            latency_ms,
            outcome,
            risk_tier,
            args_redacted=args_redacted,
            fail_open=fail_open,
            approval_grant_id=approval_grant_id,
            prompt_entropy=prompt_entropy,
        )
        client._log_buffer.push(ev)
        if prompt_log_ctx is not None and outcome == "allowed":
            _enqueue_prompt_log(
                client, prompt_log_ctx,
                task_id=task_id, agent_id=agent_id, result=result, latency_ms=latency_ms,
            )


async def _async_execute_audit(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    client: SigilClient,
    agent_id: str,
    task_id: str,
    tool_fqn: str,
    namespace: str,
    ah: str,
    risk_tier: str,
    fail_open: bool,
    args_redacted: Any,
    approval_grant_id: str | None = None,
    prompt_entropy: float = 0.0,
    prompt_log_ctx: dict[str, Any] | None = None,
) -> Any:
    """Async variant of ``_execute_audit`` — awaits *fn* and emits audit in finally.

    The additive prompt-log (when *prompt_log_ctx* is supplied) is enqueued on the same
    ring buffer via a non-blocking ``push`` — no executor offload is needed because the
    push does no I/O; the background flusher thread performs the network delivery.
    """
    t0 = time.monotonic()
    outcome = "allowed"
    result: Any = None
    try:
        result = await fn(*args, **kwargs)
        return result
    except BaseException:
        outcome = "error"
        raise
    finally:
        latency_ms = int((time.monotonic() - t0) * 1000)
        ev = _make_event(
            agent_id,
            task_id,
            tool_fqn,
            namespace,
            ah,
            latency_ms,
            outcome,
            risk_tier,
            args_redacted=args_redacted,
            fail_open=fail_open,
            approval_grant_id=approval_grant_id,
            prompt_entropy=prompt_entropy,
        )
        client._log_buffer.push(ev)
        if prompt_log_ctx is not None and outcome == "allowed":
            _enqueue_prompt_log(
                client, prompt_log_ctx,
                task_id=task_id, agent_id=agent_id, result=result, latency_ms=latency_ms,
            )


# ──────────────────────────────────────────────────────────────────────────────
# @instrumented_tool
# ──────────────────────────────────────────────────────────────────────────────


def instrumented_tool(
    namespace: str,
    name: str,
    risk_tier: str = "low",
) -> Callable[[F], F]:
    """Decorator that instruments a tool function with Sigil governance.

    Supports both sync and async functions.  At decoration time,
    ``inspect.iscoroutinefunction`` detects coroutine functions and returns
    an ``async def`` wrapper that offloads the synchronous governance steps
    (2–4) to a thread via ``run_in_executor`` (I-1) and then ``await fn(...)``.

    Note: for ``risk_tier`` in ``{"high", "critical"}``, preflight makes a
    blocking ``requests`` HTTP call.  In the async path this is offloaded to
    the default thread-pool executor so the event loop is not stalled.
    AsyncSigilClient v1.1 will provide a natively async preflight path.

    Args:
        namespace: Tool namespace, e.g. ``"zep"``.
        name: Tool name within the namespace, e.g. ``"search"``.
            Combined as ``namespace.name`` for token scope checks.
        risk_tier: Risk classification — ``"low"`` (default), ``"med"``,
            ``"high"``, or ``"critical"``.  Tiers ``high`` and ``critical``
            always trigger an HTTP preflight call to sigil-core regardless
            of the local token result.

    Returns:
        Decorator that wraps the target callable (sync or async preserved).

    Raises:
        ValueError: At decoration time if *risk_tier* is not one of the four
            valid values (F6 — catches typos like ``"medium"`` or ``"HIGH"``
            at import rather than silently skipping preflight at call time).
        SigilDeniedError: On revocation, out-of-scope token, or preflight deny.
        SigilUnreachableDeniedError: When sigil-core is unreachable and
            ``fail_mode="closed"`` (the default).
    """
    # F6: validate at decoration time so typos fail loudly at import.
    if risk_tier not in _VALID_RISK_TIERS:
        raise ValueError(
            f"instrumented_tool: invalid risk_tier {risk_tier!r}; "
            f"must be one of {sorted(_VALID_RISK_TIERS)}"
        )

    tool_fqn = f"{namespace}.{name}"

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # ── Step 1: require active task context ───────────────────────
                task = _require_task(tool_fqn)
                client: SigilClient = task._client
                agent_id: str = task.effective_agent_id
                task_id: str = task.task_id or ""

                raw = _raw_args(args, kwargs)
                ah = args_hash(raw)

                # ── Steps 2-4: governance offloaded to thread (I-1) ───────────
                # high/critical preflight is a blocking requests.post call;
                # run_in_executor prevents stalling the event loop.  Exceptions
                # raised inside _governance_check propagate correctly out of the
                # awaited future.
                fail_open, approval_id = await asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(
                        _governance_check,
                        task,
                        client,
                        tool_fqn,
                        namespace,
                        name,
                        ah,
                        risk_tier,
                        agent_id,
                        task_id,
                        None,  # args_redacted
                    ),
                )

                # ENT-81 (#311): "approve" verdict → poll the approval NATIVELY async.
                # Only the individual status calls are offloaded per-poll; the idle waits
                # run on the event loop, so no executor thread is pinned for the (up to
                # 300s) approval window.
                approval_grant_id: str | None = None
                if approval_id is not None:
                    outcome = await _poll_approval_async(client, approval_id)
                    _finalize_approval(
                        client,
                        outcome,
                        agent_id=agent_id,
                        task_id=task_id,
                        tool_fqn=tool_fqn,
                        namespace=namespace,
                        ah=ah,
                        risk_tier=risk_tier,
                        args_redacted=None,
                    )
                    # ENT-82: approved → redeem once for a one-shot single-use grant (fail-closed).
                    approval_grant_id = await _redeem_approval_async(
                        client,
                        approval_id,
                        task._biscuit_token or "",
                        agent_id=agent_id,
                        task_id=task_id,
                        tool_fqn=tool_fqn,
                        namespace=namespace,
                        ah=ah,
                        risk_tier=risk_tier,
                        args_redacted=None,
                    )

                # ── Steps 5-6: execute + audit (I-3: one-liner via helper) ────
                return await _async_execute_audit(
                    fn,
                    args,
                    kwargs,
                    client=client,
                    agent_id=agent_id,
                    task_id=task_id,
                    tool_fqn=tool_fqn,
                    namespace=namespace,
                    ah=ah,
                    risk_tier=risk_tier,
                    fail_open=fail_open,
                    args_redacted=None,
                    approval_grant_id=approval_grant_id,
                )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # ── Step 1: require active task context ───────────────────────────
            task = _require_task(tool_fqn)
            client: SigilClient = task._client
            agent_id: str = task.effective_agent_id
            task_id: str = task.task_id or ""

            raw = _raw_args(args, kwargs)
            ah = args_hash(raw)

            # ── Steps 2-4: governance (shared helper) ─────────────────────────
            fail_open, approval_id = _governance_check(
                task,
                client,
                tool_fqn,
                namespace,
                name,
                ah,
                risk_tier,
                agent_id,
                task_id,
                None,  # args_redacted
            )

            # ENT-81 (#311): "approve" verdict → block-poll the approval (sync path).
            # _finalize_approval turns the outcome into a fall-through or fail-closed deny.
            approval_grant_id: str | None = None
            if approval_id is not None:
                outcome = _poll_approval(client, approval_id)
                _finalize_approval(
                    client,
                    outcome,
                    agent_id=agent_id,
                    task_id=task_id,
                    tool_fqn=tool_fqn,
                    namespace=namespace,
                    ah=ah,
                    risk_tier=risk_tier,
                    args_redacted=None,
                )
                # ENT-82: approved → redeem once for a one-shot single-use grant (fail-closed).
                approval_grant_id = _redeem_approval(
                    client,
                    approval_id,
                    task._biscuit_token or "",
                    agent_id=agent_id,
                    task_id=task_id,
                    tool_fqn=tool_fqn,
                    namespace=namespace,
                    ah=ah,
                    risk_tier=risk_tier,
                    args_redacted=None,
                )

            # ── Steps 5-6: execute + audit (I-3: one-liner via helper) ────────
            return _execute_audit(
                fn,
                args,
                kwargs,
                client=client,
                agent_id=agent_id,
                task_id=task_id,
                tool_fqn=tool_fqn,
                namespace=namespace,
                ah=ah,
                risk_tier=risk_tier,
                fail_open=fail_open,
                args_redacted=None,
                approval_grant_id=approval_grant_id,
            )

        return wrapper  # type: ignore[return-value]

    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# @instrumented_llm
# ──────────────────────────────────────────────────────────────────────────────


def instrumented_llm(
    namespace: str,
    name: str,
    risk_tier: str = "low",
    model: str | None = None,
) -> Callable[[F], F]:
    """Decorator that instruments an LLM call with Sigil governance + DLP.

    Supports both sync and async functions (same async dispatch as
    :func:`instrumented_tool`, including the ``run_in_executor`` offload for
    high/critical preflight — see that decorator's note for details).

    Identical to :func:`instrumented_tool` with these additions:

    * ``args_redacted`` — DLP-scrubbed copy of the call arguments is included
      in the buffered audit event (PII → ``<PII:TYPE>`` placeholders).
      Uses :func:`~sigil.redaction.redact_safe` so a DLP regex failure
      degrades gracefully instead of breaking the tool call.
    * ``args_hash`` — computed over the **original, unredacted** args so that
      sigil-core can verify argument integrity without storing sensitive data.
    * **Prompt-log lane (SG-9 SP-1 / ENT-91)** — on a *successful* call an
      additive prompt-log event is buffered alongside the tool-invocation event:
      ``prompt_hash``/``response_hash`` + DLP-redacted samples, ``model`` /
      ``model_provider`` (=``namespace``), best-effort token counts, and
      ``latency_ms``. This feeds the Task-Replay prompt lane and is enrichment
      on top of the authoritative tool-invocation event — a failure to emit it
      never affects the governed call. Raw prompt/response text is never stored.

    Args:
        namespace: Tool namespace, e.g. ``"openai"``. Also used as
            ``model_provider`` on the prompt-log row.
        name: Tool name, e.g. ``"chat"``.
        risk_tier: Risk classification (default ``"low"``).
        model: Concrete model id for the prompt-log row (e.g. ``"gpt-4o"``). When
            omitted, the SDK derives it best-effort from the call's ``model=``
            kwarg, then the result object's ``.model``, then falls back to *name*.

    Returns:
        Decorator that wraps the target callable (sync or async preserved).

    Raises:
        ValueError: At decoration time if *risk_tier* is not one of the four
            valid values (F6).
    """
    # F6: validate at decoration time so typos fail loudly at import.
    if risk_tier not in _VALID_RISK_TIERS:
        raise ValueError(
            f"instrumented_llm: invalid risk_tier {risk_tier!r}; "
            f"must be one of {sorted(_VALID_RISK_TIERS)}"
        )

    tool_fqn = f"{namespace}.{name}"

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # ── Step 1: require active task context ───────────────────────
                task = _require_task(tool_fqn)
                client: SigilClient = task._client
                agent_id: str = task.effective_agent_id
                task_id: str = task.task_id or ""

                raw = _raw_args(args, kwargs)
                ah = args_hash(raw)  # over UNREDACTED original
                redacted = redact_safe(raw)  # DLP-scrubbed copy for audit log

                # SG-5: extract prompt text using the robust helper (handles str,
                # messages-list, and kwarg calling conventions) then compute entropy.
                # SG-9 SP-1: the same extracted text feeds the prompt-log lane below.
                prompt_text = _extract_prompt(args, kwargs)
                entropy = _shannon_entropy(prompt_text)
                prompt_log_ctx = {
                    "namespace": namespace,
                    "tool_name": name,
                    "explicit_model": model,
                    "prompt_text": prompt_text,
                    "prompt_redacted": redacted,
                    "kwargs": kwargs,
                }

                # ── Steps 2-4: governance offloaded to thread (I-1) ───────────
                fail_open, approval_id = await asyncio.get_running_loop().run_in_executor(
                    None,
                    functools.partial(
                        _governance_check,
                        task,
                        client,
                        tool_fqn,
                        namespace,
                        name,
                        ah,
                        risk_tier,
                        agent_id,
                        task_id,
                        redacted,  # args_redacted
                    ),
                )

                # ENT-81 (#311): "approve" verdict → poll the approval NATIVELY async.
                # Only the individual status calls are offloaded per-poll; the idle waits
                # run on the event loop, so no executor thread is pinned for the (up to
                # 300s) approval window.
                approval_grant_id: str | None = None
                if approval_id is not None:
                    outcome = await _poll_approval_async(client, approval_id)
                    _finalize_approval(
                        client,
                        outcome,
                        agent_id=agent_id,
                        task_id=task_id,
                        tool_fqn=tool_fqn,
                        namespace=namespace,
                        ah=ah,
                        risk_tier=risk_tier,
                        args_redacted=redacted,
                    )
                    # ENT-82: approved → redeem once for a one-shot single-use grant (fail-closed).
                    approval_grant_id = await _redeem_approval_async(
                        client,
                        approval_id,
                        task._biscuit_token or "",
                        agent_id=agent_id,
                        task_id=task_id,
                        tool_fqn=tool_fqn,
                        namespace=namespace,
                        ah=ah,
                        risk_tier=risk_tier,
                        args_redacted=redacted,
                    )

                # ── Steps 5-6: execute + audit (I-3: one-liner via helper) ────
                return await _async_execute_audit(
                    fn,
                    args,
                    kwargs,
                    client=client,
                    agent_id=agent_id,
                    task_id=task_id,
                    tool_fqn=tool_fqn,
                    namespace=namespace,
                    ah=ah,
                    risk_tier=risk_tier,
                    fail_open=fail_open,
                    args_redacted=redacted,
                    approval_grant_id=approval_grant_id,
                    prompt_entropy=entropy,
                    prompt_log_ctx=prompt_log_ctx,
                )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # ── Step 1: require active task context ───────────────────────────
            task = _require_task(tool_fqn)
            client: SigilClient = task._client
            agent_id: str = task.effective_agent_id
            task_id: str = task.task_id or ""

            raw = _raw_args(args, kwargs)
            ah = args_hash(raw)  # over UNREDACTED original
            redacted = redact_safe(raw)  # DLP-scrubbed copy for audit log

            # SG-5: extract prompt text using the robust helper (handles str,
            # messages-list, and kwarg calling conventions) then compute entropy.
            # SG-9 SP-1: the same extracted text feeds the prompt-log lane below.
            prompt_text = _extract_prompt(args, kwargs)
            entropy = _shannon_entropy(prompt_text)
            prompt_log_ctx = {
                "namespace": namespace,
                "tool_name": name,
                "explicit_model": model,
                "prompt_text": prompt_text,
                "prompt_redacted": redacted,
                "kwargs": kwargs,
            }

            # ── Steps 2-4: governance (shared helper) ─────────────────────────
            fail_open, approval_id = _governance_check(
                task,
                client,
                tool_fqn,
                namespace,
                name,
                ah,
                risk_tier,
                agent_id,
                task_id,
                redacted,  # args_redacted
            )

            # ENT-81 (#311): "approve" verdict → block-poll the approval (sync path).
            # _finalize_approval turns the outcome into a fall-through or fail-closed deny.
            approval_grant_id: str | None = None
            if approval_id is not None:
                outcome = _poll_approval(client, approval_id)
                _finalize_approval(
                    client,
                    outcome,
                    agent_id=agent_id,
                    task_id=task_id,
                    tool_fqn=tool_fqn,
                    namespace=namespace,
                    ah=ah,
                    risk_tier=risk_tier,
                    args_redacted=redacted,
                )
                # ENT-82: approved → redeem once for a one-shot single-use grant (fail-closed).
                approval_grant_id = _redeem_approval(
                    client,
                    approval_id,
                    task._biscuit_token or "",
                    agent_id=agent_id,
                    task_id=task_id,
                    tool_fqn=tool_fqn,
                    namespace=namespace,
                    ah=ah,
                    risk_tier=risk_tier,
                    args_redacted=redacted,
                )

            # ── Steps 5-6: execute + audit (I-3: one-liner via helper) ────────
            return _execute_audit(
                fn,
                args,
                kwargs,
                client=client,
                agent_id=agent_id,
                task_id=task_id,
                tool_fqn=tool_fqn,
                namespace=namespace,
                ah=ah,
                risk_tier=risk_tier,
                fail_open=fail_open,
                args_redacted=redacted,
                approval_grant_id=approval_grant_id,
                prompt_entropy=entropy,
                prompt_log_ctx=prompt_log_ctx,
            )

        return wrapper  # type: ignore[return-value]

    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# @task decorator (internal stub — NOT in public __all__; use client.task())
# ──────────────────────────────────────────────────────────────────────────────


def task(task_type: str, scope: dict[str, Any]) -> NoReturn:
    """Decorator stub — raises ``NotImplementedError`` at decoration time.

    This decorator requires a default ``SigilClient`` mechanism
    (``SigilClient.set_default()``, not yet implemented).
    Use ``with client.task([...]) as task:`` context manager instead.

    Raises:
        NotImplementedError: Always — at decoration time, not call time.
    """
    raise NotImplementedError(
        "task() decorator requires a configured default SigilClient. "
        "Use 'with client.task([...]) as task:' context manager instead."
    )
