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
import inspect
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from sigil._context import _current_task
from sigil.errors import (
    SigilAPIError,
    SigilDeniedError,
    SigilTransportError,
    SigilUnreachableDeniedError,
)
from sigil.redaction import args_hash, redact_safe
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
    }
    if denied_reason is not None:
        ev["denied_reason"] = denied_reason
    if args_redacted is not None:
        ev["args_redacted"] = args_redacted
    if fail_open:
        ev["fail_open"] = True
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
) -> bool:
    """Run governance steps 2-4 shared by sync and async wrappers.

    Steps executed:

    2. **Revocation check** — in-memory, no network.
    3. **Local verify** — ed25519 biscuit verification, <1 ms.
    4. **Preflight** — HTTP call to sigil-core for high/critical risk only.

    Returns:
        ``fail_open`` flag — ``True`` only when sigil-core is unreachable and
        ``fail_mode="open"``.  Always ``False`` for low/med risk.

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
                raise SigilDeniedError(
                    f"Preflight denied '{tool_fqn}': {reason}",
                    denied_reason=reason,
                    tool_name=tool_fqn,
                    task_id=task_id,
                )
            elif v == "approve":
                # v1: approval gates are v2 — treat approve as deny.
                ev = _make_event(
                    agent_id,
                    task_id,
                    tool_fqn,
                    namespace,
                    ah,
                    0,
                    "denied",
                    risk_tier,
                    denied_reason="approval_required",
                    args_redacted=args_redacted,
                )
                client._log_buffer.push(ev)
                raise SigilDeniedError(
                    f"Preflight returned 'approve' for '{tool_fqn}'; "
                    "approval gates are v2 — treating as deny",
                    denied_reason="approval_required",
                    tool_name=tool_fqn,
                    task_id=task_id,
                )
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

    return fail_open


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
) -> Any:
    """Execute *fn* synchronously, time it, and emit one audit event in all cases.

    The ``try/finally`` guarantees the audit event is emitted even when *fn*
    raises, producing ``outcome="error"`` instead of ``outcome="allowed"``.
    """
    t0 = time.monotonic()
    outcome = "allowed"
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
        )
        client._log_buffer.push(ev)


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
) -> Any:
    """Async variant of ``_execute_audit`` — awaits *fn* and emits audit in finally."""
    t0 = time.monotonic()
    outcome = "allowed"
    try:
        result: Any = await fn(*args, **kwargs)
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
        )
        client._log_buffer.push(ev)


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
                agent_id: str = client.agent_id
                task_id: str = task.task_id or ""

                raw = _raw_args(args, kwargs)
                ah = args_hash(raw)

                # ── Steps 2-4: governance offloaded to thread (I-1) ───────────
                # high/critical preflight is a blocking requests.post call;
                # run_in_executor prevents stalling the event loop.  Exceptions
                # raised inside _governance_check propagate correctly out of the
                # awaited future.
                fail_open: bool = await asyncio.get_running_loop().run_in_executor(
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
                )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # ── Step 1: require active task context ───────────────────────────
            task = _require_task(tool_fqn)
            client: SigilClient = task._client
            agent_id: str = client.agent_id
            task_id: str = task.task_id or ""

            raw = _raw_args(args, kwargs)
            ah = args_hash(raw)

            # ── Steps 2-4: governance (shared helper) ─────────────────────────
            fail_open = _governance_check(
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
) -> Callable[[F], F]:
    """Decorator that instruments an LLM call with Sigil governance + DLP.

    Supports both sync and async functions (same async dispatch as
    :func:`instrumented_tool`, including the ``run_in_executor`` offload for
    high/critical preflight — see that decorator's note for details).

    Identical to :func:`instrumented_tool` with two additions:

    * ``args_redacted`` — DLP-scrubbed copy of the call arguments is included
      in the buffered audit event (PII → ``<PII:TYPE>`` placeholders).
      Uses :func:`~sigil.redaction.redact_safe` so a DLP regex failure
      degrades gracefully instead of breaking the tool call.
    * ``args_hash`` — computed over the **original, unredacted** args so that
      sigil-core can verify argument integrity without storing sensitive data.

    Args:
        namespace: Tool namespace, e.g. ``"openai"``.
        name: Tool name, e.g. ``"chat"``.
        risk_tier: Risk classification (default ``"low"``).

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
                agent_id: str = client.agent_id
                task_id: str = task.task_id or ""

                raw = _raw_args(args, kwargs)
                ah = args_hash(raw)  # over UNREDACTED original
                redacted = redact_safe(raw)  # DLP-scrubbed copy for audit log

                # ── Steps 2-4: governance offloaded to thread (I-1) ───────────
                fail_open: bool = await asyncio.get_running_loop().run_in_executor(
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
                )

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # ── Step 1: require active task context ───────────────────────────
            task = _require_task(tool_fqn)
            client: SigilClient = task._client
            agent_id: str = client.agent_id
            task_id: str = task.task_id or ""

            raw = _raw_args(args, kwargs)
            ah = args_hash(raw)  # over UNREDACTED original
            redacted = redact_safe(raw)  # DLP-scrubbed copy for audit log

            # ── Steps 2-4: governance (shared helper) ─────────────────────────
            fail_open = _governance_check(
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
