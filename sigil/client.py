"""SigilClient — HTTP transport to sigil-core.

Implements the internal wire protocol defined in docs/protocol.md and the
server contracts in:
  services/sigil-core/handlers/token_handler.go
  services/sigil-core/handlers/toolgate_handler.go

Pass 1 (primitives): issue_token, preflight, log_batch.
Pass 2 (governance): SigilTaskContext, background flusher, disk overflow,
    kill-switch subscriber, fail-mode.

Security notes:
  - ``internal_token`` (SIGIL_SDK_TOKEN) is NEVER logged or included in
    exception messages.
  - biscuit tokens returned by issue_token and cached by SigilTaskContext
    are never written to logs, overflow files, or audit events.
  - Raw tool arguments are never sent to sigil-core; only ``args_hash``
    (SHA-256 of the canonical JSON of the original args) is transmitted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from typing import NoReturn
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote, urlparse

import requests

from sigil._buffer import _Flusher, _LogBuffer
from sigil._context import _current_task
from sigil._overflow import _OverflowWriter
from sigil._subscriber import _RevocationSubscriber
from sigil.dpop import DPoPKey
from sigil.errors import (
    CredentialRotatedError,
    SigilAPIError,
    SigilDeniedError,
    SigilTransportError,
)
from sigil.mcp import (
    MCPToken,
    SigilTokenExchangeDeniedError,
    SigilTokenExchangeError,
    _MCPTokenCache,
)
from sigil.verify import VerifyResult, verify_local

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any

# Maximum number of events accepted by sigil-core per log-batch call.
# Mirrors services.MaxAuditBatchSize in services/sigil-core/handlers/toolgate_handler.go.
_MAX_BATCH_SIZE: int = 100

_log = logging.getLogger("sigil")


# ──────────────────────────────────────────────────────────────────────────────
# Environment / keyring helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_biscuit_keyring() -> dict[str, bytes]:
    """Build the biscuit public-key keyring from environment variables.

    Two env-var forms are supported:

    * ``SIGIL_BISCUIT_PUBKEYS`` — JSON object ``{"kid1": "<base64_pubkey>", ...}``
      for multi-key / rotation scenarios.
    * ``SIGIL_BISCUIT_PUBKEY`` + ``SIGIL_BISCUIT_KID`` — single-key shorthand.

    Returns:
        ``{kid: 32-byte-pubkey-bytes}`` mapping; empty dict if no env vars set.
    """
    keyring: dict[str, bytes] = {}

    pubkeys_json = os.environ.get("SIGIL_BISCUIT_PUBKEYS", "")
    if pubkeys_json:
        try:
            raw: dict[str, str] = json.loads(pubkeys_json)
            for kid, b64_key in raw.items():
                keyring[kid] = base64.b64decode(b64_key)
        except Exception:  # noqa: BLE001
            pass  # malformed env var — caller's keyring takes precedence

    single_key = os.environ.get("SIGIL_BISCUIT_PUBKEY", "")
    single_kid = os.environ.get("SIGIL_BISCUIT_KID", "")
    if single_key and single_kid and single_kid not in keyring:
        try:
            keyring[single_kid] = base64.b64decode(single_key)
        except Exception:  # noqa: BLE001
            pass

    return keyring


# ──────────────────────────────────────────────────────────────────────────────
# SigilTaskContext
# ──────────────────────────────────────────────────────────────────────────────


class SigilTaskContext:
    """Context manager for a single agent task.

    Usage::

        with client.task(["zep.search", "memory.store"], ttl_seconds=3600) as task:
            results = search_knowledge_base("Q4 revenue")  # @instrumented_tool

    On entry:
      1. Generates a ``task_id`` (UUIDv4).
      2. Issues a task-scoped biscuit token via ``client.issue_token``.
      3. Runs ``verify_local`` to warm the effective tool set (if keyring is set).
      4. Installs itself as the current task in a contextvar so decorators find it.

    On exit (including on exception):
      - Calls ``client._flusher.flush_all()`` to drain buffered audit events
        (best-effort; exceptions are swallowed).
      - Resets the contextvar.

    The ``biscuit_token`` is deliberately NOT exposed as a public property.
    Use ``task.task_id`` for correlation; the token stays internal.

    Do NOT share a ``SigilTaskContext`` across threads.  Each concurrent task
    must instantiate its own context.
    """

    def __init__(
        self,
        client: SigilClient,
        tool_allowlist: list[str],
        agent_id: str | None,
        service_account_id: str | None,
        ttl_seconds: int,
    ) -> None:
        self._client = client
        self._tool_allowlist = tool_allowlist
        self._agent_id_override = agent_id
        self._sa_id_override = service_account_id
        self._ttl_seconds = ttl_seconds

        self._task_id: str | None = None
        self._biscuit_token: str | None = None  # NEVER log or expose
        self._effective_tools: list[str] = []
        self._ctx_token: Any = None  # contextvars.Token returned by .set()

    def __enter__(self) -> SigilTaskContext:
        agent_id = self._agent_id_override or self._client.agent_id
        sa_id = self._sa_id_override or self._client.service_account_id
        task_id = str(uuid.uuid4())
        self._task_id = task_id

        resp = self._client.issue_token(
            agent_id=agent_id,
            service_account_id=sa_id,
            task_id=task_id,
            tool_allowlist=self._tool_allowlist,
            ttl_seconds=self._ttl_seconds,
        )
        biscuit_token: str = resp["biscuit_token"]
        self._biscuit_token = biscuit_token  # stays private

        # Warm the effective tool set via local verification.
        # If the keyring is empty we cannot verify — per-call verify_local will
        # deny each tool call with token_invalid, which is the correct behaviour
        # for a misconfigured SDK (no public key → closed by default).
        if self._client.biscuit_keyring:
            result: VerifyResult = verify_local(
                biscuit_token,
                self._client.biscuit_keyring,
                active_kid=self._client.active_kid or None,
            )
            if result.ok:
                self._effective_tools = result.effective_tools
            # ok=False: effective_tools stays []; per-call verify will deny
        else:
            # No keyring configured — expose the requested allowlist so that
            # callers in dev environments with mocked verify can still introspect
            # what was requested, but per-call verify_local will deny.
            self._effective_tools = list(self._tool_allowlist)

        # Make this context visible to decorators via the contextvar.
        self._ctx_token = _current_task.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Best-effort flush of buffered audit events for this task.
        try:
            self._client._flusher.flush_all()
        except Exception:  # noqa: BLE001
            pass

        # Reset the contextvar regardless of whether flush succeeded.
        if self._ctx_token is not None:
            _current_task.reset(self._ctx_token)
            self._ctx_token = None

    @property
    def task_id(self) -> str | None:
        """UUID of the open task, or None before entry."""
        return self._task_id

    @property
    def effective_agent_id(self) -> str:
        """The agent this task runs as: per-task override ?? client default.

        This is the agent the task's biscuit was issued for (see __enter__), so audit/prompt
        attribution must use THIS — not ``client.agent_id`` — or sigil-core rejects the row when a
        per-task agent override is set (the server enforces agent_id == the task's owning agent).
        Mirrors the TS SDK's ``effectiveAgentId``.
        """
        return self._agent_id_override or self._client.agent_id

    def handoff(
        self,
        child_agent_id: str,
        scope_delta: dict[str, Any],
        *,
        child_task_id: str | None = None,
        authorized_by_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Record an agent handoff from THIS task to *child_agent_id* (SG-9 SP-1 / ENT-91).

        Parent task + agent are taken from this active context (never client-supplied). The child
        task is the caller's to create; if *child_task_id* is omitted a fresh UUID is generated and
        returned so the caller can open the child agent's task with it.

        Returns ``{"handoff_id": ..., "child_task_id": ...}``.
        """
        if self._task_id is None:
            raise SigilDeniedError(
                "handoff() requires an active task (use 'with client.task([...]) as task:')",
                denied_reason="no_task",
                tool_name="handoff",
                task_id="",
            )
        child_task = child_task_id or str(uuid.uuid4())
        result = self._client.record_handoff(
            parent_task_id=self._task_id,
            parent_agent_id=self.effective_agent_id,
            child_agent_id=child_agent_id,
            child_task_id=child_task,
            scope_delta=scope_delta,
            authorized_by_user_id=authorized_by_user_id,
        )
        return {"handoff_id": result.get("handoff_id"), "child_task_id": child_task}

    # _biscuit_token is intentionally NOT exposed — it must never appear in
    # logs, error messages, or external interfaces.


# ──────────────────────────────────────────────────────────────────────────────
# SigilClient
# ──────────────────────────────────────────────────────────────────────────────


class SigilClient:
    """HTTP client for the Sigil agent governance SDK.

    Thread-safe: one HTTP keep-alive connection pool (``requests.Session``)
    per instance. Not async — use ``AsyncSigilClient`` (v1.1) for asyncio.

    Config resolution order for each parameter: constructor arg → environment
    variable → built-in default.

    Environment variables:

    * ``SIGIL_CORE_URL``           — sigil-core base URL (default
      ``http://sigil-core:8120``)
    * ``SIGIL_SDK_TOKEN``          — internal service auth token (**required**;
      never logged)
    * ``SIGIL_SERVICE_ACCOUNT``    — service account label (default
      ``"sigil-python-sdk"``)
    * ``SIGIL_TENANT_ID``          — owning tenant UUID
    * ``SIGIL_AGENT_ID``           — agent UUID
    * ``SIGIL_SERVICE_ACCOUNT_ID`` — active service-account UUID
    * ``SIGIL_FAIL_MODE``          — ``"closed"`` (default) or ``"open"``
    * ``SIGIL_BISCUIT_PUBKEYS``    — JSON ``{"kid": "<base64_pubkey>", ...}``
    * ``SIGIL_BISCUIT_PUBKEY`` + ``SIGIL_BISCUIT_KID`` — single-key shorthand
    * ``SIGIL_OVERFLOW_DIR``       — override disk overflow directory
      (default ``~/.sigil/overflow/``)
    * ``SIGIL_REDIS_URL``          — Redis URL for kill-switch subscriber
      (e.g. ``redis://localhost:6379/0``); subscriber disabled if unset

    **fail_mode="open" warning:** If ``fail_mode="open"`` is set at construction
    time, a ``WARNING`` is emitted immediately.  This mode is for development
    only and must never be used in production — tool calls will be allowed even
    when sigil-core is unreachable.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        internal_token: str | None = None,
        service_account: str | None = None,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        service_account_id: str | None = None,
        fail_mode: str | None = None,
        kill_switch_fail_mode: str | None = None,
        biscuit_keyring: dict[str, bytes] | None = None,
        active_kid: str | None = None,
        timeout: float = 5.0,
        approval_timeout: float = 300.0,
        approval_poll_interval: float = 2.0,
        credential_near_expiry_seconds: float = 604800.0,  # 7 days
        overflow_dir: str | None = None,
        oauth_issuer: str | None = None,
        mcp_token_leeway: float = 30.0,
    ) -> None:
        self.base_url: str = (
            base_url or os.environ.get("SIGIL_CORE_URL") or "http://sigil-core:8120"
        ).rstrip("/")
        # NEVER log _internal_token — it is the internalauth service credential.
        # L2: private attribute to prevent exposure via vars()/repr.
        self._internal_token: str = internal_token or os.environ.get("SIGIL_SDK_TOKEN") or ""
        self.service_account: str = (
            service_account or os.environ.get("SIGIL_SERVICE_ACCOUNT") or "sigil-python-sdk"
        )
        self.tenant_id: str = tenant_id or os.environ.get("SIGIL_TENANT_ID") or ""
        self.agent_id: str = agent_id or os.environ.get("SIGIL_AGENT_ID") or ""
        self.service_account_id: str = (
            service_account_id or os.environ.get("SIGIL_SERVICE_ACCOUNT_ID") or ""
        )
        self.fail_mode: str = fail_mode or os.environ.get("SIGIL_FAIL_MODE") or "closed"
        # How governed calls behave when the revocation subscriber is configured but
        # DEGRADED (can't receive kills). Default "open" = prior behavior (allow; rely on
        # the subscriber's WARNING + is_kill_switch_healthy()). "closed" = strict posture:
        # deny governed calls while the kill-switch cannot be guaranteed (see
        # _governance_check in decorators.py). Independent of fail_mode (sigil-core reach).
        self.kill_switch_fail_mode: str = (
            kill_switch_fail_mode or os.environ.get("SIGIL_KILL_SWITCH_FAIL_MODE") or "open"
        )
        self.timeout: float = timeout
        # ENT-81 (SG-4) approval gate: how long the SDK blocks polling a HIGH/CRITICAL
        # tool call's approval before giving up (fail-closed with "approval_timeout"),
        # and the interval between status polls. Kept in sync with sigil-core's server-
        # side approval TTL (300s). The server-sent expires_at is advisory (#307): this
        # local timeout is the authoritative bound so clock skew cannot over-extend a wait.
        self.approval_timeout: float = float(
            approval_timeout
            if os.environ.get("SIGIL_APPROVAL_TIMEOUT") is None
            else os.environ["SIGIL_APPROVAL_TIMEOUT"]
        )
        self.approval_poll_interval: float = float(
            approval_poll_interval
            if os.environ.get("SIGIL_APPROVAL_POLL_INTERVAL") is None
            else os.environ["SIGIL_APPROVAL_POLL_INTERVAL"]
        )
        # SG-7 (ENT-94c) near-expiry warning: when a token-issue response carries the agent
        # credential's effective "valid until" (credential_expires_at — present only once the
        # server's credential-lifecycle feature is enabled), warn if it is within this many
        # seconds, so the operator can rotate before the credential stops working. Default 7 days.
        _near_expiry_raw = (
            credential_near_expiry_seconds
            if os.environ.get("SIGIL_CREDENTIAL_NEAR_EXPIRY_SECONDS") is None
            else os.environ["SIGIL_CREDENTIAL_NEAR_EXPIRY_SECONDS"]
        )
        try:
            self.credential_near_expiry_seconds: float = float(_near_expiry_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "credential_near_expiry_seconds (SIGIL_CREDENTIAL_NEAR_EXPIRY_SECONDS) must be a "
                f"number, got {_near_expiry_raw!r}"
            ) from exc
        # Dedup: warn at most once per distinct credential expiry so per-task issuance does not
        # flood logs. A rotation/renewal changes the timestamp and re-arms the warning.
        self._near_expiry_warned: set[str] = set()
        self.biscuit_keyring: dict[str, bytes] = (
            biscuit_keyring if biscuit_keyring is not None else _parse_biscuit_keyring()
        )
        # F3: active_kid narrows kid-less token verification to the server's
        # active key only, matching the server's fallback behaviour.  Without
        # it, the SDK tries ALL keyring keys, which would accept a token signed
        # by a rotated-out key that the server would deny.
        self.active_kid: str = active_kid or os.environ.get("SIGIL_BISCUIT_ACTIVE_KID") or ""

        # SG-8 #393: MCP token-exchange config. oauth_issuer defaults to base_url (the
        # /oauth/token endpoint is served on the same sigil-core echo); override when the
        # token endpoint is fronted at a different public host — it also sets the DPoP htu.
        self.oauth_issuer: str = (
            oauth_issuer or os.environ.get("SIGIL_OAUTH_ISSUER") or self.base_url
        ).rstrip("/")
        self.mcp_token_leeway: float = float(
            mcp_token_leeway
            if os.environ.get("SIGIL_MCP_TOKEN_LEEWAY") is None
            else os.environ["SIGIL_MCP_TOKEN_LEEWAY"]
        )
        self._mcp_cache = _MCPTokenCache()
        self._dpop_key: DPoPKey | None = None
        self._dpop_lock = threading.Lock()

        if self.fail_mode not in ("closed", "open"):
            raise ValueError("fail_mode must be 'closed' or 'open'")
        if self.kill_switch_fail_mode not in ("closed", "open"):
            raise ValueError("kill_switch_fail_mode must be 'closed' or 'open'")
        # Reject NaN/inf too: NaN <= 0 is False and inf passes > 0, either of which would
        # produce a poll loop that never times out (a hung, ungated tool call).
        if not math.isfinite(self.approval_timeout) or self.approval_timeout <= 0:
            raise ValueError("approval_timeout must be a finite number > 0")
        if not math.isfinite(self.approval_poll_interval) or self.approval_poll_interval <= 0:
            raise ValueError("approval_poll_interval must be a finite number > 0")
        if not math.isfinite(self.credential_near_expiry_seconds) or self.credential_near_expiry_seconds < 0:
            raise ValueError("credential_near_expiry_seconds must be a finite number >= 0")
        # A NaN leeway would make the cache freshness test always False (mint every call); a
        # negative one would serve tokens right up to expiry. Match the other numeric guards.
        if not math.isfinite(self.mcp_token_leeway) or self.mcp_token_leeway < 0:
            raise ValueError("mcp_token_leeway must be a finite number >= 0")
        if not self._internal_token:
            raise ValueError(
                "internal_token is required. " "Pass it directly or set SIGIL_SDK_TOKEN."
            )

        # L2: warn when using plaintext HTTP to a non-loopback host.
        _parsed = urlparse(self.base_url)
        if _parsed.scheme == "http" and _parsed.hostname not in (
            "localhost",
            "127.0.0.1",
        ):
            _log.warning(
                "sigil: base_url %r uses http:// with non-localhost host; "
                "use https:// in production.",
                self.base_url,
            )
        # Same check for oauth_issuer when it diverges from base_url: mcp_token() sends the
        # agent Biscuit (the credential) to {oauth_issuer}/oauth/token, so a plaintext public
        # issuer would leak it. (When it equals base_url the check above already covered it.)
        if self.oauth_issuer != self.base_url:
            _parsed_issuer = urlparse(self.oauth_issuer)
            if _parsed_issuer.scheme == "http" and _parsed_issuer.hostname not in (
                "localhost",
                "127.0.0.1",
            ):
                _log.warning(
                    "sigil: oauth_issuer %r uses http:// with non-localhost host; the Biscuit "
                    "subject_token will be sent in plaintext. Use https:// in production.",
                    self.oauth_issuer,
                )

        # Loud warning for fail_mode=open — must be explicit, never the default.
        if self.fail_mode == "open":
            _log.warning(
                "sigil: fail_mode='open' — tool calls will be ALLOWED when sigil-core is "
                "unreachable. This is a DEVELOPMENT-ONLY setting. NEVER deploy in production."
            )

        self._session: requests.Session = requests.Session()

        # ── Pass 2 infrastructure ─────────────────────────────────────────────
        self._log_buffer: _LogBuffer = _LogBuffer()
        self._overflow: _OverflowWriter = _OverflowWriter(
            overflow_dir=overflow_dir, agent_id=self.agent_id
        )
        self._flusher: _Flusher = _Flusher(
            buffer=self._log_buffer, client=self, overflow=self._overflow
        )
        self._flusher.start()

        # Kill-switch subscriber — only if SIGIL_REDIS_URL is set.
        redis_url = os.environ.get("SIGIL_REDIS_URL", "")
        self._redis_url_configured = bool(redis_url)
        if redis_url and self.tenant_id:
            self._subscriber: _RevocationSubscriber | None = _RevocationSubscriber(
                tenant_id=self.tenant_id,
                agent_id=self.agent_id,
                redis_url=redis_url,
            )
            self._subscriber.start()
        elif redis_url and not self.tenant_id:
            # Fail LOUD: the operator configured a kill-switch but it cannot start (no
            # tenant_id → no channel), so remote revocations would be silently missed.
            self._subscriber = None
            _log.warning(
                "sigil: SIGIL_REDIS_URL is set but tenant_id is empty — kill-switch "
                "subscriber NOT started; remote revocations will NOT be enforced. "
                "Set SIGIL_TENANT_ID."
            )
        else:
            self._subscriber = None
            _log.debug(
                "sigil: SIGIL_REDIS_URL not set — revocation subscriber disabled (by config)"
            )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop background threads and do a final synchronous flush.

        Safe to call multiple times (idempotent after first call).
        """
        if self._subscriber is not None:
            self._subscriber.stop()
        self._flusher.stop()
        # Final drain — best-effort; log on failure so events aren't silently lost.
        try:
            self._flusher.flush_all()
        except Exception:  # noqa: BLE001
            _log.warning(
                "sigil: final flush on close() raised; some events may be lost",
                exc_info=True,
            )

    def __enter__(self) -> SigilClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── Revocation (used by decorators) ──────────────────────────────────────

    def is_revoked(self, agent_id: str, task_id: str | None) -> bool:
        """Return True if *agent_id* or *task_id* is in the revocation registry.

        No network call — reads the in-memory cache populated by the
        kill-switch subscriber.  Returns False if the subscriber is disabled.
        """
        if self._subscriber is None:
            return False
        return self._subscriber.is_revoked(agent_id, task_id)

    def is_kill_switch_healthy(self) -> bool:
        """True if the remote kill-switch is operational (or intentionally disabled).

        Returns False ONLY for the dangerous case: SIGIL_REDIS_URL was configured (the
        operator wants remote revocation) but the subscriber is not currently connected
        — a kill signal would be silently missed. Surface this on your service's health
        probe / a metric so a degraded kill-switch fails LOUD instead of open.
        """
        if self._subscriber is not None:
            return self._subscriber.healthy()
        # No subscriber: healthy only if the kill-switch was intentionally not configured.
        return not self._redis_url_configured

    def kill_switch_status(self) -> dict[str, Any]:
        """Diagnostic snapshot of the kill-switch subscriber for health endpoints/metrics."""
        if self._subscriber is not None:
            return {
                "enabled": True,
                "healthy": self._subscriber.healthy(),
                **self._subscriber.status(),
            }
        return {
            "enabled": False,
            "healthy": not self._redis_url_configured,
            "detail": (
                "SIGIL_REDIS_URL set but tenant_id missing — subscriber not started"
                if self._redis_url_configured
                else "revocation subscriber disabled (SIGIL_REDIS_URL not set)"
            ),
        }

    # ── Internal header helpers ───────────────────────────────────────────────

    def _base_headers(self) -> dict[str, str]:
        """Headers required by every sigil-core internal route."""
        return {
            "X-Internal-Service-Token": self._internal_token,
            "X-Internal-Service-Account": self.service_account,
            "X-Tenant-ID": self.tenant_id,
            "Content-Type": "application/json",
        }

    def _toolgate_headers(self) -> dict[str, str]:
        """Headers for the rate-limited toolgate group (/toolgate/*).

        Adds ``X-Sigil-Agent-ID`` which is REQUIRED by the rate-limiter
        middleware on the toolgate route group.
        """
        h = self._base_headers()
        if self.agent_id:
            h["X-Sigil-Agent-ID"] = self.agent_id
        return h

    # ── Task context ──────────────────────────────────────────────────────────

    def task(
        self,
        tool_allowlist: list[str],
        *,
        agent_id: str | None = None,
        service_account_id: str | None = None,
        ttl_seconds: int = 3600,
    ) -> SigilTaskContext:
        """Return a ``SigilTaskContext`` for the given tool allowlist.

        The context manager issues a task-scoped biscuit token on entry and
        flushes buffered audit events on exit (including on exception).

        Args:
            tool_allowlist: Fully-qualified tool names the task is allowed to
                call, e.g. ``["zep.search", "memory.store"]``.
            agent_id: Override the client's ``agent_id`` for this task.
                Defaults to ``self.agent_id``.
            service_account_id: Override the client's ``service_account_id``
                for this task.  Defaults to ``self.service_account_id``.
            ttl_seconds: Requested biscuit token lifetime.  sigil-core clamps
                to ``[1, 86400]``.  Defaults to 3600 (1 hour).

        Returns:
            A ``SigilTaskContext`` (not yet opened — use as a context manager).
        """
        return SigilTaskContext(
            client=self,
            tool_allowlist=tool_allowlist,
            agent_id=agent_id,
            service_account_id=service_account_id,
            ttl_seconds=ttl_seconds,
        )

    # ── Pass 1 primitives ─────────────────────────────────────────────────────

    def issue_token(
        self,
        agent_id: str,
        service_account_id: str,
        task_id: str,
        tool_allowlist: list[str],
        ttl_seconds: int,
        attenuation_root: str | None = None,
    ) -> dict[str, Any]:
        """Issue a Biscuit task token via sigil-core.

        ``POST {base}/internal/v1/sigil/tokens/issue``

        Required headers: ``X-Internal-Service-Token``,
        ``X-Internal-Service-Account``, ``X-Tenant-ID``.

        Args:
            agent_id: UUID of the registered agent.
            service_account_id: UUID of the agent's active service account.
            task_id: UUID of the task to scope the token to.
            tool_allowlist: List of fully-qualified tool names the token
                should grant (``["zep.search", "memory.store"]``).
            ttl_seconds: Requested token lifetime in seconds.
                sigil-core clamps to ``[1, 86400]``.
            attenuation_root: Optional parent biscuit token for attenuated
                child issuance. Never logged.

        Returns:
            dict with keys ``grant_id``, ``biscuit_token``,
            ``revocation_id``, ``expires_at``.  **Never log biscuit_token.**

        Raises:
            :class:`~sigil.errors.SigilTransportError`: On network failure.
            :class:`~sigil.errors.SigilAPIError`: If the server returns
                a non-201 status code.
        """
        url = f"{self.base_url}/internal/v1/sigil/tokens/issue"
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "service_account_id": service_account_id,
            "task_id": task_id,
            "tool_allowlist": tool_allowlist,
            "ttl_seconds": ttl_seconds,
        }
        if attenuation_root is not None:
            body["attenuation_root"] = attenuation_root

        try:
            resp = self._session.post(
                url,
                headers=self._base_headers(),
                json=body,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SigilTransportError(
                "issue_token: transport error reaching sigil-core",
                method="POST",
                url=url,
            ) from exc

        if resp.status_code != 201:
            # SG-7 (ENT-94c): a 422 with code="credential_rejected" means the agent's credential
            # has been rotated past its grace window or expired under enforcement. Surface a
            # distinct TERMINAL error so retry wrappers stop and the operator rotates the config.
            if resp.status_code == 422:
                try:
                    err_body = resp.json()
                except Exception:  # noqa: BLE001
                    err_body = {}
                if isinstance(err_body, dict) and err_body.get("code") == "credential_rejected":
                    raise CredentialRotatedError(
                        "issue_token: credential rejected — it has been rotated or expired; "
                        "obtain the new service_account_id/credential and reconfigure the client",
                        status_code=resp.status_code,
                    )
            raise SigilAPIError(
                f"issue_token: expected 201, got {resp.status_code}",
                status_code=resp.status_code,
            )

        # F2: wrap resp.json() — a 200 with a non-JSON body (WAF/nginx HTML)
        # would raise JSONDecodeError outside the transport try/except, escaping
        # _governance_check and dropping any audit event.  Raise SigilAPIError
        # so it routes through the existing fail-mode path.
        try:
            result: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise SigilAPIError(
                f"issue_token: response body is not valid JSON (status={resp.status_code})",
                status_code=resp.status_code,
            ) from exc
        self._warn_if_credential_near_expiry(result)
        return result

    def _get_dpop_key(self) -> DPoPKey:
        with self._dpop_lock:
            if self._dpop_key is None:
                self._dpop_key = DPoPKey()
            return self._dpop_key

    def mcp_token(
        self,
        resource: str,
        *,
        scope: list[str] | None = None,
        dpop: bool = False,
        subject_token: str | None = None,
    ) -> MCPToken:
        """Exchange the agent Biscuit for a short-lived MCP access token.

        POST {oauth_issuer}/oauth/token (RFC 8693, unauthenticated — the Biscuit is the
        credential). The Biscuit is taken from ``subject_token`` or, when omitted, from the
        current ``client.task(...)`` context. Results are cached per (resource, scope, dpop)
        until ``mcp_token_leeway`` seconds before expiry.

        Args:
            resource: the target MCP tool's registered resource_url (RFC 8707).
            scope: requested scopes (downscoped server-side to biscuit ∩ tool.allowed).
            dpop: request a DPoP-bound token; the result exposes ``proof_for(htu, htm)``.
            subject_token: explicit Biscuit override (else the current task's Biscuit).

        Returns:
            MCPToken.

        Raises:
            ValueError: no Biscuit available.
            SigilTokenExchangeError / SigilTokenExchangeDeniedError: an RFC 6749 error.
            SigilTransportError: network failure.
            SigilAPIError: an unexpected non-200 with no error object, or bad JSON.
        """
        biscuit = subject_token
        if biscuit is None:
            task = _current_task.get()
            biscuit = getattr(task, "_biscuit_token", None) if task is not None else None
        if not biscuit:
            raise ValueError(
                "mcp_token: no agent biscuit — call inside client.task(...) or pass subject_token"
            )
        # Fingerprint (not the Biscuit itself) partitions the cache per-agent so two callers with
        # different subject_tokens for the same resource/scope/dpop never share a token.
        biscuit_fp = hashlib.sha256(biscuit.encode("utf-8")).hexdigest()[:32]
        key = _MCPTokenCache.key(biscuit_fp, resource, scope, dpop)
        return self._mcp_cache.get_or_mint(
            key,
            now=time.time(),
            leeway=self.mcp_token_leeway,
            mint=lambda: self._mint_mcp_token(biscuit, resource, scope, dpop),
        )

    def _mint_mcp_token(
        self, biscuit: str, resource: str, scope: list[str] | None, dpop: bool
    ) -> MCPToken:
        url = f"{self.oauth_issuer}/oauth/token"
        data: dict[str, str] = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token_type": "urn:qwentrix:biscuit",
            "subject_token": biscuit,
            "resource": resource,
        }
        if scope:
            data["scope"] = " ".join(scope)
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        dpop_key: DPoPKey | None = None
        if dpop:
            dpop_key = self._get_dpop_key()
            headers["DPoP"] = dpop_key.proof(url, "POST")

        try:
            # allow_redirects=False: a 307/308 would re-send this POST body — which carries the
            # Biscuit (the credential) — to the redirect target. On an UNAUTHENTICATED endpoint that
            # is a credential-exfiltration vector, so never follow a redirect here; treat it as an error.
            resp = self._session.post(
                url, headers=headers, data=data, timeout=self.timeout, allow_redirects=False
            )
        except Exception as exc:  # noqa: BLE001
            raise SigilTransportError(
                "mcp_token: transport error reaching oauth endpoint", method="POST", url=url
            ) from exc

        if resp.is_redirect:
            raise SigilAPIError(
                f"mcp_token: unexpected redirect ({resp.status_code}) from oauth endpoint",
                status_code=resp.status_code,
            )
        if resp.status_code != 200:
            self._raise_token_exchange_error(resp)

        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise SigilAPIError(
                "mcp_token: response is not valid JSON", status_code=resp.status_code
            ) from exc

        access_token = str(body.get("access_token") or "")
        if not access_token:
            raise SigilAPIError("mcp_token: response missing access_token", status_code=resp.status_code)
        # A non-positive expires_in would make expires_at ≈ now, so every subsequent call would
        # treat the cached token as stale and re-mint — reject it as a malformed response.
        raw_expires_in = body.get("expires_in")
        if not isinstance(raw_expires_in, (int, float)) or isinstance(raw_expires_in, bool) or raw_expires_in <= 0:
            raise SigilAPIError(
                f"mcp_token: invalid expires_in {raw_expires_in!r}", status_code=resp.status_code
            )
        expires_in = int(raw_expires_in)
        scope_str = str(body.get("scope") or "")
        return MCPToken(
            access_token=access_token,
            token_type=str(body.get("token_type") or "Bearer"),
            scope=scope_str.split() if scope_str else [],
            expires_in=expires_in,
            expires_at=time.time() + expires_in,
            resource=resource,
            _dpop=dpop_key,
        )

    def _raise_token_exchange_error(self, resp: requests.Response) -> NoReturn:
        code, desc = "", ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                code = str(body.get("error") or "")
                desc = str(body.get("error_description") or "")
        except Exception:  # noqa: BLE001
            pass
        status = getattr(resp, "status_code", 0)
        if not code:
            raise SigilAPIError(f"mcp_token: unexpected status {status}", status_code=status)
        if code == "access_denied":
            raise SigilTokenExchangeDeniedError(code, desc, status_code=status)
        raise SigilTokenExchangeError(code, desc, status_code=status)

    def _warn_if_credential_near_expiry(self, result: dict[str, Any]) -> None:
        """Emit a one-time WARNING per distinct credential expiry when a token-issue response's
        ``credential_expires_at`` (SG-7/ENT-94c) is within ``credential_near_expiry_seconds``.

        Best-effort and non-fatal: a missing/malformed field is silently ignored so a governed
        call is never broken by this advisory path.
        """
        raw = result.get("credential_expires_at")
        if not raw or not isinstance(raw, str) or raw in self._near_expiry_warned:
            return
        try:
            # RFC3339 → aware datetime. Python 3.11 fromisoformat handles the trailing "Z".
            expires = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return
        seconds_left = (expires - datetime.now(timezone.utc)).total_seconds()
        if seconds_left <= self.credential_near_expiry_seconds:
            self._near_expiry_warned.add(raw)
            # Log the normalized (re-serialized) timestamp, never the raw server string, so a
            # rogue/MITM'd sigil-core cannot inject newlines/escape sequences into the log line.
            # Clamp days-left at 0: an already-expired credential (accepted in observe mode) must
            # not print a misleading negative "days left".
            days_left = max(0.0, seconds_left / 86400.0)
            _log.warning(
                "sigil: agent credential is near expiry (valid until %s, ~%.1f days left); "
                "rotate the credential before it stops working.",
                expires.isoformat(),
                days_left,
            )

    def preflight(
        self,
        token: str,
        tool_namespace: str,
        tool_name: str,
        args_hash: str,
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a preflight request to sigil-core and return the verdict dict.

        ``POST {base}/internal/v1/sigil/toolgate/preflight``

        Required headers: ``X-Internal-Service-Token``,
        ``X-Internal-Service-Account``, ``X-Tenant-ID``,
        ``X-Sigil-Agent-ID`` (required by the rate-limited toolgate group).

        Args:
            token: Active Biscuit task token.  Never logged.
            tool_namespace: Tool namespace, e.g. ``"zep"``.
            tool_name: Tool name within the namespace, e.g. ``"search"``.
            args_hash: SHA-256 hex of the canonical JSON of the original
                (unredacted) tool arguments.
            agent_id: Optional agent UUID to include in the request body
                for attribution.  Falls back to ``self.agent_id``.
            task_id: Optional task UUID to include in the request body.

        Returns:
            dict with ``verdict`` (``"allow"`` | ``"deny"`` | ``"approve"``),
            optional ``denied_reason``, optional ``required_role``.

        Raises:
            :class:`~sigil.errors.SigilTransportError`: On network failure.
                Pass 2 applies fail-mode logic (closed → deny, open → allow).
            :class:`~sigil.errors.SigilAPIError`: On unexpected HTTP status.
        """
        url = f"{self.base_url}/internal/v1/sigil/toolgate/preflight"
        body: dict[str, Any] = {
            "token": token,
            "tool_namespace": tool_namespace,
            "tool_name": tool_name,
            "args_hash": args_hash,
        }
        effective_agent_id = agent_id or self.agent_id
        if effective_agent_id:
            body["agent_id"] = effective_agent_id
        if task_id:
            body["task_id"] = task_id

        try:
            resp = self._session.post(
                url,
                headers=self._toolgate_headers(),
                json=body,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SigilTransportError(
                "preflight: transport error reaching sigil-core",
                method="POST",
                url=url,
            ) from exc

        if resp.status_code != 200:
            raise SigilAPIError(
                f"preflight: expected 200, got {resp.status_code}",
                status_code=resp.status_code,
            )

        # F2: wrap resp.json() so a non-JSON 200 (WAF intercept) raises
        # SigilAPIError, keeping it in the fail-mode path.
        try:
            result: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise SigilAPIError(
                f"preflight: response body is not valid JSON (status={resp.status_code})",
                status_code=resp.status_code,
            ) from exc
        return result

    def approval_status(self, approval_id: str) -> str:
        """Poll a pending approval's status (ENT-81/SG-4).

        ``GET {base}/internal/v1/sigil/toolgate/approval/{id}``

        The tenant is derived server-side from sigil-core's verified context (the
        SDK's scoped token), so an SDK can only read approvals in its own tenant.

        Args:
            approval_id: the ``approval_id`` returned by :meth:`preflight` on an
                ``"approve"`` verdict.

        Returns:
            The approval status string: ``"pending"`` | ``"approved"`` |
            ``"rejected"`` | ``"expired"``.

        Raises:
            :class:`~sigil.errors.SigilTransportError`: on network failure.
            :class:`~sigil.errors.SigilAPIError`: on any non-200 status (incl. 404
                for an unknown/cross-tenant id) or a non-JSON body. Callers MUST
                fail closed (deny) on either error.
        """
        # URL-encode the id (defence-in-depth path-injection guard; it comes from
        # sigil-core's own response but must never be interpolated raw).
        url = f"{self.base_url}/internal/v1/sigil/toolgate/approval/{quote(approval_id, safe='')}"
        try:
            resp = self._session.get(
                url,
                headers=self._toolgate_headers(),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SigilTransportError(
                "approval_status: transport error reaching sigil-core",
                method="GET",
                url=url,
            ) from exc

        if resp.status_code != 200:
            raise SigilAPIError(
                f"approval_status: expected 200, got {resp.status_code}",
                status_code=resp.status_code,
            )
        try:
            result: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise SigilAPIError(
                f"approval_status: response body is not valid JSON (status={resp.status_code})",
                status_code=resp.status_code,
            ) from exc
        status = str(result.get("status") or "")
        if not status:
            # A missing/empty status must fail fast (→ fail-closed deny at the poll site)
            # rather than looking like "pending" and stalling the poll for the full timeout.
            raise SigilAPIError(
                "approval_status: response missing 'status' field",
                status_code=resp.status_code,
            )
        return status

    def redeem_approval(self, approval_id: str, token: str) -> dict[str, Any]:
        """Redeem an APPROVED approval for a one-shot, single-use grant (ENT-82/SG-4).

        ``POST {base}/internal/v1/sigil/toolgate/approval/{id}/redeem``

        Call this exactly once after :meth:`approval_status` returns ``"approved"``.
        sigil-core derives the approved tool + agent from the approval record (never from
        this call), verifies *token* authorizes that tool and belongs to that agent, and
        atomically consumes a single-use nonce before minting a tool-scoped, short-TTL
        one-shot token. The response's ``revocation_id`` is the cryptographic proof that
        the call ran under a fresh, human-approved, single-use grant.

        Args:
            approval_id: the ``approval_id`` returned by :meth:`preflight`.
            token: the active Biscuit task token (the parent to attenuate). Never logged.

        Returns:
            dict with ``one_shot_token``, ``revocation_id``, ``expires_at``, ``tool_name``.

        Raises:
            :class:`~sigil.errors.SigilTransportError`: on network failure.
            :class:`~sigil.errors.SigilAPIError`: on any non-200 (409 = already redeemed;
                403/404/502/503 = not redeemable / unavailable). The caller MUST fail
                closed (deny) on either — a governed call must never proceed without a
                confirmed single-use redemption. The ``status_code`` distinguishes 409.
        """
        url = (
            f"{self.base_url}/internal/v1/sigil/toolgate/approval/"
            f"{quote(approval_id, safe='')}/redeem"
        )
        try:
            resp = self._session.post(
                url,
                headers=self._toolgate_headers(),
                json={"token": token},
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SigilTransportError(
                "redeem_approval: transport error reaching sigil-core",
                method="POST",
                url=url,
            ) from exc

        if resp.status_code != 200:
            raise SigilAPIError(
                f"redeem_approval: expected 200, got {resp.status_code}",
                status_code=resp.status_code,
            )
        try:
            result: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise SigilAPIError(
                f"redeem_approval: response body is not valid JSON (status={resp.status_code})",
                status_code=resp.status_code,
            ) from exc
        if not str(result.get("revocation_id") or ""):
            # No grant proof → treat as a failed redemption (fail-closed at the call site).
            raise SigilAPIError(
                "redeem_approval: response missing 'revocation_id'",
                status_code=resp.status_code,
            )
        return result

    def log_batch(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Flush a batch of audit events to sigil-core.

        ``POST {base}/internal/v1/sigil/toolgate/log-batch``

        Required headers: ``X-Internal-Service-Token``,
        ``X-Internal-Service-Account``, ``X-Tenant-ID``,
        ``X-Sigil-Agent-ID``.

        Args:
            events: List of event dicts (see docs/protocol.md §4.1).
                Raw tool arguments must NOT be included — pass
                ``args_hash`` (over original) and ``args_redacted``
                (DLP-scrubbed copy) instead.
                Maximum ``100`` events per call.

        Returns:
            dict with ``accepted`` (int) — number of events accepted by
            sigil-core.

        Raises:
            ValueError: If *events* contains more than 100 items.
            :class:`~sigil.errors.SigilTransportError`: On network failure.
            :class:`~sigil.errors.SigilAPIError`: On unexpected HTTP status.
        """
        if len(events) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"log_batch: batch size {len(events)} exceeds the maximum of "
                f"{_MAX_BATCH_SIZE} events per call"
            )

        url = f"{self.base_url}/internal/v1/sigil/toolgate/log-batch"
        body: dict[str, Any] = {"events": events}

        try:
            resp = self._session.post(
                url,
                headers=self._toolgate_headers(),
                json=body,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SigilTransportError(
                "log_batch: transport error reaching sigil-core",
                method="POST",
                url=url,
            ) from exc

        if resp.status_code != 202:
            raise SigilAPIError(
                f"log_batch: expected 202, got {resp.status_code}",
                status_code=resp.status_code,
            )

        # F2: wrap resp.json() so a non-JSON 202 (proxy/CDN body) raises
        # SigilAPIError instead of an escaping JSONDecodeError.
        try:
            result: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise SigilAPIError(
                f"log_batch: response body is not valid JSON (status={resp.status_code})",
                status_code=resp.status_code,
            ) from exc
        return result

    def log_prompt(
        self,
        task_id: str,
        agent_id: str,
        *,
        prompt_hash: str,
        prompt_redacted: dict[str, Any],
        response_hash: str,
        response_sampled: Any,
        model: str,
        model_provider: str,
        token_count_input: int = 0,
        token_count_output: int = 0,
        latency_ms: int | None = None,
    ) -> dict[str, Any]:
        """Write one prompt-log row to sigil-core (SG-9 SP-1 / ENT-91 Task Replay prompt lane).

        ``POST {base}/internal/v1/sigil/tasks/{task_id}/log-prompt``  (returns 202)

        Unlike :meth:`log_batch` (a homogeneous batch of tool-invocation events to a single
        toolgate endpoint), a prompt-log targets a per-task endpoint and is far lower-volume
        (one per governed LLM call, not one per tool op), so it is sent directly rather than
        through the audit buffer. The ``@instrumented_llm`` decorator calls this **best-effort**:
        a failure here never breaks the governed call, because the tool-invocation audit event
        remains the authoritative record.

        Tenant identity is carried by the internalauth headers (``X-Tenant-ID``); ``task_id`` is
        the path. Raw prompt/response text is NEVER sent — only ``prompt_hash``/``response_hash``
        and the DLP-redacted ``prompt_redacted``/``response_sampled`` samples.

        Args:
            task_id: The active task's UUID (path scope).
            agent_id: The calling agent's UUID (bound on the row).
            prompt_hash: SHA-256 hex of the original prompt text (64 chars).
            prompt_redacted: DLP-scrubbed copy of the call arguments.
            response_hash: SHA-256 hex of the response text (64 chars).
            response_sampled: DLP-scrubbed, truncated response sample (JSON-serialisable).
            model: Concrete model id when known (e.g. ``"gpt-4o"``), else a coarse label.
            model_provider: Provider namespace (e.g. ``"openai"``).
            token_count_input: Prompt-side token count (0 when the SDK cannot derive it).
            token_count_output: Completion-side token count (0 when unknown).
            latency_ms: Wall-clock latency of the governed LLM call, if measured.

        Returns:
            dict — sigil-core's acknowledgement body.

        Raises:
            :class:`~sigil.errors.SigilTransportError`: On network failure.
            :class:`~sigil.errors.SigilAPIError`: On a non-202 status or non-JSON body.
        """
        # quote task_id — defence-in-depth path-injection guard (mirrors approval_status).
        safe_task = quote(task_id, safe="")
        url = f"{self.base_url}/internal/v1/sigil/tasks/{safe_task}/log-prompt"
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "prompt_hash": prompt_hash,
            "prompt_redacted": prompt_redacted,
            "response_hash": response_hash,
            "response_sampled": response_sampled,
            "model": model,
            "model_provider": model_provider,
            "token_count_input": token_count_input,
            "token_count_output": token_count_output,
            "latency_ms": latency_ms,
        }

        try:
            resp = self._session.post(
                url,
                headers=self._toolgate_headers(),
                json=body,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SigilTransportError(
                "log_prompt: transport error reaching sigil-core",
                method="POST",
                url=url,
            ) from exc

        if resp.status_code != 202:
            raise SigilAPIError(
                f"log_prompt: expected 202, got {resp.status_code}",
                status_code=resp.status_code,
            )

        try:
            result: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise SigilAPIError(
                f"log_prompt: response body is not valid JSON (status={resp.status_code})",
                status_code=resp.status_code,
            ) from exc
        return result

    def record_handoff(
        self,
        *,
        parent_task_id: str,
        parent_agent_id: str,
        child_agent_id: str,
        child_task_id: str,
        scope_delta: dict[str, Any],
        authorized_by_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Record an agent handoff (SG-9 SP-1 / ENT-91 Task-Replay handoff lane).

        ``POST {base}/internal/v1/sigil/handoffs`` (returns 201). Records the handoff EVENT only —
        parent task/agent identify the spawning task; the child task is created by the caller
        out-of-band. Tenant is carried by the internalauth headers, never the body. Prefer
        :meth:`SigilTaskContext.handoff`, which fills parent task/agent from the active context.
        """
        url = f"{self.base_url}/internal/v1/sigil/handoffs"
        body: dict[str, Any] = {
            "parent_task_id": parent_task_id,
            "parent_agent_id": parent_agent_id,
            "child_agent_id": child_agent_id,
            "child_task_id": child_task_id,
            "scope_delta": scope_delta,
        }
        if authorized_by_user_id:
            body["authorized_by_user_id"] = authorized_by_user_id

        # _toolgate_headers() is reused for its X-Tenant-ID + internalauth bundle; the extra
        # X-Sigil-Agent-ID it adds is harmless here (the /handoffs route is its own group, not the
        # rate-limited toolgate group), and carrying the acting agent is useful for attribution.
        try:
            resp = self._session.post(
                url,
                headers=self._toolgate_headers(),
                json=body,
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SigilTransportError(
                "record_handoff: transport error reaching sigil-core",
                method="POST",
                url=url,
            ) from exc
        if resp.status_code != 201:
            raise SigilAPIError(
                f"record_handoff: expected 201, got {resp.status_code}",
                status_code=resp.status_code,
            )
        try:
            result: dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise SigilAPIError(
                f"record_handoff: response body is not valid JSON (status={resp.status_code})",
                status_code=resp.status_code,
            ) from exc
        return result
