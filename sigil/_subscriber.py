"""Redis kill-switch subscriber (optional dependency).

Subscribes to the per-tenant channel::

    drm:revocation-events:{tenant_id}

and maintains a thread-safe in-memory revocation registry keyed by agent_id
or task_id.  Decorators check ``client.is_revoked(agent_id, task_id)`` before
every tool call — no network round-trip required.

**Optional dependency:** ``redis`` must be installed (``pip install sigil-py[revocation]``).
If ``SIGIL_REDIS_URL`` is unset or the ``redis`` package is not importable, the
subscriber simply does not start and the SDK continues to work normally
(fail-mode governs tool calls in the absence of a live revocation signal).

If the Redis connection drops, the subscriber reconnects with exponential
backoff (0.5 s → 30 s cap) and never crashes the host application.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from typing import Any

_log = logging.getLogger("sigil")

# Per-tenant channel prefix — matches stream_writer.go DefaultRevocationChannel
# plus the :{tenantID} suffix added by PublishRevocation.
_CHANNEL_PREFIX: str = "drm:revocation-events"

# L4: cap the revocation registry to prevent unbounded memory growth if the
# kill-switch channel is flooded.  Oldest entries are evicted FIFO.
_REVOKED_MAX: int = 10_000


class _RevocationSubscriber:
    """Background thread that listens for kill-switch events on Redis pub/sub."""

    def __init__(self, tenant_id: str, agent_id: str, redis_url: str) -> None:
        self._tenant_id = tenant_id
        self._agent_id = agent_id
        self._redis_url = redis_url
        self._channel = f"{_CHANNEL_PREFIX}:{tenant_id}"

        # {agent_id | task_id → denial_reason}  (L4: bounded OrderedDict)
        self._revoked: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the subscriber daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="sigil-revocation-sub",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and wait up to 2 s for it to exit."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def is_revoked(self, agent_id: str, task_id: str | None) -> bool:
        """Return True if *agent_id* or *task_id* appears in the revocation registry."""
        with self._lock:
            if agent_id and agent_id in self._revoked:
                return True
            return bool(task_id and task_id in self._revoked)

    def _set_revoked(self, key: str, reason: str) -> None:
        """Add or update a revocation entry; evict oldest if at cap.

        Must be called with ``self._lock`` already held.
        """
        self._revoked[key] = reason
        # L4: evict oldest entries (FIFO via OrderedDict insertion order).
        while len(self._revoked) > _REVOKED_MAX:
            self._revoked.popitem(last=False)

    def _apply(self, payload: dict[str, Any]) -> None:
        """Parse a RevocationPayload and update the in-memory registry.

        Mirrors the RevocationPayload struct from stream_writer.go:
          RevocationID, TenantID, AgentID, TaskID, Scope, Reason, RevokedAt.

        Scope semantics:
        - "tenant": revoke all agents under this tenant (we flag our own agent_id).
        - "agent":  revoke the specific agent_id — only recorded when it is OUR agent.
        - "task" / "single": revoke the task_id — only recorded when the payload's
          agent_id matches OUR agent (tasks belonging to other agents are irrelevant).

        M3: tenant guard — "agent" and "task"/"single" scopes check that the
        payload's tenant_id matches ours (or is absent for backward compat with
        messages that omit tenant_id) to prevent cross-tenant revocations.

        F4: this SDK only enforces its own agent.  Revocations for OTHER agents
        are ignored entirely so a flood of 10 k+ distinct agent revocations
        cannot FIFO-evict our own revocation from the bounded registry.
        """
        scope = payload.get("scope", "")
        reason: str = payload.get("reason") or "agent_revoked"
        pa_id: str = payload.get("agent_id", "")
        pt_id: str = payload.get("task_id", "")
        ptenant: str = payload.get("tenant_id", "")

        # Tenant guard: absent tenant_id is backward-compat allowed;
        # a non-empty tenant_id must match ours.
        tenant_ok = not ptenant or ptenant == self._tenant_id

        with self._lock:
            if scope == "tenant":
                # Flag our own agent if the tenant matches.
                if ptenant == self._tenant_id and self._agent_id:
                    self._set_revoked(self._agent_id, reason)
            elif scope == "agent":
                # M3 + F4: only record if this revocation targets OUR agent and
                # comes from our tenant.  Revocations for other agents are
                # irrelevant — ignore them to prevent flood-eviction bypass.
                if pa_id == self._agent_id and tenant_ok:
                    self._set_revoked(pa_id, reason)
            elif scope in ("task", "single"):
                # M3 + F4: only record tasks belonging to OUR agent (pa_id must
                # match self._agent_id) — tasks of other agents are irrelevant.
                if tenant_ok and pa_id == self._agent_id and pt_id:
                    self._set_revoked(pt_id, reason)
                # Also revoke the agent to cease all active work.
                if tenant_ok and pa_id == self._agent_id:
                    self._set_revoked(pa_id, reason)

    def _run(self) -> None:
        try:
            import redis as _redis  # optional dep; stubs governed by pyproject.toml mypy override
        except ImportError:
            _log.info(
                "sigil: redis package not installed — kill-switch subscriber disabled. "
                "Install with: pip install sigil-py[revocation]"
            )
            return

        backoff = 0.5
        while not self._stop.is_set():
            try:
                r: Any = _redis.from_url(self._redis_url, socket_timeout=5.0)
                ps: Any = r.pubsub(ignore_subscribe_messages=True)
                ps.subscribe(self._channel)
                backoff = 0.5  # reset on successful connect
                _log.debug("sigil: revocation subscriber connected on %s", self._channel)

                while not self._stop.is_set():
                    msg: Any = ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if msg and msg.get("type") == "message":
                        try:
                            raw: Any = msg["data"]
                            if isinstance(raw, bytes):
                                raw = raw.decode()
                            self._apply(json.loads(raw))
                        except Exception:  # noqa: BLE001
                            _log.debug(
                                "sigil: failed to parse revocation message",
                                exc_info=True,
                            )

                try:
                    ps.unsubscribe()
                    ps.close()
                except Exception:  # noqa: BLE001
                    pass

            except Exception:  # noqa: BLE001
                if self._stop.is_set():
                    break
                _log.debug(
                    "sigil: revocation subscriber error; reconnecting in %.1fs",
                    backoff,
                    exc_info=True,
                )
                # I3: use stop event instead of sleep so stop() is immediate.
                self._stop.wait(timeout=backoff)
                backoff = min(backoff * 2, 30.0)
