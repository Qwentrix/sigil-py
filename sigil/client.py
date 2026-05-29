"""SigilClient — HTTP client to sigil-core.

Implements the public SDK surface defined in docs-2026/agent-governance/04-requirements-sigil.md §4.1
and the wire protocol in docs/protocol.md.
"""

from __future__ import annotations

import os
from typing import Any


class SigilTaskContext:
    """Context manager for a single agent task.

    Opens a task on sigil-core (``POST /internal/v1/sigil/tasks/open``) on
    entry and closes it on exit.  Maintains the Biscuit task token for the
    duration of the task.

    Do NOT share a ``SigilTaskContext`` across threads.  Each concurrent task
    must instantiate its own context.

    Use as an async context manager via ``AsyncSigilTaskContext`` for asyncio.
    """

    def __init__(
        self,
        client: "SigilClient",
        task_type: str,
        scope: dict[str, Any],
    ) -> None:
        self._client = client
        self._task_type = task_type
        self._scope = scope
        self._task_id: str | None = None
        self._biscuit_token: str | None = None

    def __enter__(self) -> "SigilTaskContext":
        raise NotImplementedError("SigilTaskContext.__enter__ is not yet implemented.")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        raise NotImplementedError("SigilTaskContext.__exit__ is not yet implemented.")

    @property
    def task_id(self) -> str | None:
        """UUID of the open task, or None before entry."""
        return self._task_id

    @property
    def biscuit_token(self) -> str | None:
        """Active Biscuit task token, or None before entry."""
        return self._biscuit_token


class SigilClient:
    """HTTP client for the Sigil agent governance SDK.

    Thread-safe: one HTTP keep-alive connection pool per instance.
    Not async — use ``AsyncSigilClient`` (v1.1) for asyncio workflows.

    Required environment variables (used when constructor args are omitted):
    - ``SIGIL_AGENT_ID``   — UUID of the registered agent.
    - ``SIGIL_API_KEY``    — Service account credential.
    - ``SIGIL_BASE_URL``   — sigil-core base URL, e.g. ``http://sigil-core:8120``.
    - ``SIGIL_FAIL_MODE``  — ``"closed"`` (default) or ``"open"`` (dev only).

    Args:
        agent_id: UUID of the registered agent.  Defaults to ``SIGIL_AGENT_ID``.
        api_key: Service account credential.  Defaults to ``SIGIL_API_KEY``.
        base_url: sigil-core base URL.  Defaults to ``SIGIL_BASE_URL``.
        fail_mode: ``"closed"`` (deny + overflow on unreachable) or
            ``"open"`` (allow on unreachable, development only).
            Defaults to ``SIGIL_FAIL_MODE`` or ``"closed"``.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        fail_mode: str = "closed",
    ) -> None:
        self.agent_id: str = agent_id or os.environ.get("SIGIL_AGENT_ID", "")
        self.api_key: str = api_key or os.environ.get("SIGIL_API_KEY", "")
        self.base_url: str = (
            base_url or os.environ.get("SIGIL_BASE_URL", "http://sigil-core:8120")
        ).rstrip("/")
        self.fail_mode: str = fail_mode or os.environ.get("SIGIL_FAIL_MODE", "closed")

        if not self.agent_id:
            raise ValueError(
                "agent_id is required. Pass it directly or set SIGIL_AGENT_ID."
            )
        if not self.api_key:
            raise ValueError(
                "api_key is required. Pass it directly or set SIGIL_API_KEY."
            )
        if self.fail_mode not in ("closed", "open"):
            raise ValueError("fail_mode must be 'closed' or 'open'.")

    def task(self, task_type: str, scope: dict[str, Any]) -> SigilTaskContext:
        """Return a ``SigilTaskContext`` for the given task type and scope.

        The context manager opens a task on sigil-core when entered and closes
        it (including on exception) when exited.

        Args:
            task_type: Human-readable label, e.g. ``"summarize-document"``.
            scope: Task scope dict.  At minimum ``{"tools": [...], "ttl_seconds": N}``.

        Returns:
            A ``SigilTaskContext`` (not yet opened).
        """
        return SigilTaskContext(client=self, task_type=task_type, scope=scope)

    def preflight(
        self,
        task_id: str,
        tool_name: str,
        args_hash: str,
        args_redacted: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a preflight request to sigil-core and return the verdict dict.

        Used internally by instrumented_tool for ``risk_tier >= high`` calls.

        Args:
            task_id: Active task UUID.
            tool_name: Fully-qualified tool name.
            args_hash: SHA-256 hex of canonical JSON of original args.
            args_redacted: Redacted args dict.

        Returns:
            Preflight response dict with at least a ``verdict`` key.

        Raises:
            SigilUnreachableDeniedError: If unreachable and ``fail_mode="closed"``.
        """
        raise NotImplementedError("SigilClient.preflight is not yet implemented.")

    def log_batch(self, events: list[dict[str, Any]]) -> list[str]:
        """Flush a batch of audit events to sigil-core.

        Args:
            events: List of event dicts (see docs/protocol.md §4.1).

        Returns:
            List of server-assigned invocation_ids.

        Raises:
            SigilUnreachableDeniedError: If unreachable and ``fail_mode="closed"``.
        """
        raise NotImplementedError("SigilClient.log_batch is not yet implemented.")
