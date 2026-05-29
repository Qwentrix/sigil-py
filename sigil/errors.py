"""Sigil SDK exception hierarchy."""

from __future__ import annotations


class SigilDeniedError(Exception):
    """Raised when sigil-core (or local token verification) denies a tool call.

    Attributes:
        denied_reason: Machine-readable denial code from the preflight response
            (e.g. ``"tool_not_in_scope"``, ``"agent_revoked"``).
        tool_name: Fully-qualified tool name that was denied (``"namespace.name"``).
        task_id: UUID of the task context in which the denial occurred.
    """

    def __init__(
        self,
        message: str,
        denied_reason: str,
        tool_name: str,
        task_id: str,
    ) -> None:
        super().__init__(message)
        self.denied_reason: str = denied_reason
        self.tool_name: str = tool_name
        self.task_id: str = task_id


class SigilUnreachableDeniedError(SigilDeniedError):
    """Raised when sigil-core is unreachable and ``fail_mode="closed"``.

    The SDK could not complete a preflight or log-batch call and is enforcing
    the closed-failure policy.  Overflow events are written to
    ``~/.sigil/overflow/<agent_id>_<date>.ndjson`` for later replay.
    """
