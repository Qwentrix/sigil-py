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


class SigilTransportError(Exception):
    """Raised when an HTTP call to sigil-core fails at the transport level.

    This covers connection errors, timeouts, and DNS failures — situations
    where no HTTP response was received from the server.

    Pass 2 catches this exception and applies fail-mode logic:
    ``closed`` → raise :class:`SigilUnreachableDeniedError`;
    ``open`` → allow (development/degraded mode only).

    Attributes:
        method: HTTP method of the failed request (e.g. ``"POST"``).
        url: Full URL of the failed request.
    """

    def __init__(self, message: str, *, method: str = "", url: str = "") -> None:
        super().__init__(message)
        self.method: str = method
        self.url: str = url


class SigilAPIError(Exception):
    """Raised when sigil-core returns an unexpected HTTP status code.

    Distinct from :class:`SigilTransportError`: the request reached the server
    and received a response, but the status code was not the expected success
    code.

    Attributes:
        status_code: HTTP status code returned by sigil-core.
    """

    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code: int = status_code
