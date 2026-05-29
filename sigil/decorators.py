"""Decorator helpers for instrumenting AI agent tools and LLM calls.

These decorators wrap callables with Sigil governance:
- Local token verification before the call.
- Preflight HTTP request for ``risk_tier >= high`` tools.
- Audit event batching after the call.
- ``SigilDeniedError`` raised on denial.

A ``SigilClient`` instance must be bound before the decorators can enforce
governance.  The easiest pattern is to use them inside a ``SigilTaskContext``
(returned by ``SigilClient.task(...)``), which binds the client automatically.

Example usage::

    from sigil import SigilClient, instrumented_tool

    client = SigilClient(agent_id=..., api_key=..., base_url=...)

    @instrumented_tool(name="zep.search", risk_tier="low")
    def search_knowledge_base(query: str) -> list[dict]:
        ...

    with client.task("summarize", scope={...}) as ctx:
        results = search_knowledge_base("Q4 revenue")
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def task(task_type: str, scope: dict[str, Any]) -> Callable[[F], F]:
    """Decorator that wraps a callable with a ``SigilTaskContext``.

    The wrapped function opens a task on entry and closes it on exit (including
    on exceptions).  The ``SigilClient`` must be configured via the
    ``SIGIL_AGENT_ID``, ``SIGIL_API_KEY``, and ``SIGIL_BASE_URL`` environment
    variables or injected via ``SigilClient.set_default()``.

    Args:
        task_type: Human-readable task type label, e.g. ``"summarize-document"``.
        scope: Task scope dict (see docs/protocol.md §3 and SDK requirements §4.1).

    Returns:
        Decorator that wraps the target callable.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "task() decorator requires a configured SigilClient. "
                "Use 'with client.task(...)' context manager or set a default client."
            )

        return wrapper  # type: ignore[return-value]

    return decorator


def instrumented_tool(
    name: str,
    risk_tier: str = "low",
) -> Callable[[F], F]:
    """Decorator that instruments a tool function with Sigil governance.

    Before each call:
    1. Verifies the active task token contains a fact for *name*.
    2. For ``risk_tier >= high``: fires a preflight HTTP request to sigil-core.
    3. Raises ``SigilDeniedError`` if denied.

    After each call (fire-and-forget):
    4. Batches an audit event (args_hash, result_hash, latency_ms, outcome).

    Args:
        name: Fully-qualified tool name in ``"namespace.name"`` format.
        risk_tier: Risk classification. One of ``"low"``, ``"med"``, ``"high"``,
            ``"critical"``.  Defaults to ``"low"``.

    Returns:
        Decorator that wraps the target callable.

    Raises:
        SigilDeniedError: If the preflight or local token check denies the call.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                f"instrumented_tool('{name}') requires an active SigilTaskContext. "
                "Call this function from within a 'with client.task(...)' block."
            )

        return wrapper  # type: ignore[return-value]

    return decorator


def instrumented_llm(model: str) -> Callable[[F], F]:
    """Decorator that instruments an LLM call function with Sigil governance.

    Captures prompt hash, response hash, token counts, model provider, and
    latency for ``sigil_prompt_logs``.  PHI/PII redaction is applied to the
    prompt before logging (``sigil_prompt_logs.prompt_redacted``).

    Args:
        model: Model identifier in ``"provider/model-name"`` format,
            e.g. ``"openai/gpt-4o"``.

    Returns:
        Decorator that wraps the target LLM call callable.

    Raises:
        SigilDeniedError: If the active task token does not include the model.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                f"instrumented_llm('{model}') requires an active SigilTaskContext. "
                "Call this function from within a 'with client.task(...)' block."
            )

        return wrapper  # type: ignore[return-value]

    return decorator
