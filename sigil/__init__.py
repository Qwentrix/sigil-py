"""Sigil Python SDK — AI agent governance for Micelium.

Public API surface (v1):

    >>> from sigil import SigilClient, task, instrumented_tool, instrumented_llm
    >>> from sigil import SigilDeniedError

    >>> client = SigilClient(agent_id="...", api_key="...", base_url="http://sigil-core:8120")
    >>> with client.task("summarize-document", scope={"tools": ["zep.search"], "ttl_seconds": 600}) as ctx:
    ...     results = search(query="Q4 results")  # wrapped with @instrumented_tool

Full documentation: https://sigil.micelium.com/docs
Wire protocol spec: docs/protocol.md
"""

from sigil.client import SigilClient, SigilTaskContext
from sigil.decorators import instrumented_llm, instrumented_tool, task
from sigil.errors import SigilDeniedError, SigilUnreachableDeniedError

__all__ = [
    "SigilClient",
    "SigilTaskContext",
    "task",
    "instrumented_tool",
    "instrumented_llm",
    "SigilDeniedError",
    "SigilUnreachableDeniedError",
]

__version__ = "0.1.0"
