"""Sigil Python SDK — AI agent governance for Micelium.

Public API surface (Pass 1 + Pass 2):

    >>> from sigil import SigilClient, SigilTaskContext
    >>> from sigil import instrumented_tool, instrumented_llm
    >>> from sigil import SigilDeniedError, SigilUnreachableDeniedError
    >>> from sigil import verify_local, VerifyResult

    >>> client = SigilClient(
    ...     base_url="http://sigil-core:8120",
    ...     internal_token="...",
    ...     tenant_id="...",
    ...     agent_id="...",
    ...     biscuit_keyring={"v1": pubkey_bytes},
    ... )

    >>> @instrumented_tool("zep", "search", risk_tier="low")
    ... def search_knowledge_base(query: str) -> list[dict]:
    ...     ...

    >>> with client.task(["zep.search"], ttl_seconds=3600) as task:
    ...     results = search_knowledge_base("Q4 revenue")

Full documentation: https://sigil.micelium.com/docs
Wire protocol spec: docs/protocol.md
Kill-switch: subscribe to drm:revocation-events:{tenant_id} via SIGIL_REDIS_URL.
"""

from sigil.client import SigilClient, SigilTaskContext
from sigil.decorators import instrumented_llm, instrumented_tool
from sigil.errors import (
    SigilAPIError,
    SigilDeniedError,
    SigilTransportError,
    SigilUnreachableDeniedError,
)
from sigil.verify import VerifyResult, verify_local

__all__ = [
    "SigilClient",
    "SigilTaskContext",
    "instrumented_tool",
    "instrumented_llm",
    "SigilDeniedError",
    "SigilUnreachableDeniedError",
    "SigilTransportError",
    "SigilAPIError",
    "verify_local",
    "VerifyResult",
]

# Single source of truth — keep in sync with pyproject.toml [project] version.
__version__ = "0.1.0"
