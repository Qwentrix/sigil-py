"""MCP token-exchange result type, error types, and the per-client token cache."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from sigil.dpop import DPoPKey, ath_for
from sigil.errors import SigilAPIError

_MAX_DESC = 256


class SigilTokenExchangeError(SigilAPIError):
    """An RFC 6749 error object returned by the token-exchange endpoint."""

    def __init__(self, error_code: str, error_description: str = "", *, status_code: int = 0) -> None:
        super().__init__(f"mcp_token: token exchange failed ({error_code})", status_code=status_code)
        self.error_code = error_code
        self.error_description = (error_description or "")[:_MAX_DESC]


class SigilTokenExchangeDeniedError(SigilTokenExchangeError):
    """Terminal: the agent is revoked/quarantined (access_denied). Retrying will not help."""


CacheKey = tuple[str, str, tuple[str, ...], bool]


@dataclass(repr=False, slots=True)
class MCPToken:
    """A minted MCP access token.

    ``access_token`` is a bearer secret. ``__repr__`` omits it, and ``slots=True`` means the
    instance has no ``__dict__`` — so ``vars(tok)`` raises and a generic logger that dumps
    ``__dict__`` cannot leak the token. Read it explicitly via the ``access_token`` attribute.
    """

    access_token: str
    token_type: str  # "Bearer" | "DPoP"
    scope: list[str]
    expires_in: int
    expires_at: float  # epoch seconds (UTC), = mint_time + expires_in
    resource: str
    _dpop: Optional[DPoPKey] = field(default=None, compare=False)

    def proof_for(self, htu: str, htm: str = "POST") -> str:
        """DPoP proof for a downstream resource request (adds the ath binding)."""
        if self._dpop is None:
            raise ValueError("proof_for: this is a Bearer token (mint with dpop=True for DPoP)")
        return self._dpop.proof(htu, htm, ath=ath_for(self.access_token))

    def __repr__(self) -> str:  # never leak the access token
        return (
            f"MCPToken(token_type={self.token_type!r}, resource={self.resource!r}, "
            f"scope={self.scope!r}, expires_at={self.expires_at})"
        )


class _MCPTokenCache:
    """Per-client cache keyed by (biscuit_fp, resource, sorted-scope, dpop).

    The Biscuit fingerprint is part of the key so two callers with different subject_tokens for
    the same resource/scope/dpop never share a token (the MCP token is scoped to the issuing
    Biscuit's grants). The lock is NOT held across the network mint: it guards the store on the
    fast-path check and on the store-back, and mint() runs unlocked so a slow/hung oauth endpoint
    cannot serialize cache hits for unrelated keys. The cost is a rare benign double-mint under a
    race (both tokens are valid; the later store wins). A failed mint never writes the store.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[CacheKey, MCPToken] = {}

    @staticmethod
    def key(biscuit_fp: str, resource: str, scope: list[str] | None, dpop: bool) -> CacheKey:
        return (biscuit_fp, resource, tuple(sorted(scope or [])), dpop)

    def get_or_mint(
        self, key: CacheKey, *, now: float, leeway: float, mint: Callable[[], MCPToken]
    ) -> MCPToken:
        with self._lock:
            tok = self._store.get(key)
            if tok is not None and tok.expires_at - now > leeway:
                return tok
        # Mint OUTSIDE the lock — a blocking HTTP call must not stall cache hits for other keys.
        fresh = mint()
        with self._lock:
            # Re-check: another thread may have minted while we were out of the lock. Prefer the
            # already-stored fresh token to keep a single cached value per key.
            existing = self._store.get(key)
            if existing is not None and existing.expires_at - now > leeway:
                return existing
            # Opportunistic eviction of fully-expired entries so a long-lived process issuing tokens
            # for many distinct keys (e.g. one biscuit per task) does not grow the cache without bound.
            if len(self._store) > 1:
                self._store = {k: v for k, v in self._store.items() if v.expires_at > now}
            self._store[key] = fresh
            return fresh
