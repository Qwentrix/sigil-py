"""Local biscuit token verification using JSON + ed25519 (PyNaCl).

Implements the DRM biscuit wire-protocol defined in docs/protocol.md §2 and
mirrored exactly in shared/biscuit/biscuit.go + services/drm-service/internal/
services/biscuit_service.go.

Two public verification paths:

* :func:`verify_token` — legacy single-key helper for the old authority-block
  format (pre-BiscuitToken era).  Kept for backward compatibility.
* :func:`verify_local` — the new sub-millisecond local verifier for the full
  BiscuitToken format (``v``/``blocks``/``kid`` envelope).  This is the path
  the decorators will use in Pass 2.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

if TYPE_CHECKING:
    from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_CHECK_EXPIRY_PREFIX: str = "check if time($t), $t < "


def _b64url_decode(data: str) -> bytes:
    """Decode a base64url string (no-padding / raw-URL variant)."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


# ──────────────────────────────────────────────────────────────────────────────
# Public API — legacy single-key helpers (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────


def decode_token(token: str) -> dict[str, Any]:
    """Decode a Sigil capability token without verifying the signature.

    Use :func:`verify_token` for signature-checked decoding in production, or
    :func:`verify_local` for the new BiscuitToken format.

    Args:
        token: A ``<authority_b64url>.<signature_b64url>`` string.

    Returns:
        The decoded authority block as a dict.

    Raises:
        ValueError: If the token is malformed.
    """
    parts = token.split(".")
    if len(parts) != 2:
        raise ValueError("Malformed token: expected exactly one '.' separator")
    authority_bytes = _b64url_decode(parts[0])
    return json.loads(authority_bytes)  # type: ignore[no-any-return]


def verify_token(token: str, public_key_bytes: bytes) -> dict[str, Any]:
    """Verify a Sigil capability token using ed25519 and return its authority block.

    This helper targets the **old authority-block format** (flat JSON with
    top-level ``facts`` and ``expires_at`` fields). For the current BiscuitToken
    format (``v``/``blocks``/``kid``) use :func:`verify_local` instead.

    Verification steps (per docs/protocol.md §2.3):
    1. Split and base64url-decode both parts.
    2. Verify ed25519 signature with the provided public key.
    3. Parse authority JSON.
    4. Check ``expires_at > now`` (UTC) if present.

    Args:
        token: A ``<authority_b64url>.<signature_b64url>`` string.
        public_key_bytes: 32-byte ed25519 public key from drm-service.

    Returns:
        The decoded and verified authority block as a dict.

    Raises:
        ValueError: If the token is malformed or expired.
        nacl.exceptions.BadSignatureError: If signature verification fails.
    """
    parts = token.split(".")
    if len(parts) != 2:
        raise ValueError("Malformed token: expected exactly one '.' separator")

    payload_bytes = _b64url_decode(parts[0])
    sig_bytes = _b64url_decode(parts[1])

    vk = VerifyKey(public_key_bytes)
    vk.verify(payload_bytes, sig_bytes)  # raises BadSignatureError on failure

    authority: dict[str, Any] = json.loads(payload_bytes)

    # Check expiry using the old top-level expires_at field (legacy format).
    expires_at_str: str = authority.get("expires_at", "")
    if expires_at_str:
        expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        if datetime.now(tz=timezone.utc) >= expires_at:
            raise ValueError(f"Token expired at {expires_at_str}")

    return authority


def has_tool_fact(authority: dict[str, Any], tool_name: str) -> bool:
    """Return True if the authority block contains a tool fact for *tool_name*.

    Fact encoding: ``tool("<namespace>.<name>")``

    Args:
        authority: Decoded authority block (output of :func:`verify_token`).
        tool_name: Fully-qualified tool name, e.g. ``"zep.search"``.

    Returns:
        True if the tool is present in the token's fact list.
    """
    expected_fact = f'tool("{tool_name}")'
    return expected_fact in authority.get("facts", [])


def is_expired(authority: dict[str, Any]) -> bool:
    """Return True if the token's ``expires_at`` is in the past (UTC).

    Args:
        authority: Decoded authority block.

    Returns:
        True if the token has expired.

    Raises:
        ValueError: If ``expires_at`` is missing or unparseable.
    """
    expires_at_str: str = authority.get("expires_at", "")
    if not expires_at_str:
        raise ValueError("Token authority block missing 'expires_at' field")
    expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
    return datetime.now(tz=timezone.utc) >= expires_at


# ──────────────────────────────────────────────────────────────────────────────
# BiscuitToken types — mirror Go shared/biscuit/biscuit.go
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TokenBlock:
    """One authority or attenuation block in a BiscuitToken.

    Field names mirror the JSON tags in biscuit.go (TokenBlock struct):
    ``facts``, ``checks``, ``rid``, ``idx``.
    """

    facts: list[str]
    checks: list[str]
    rid: str
    idx: int


@dataclass
class VerifyResult:
    """Result of a local biscuit token verification via :func:`verify_local`.

    Attributes:
        ok: True if all checks passed; False on any failure.
        effective_tools: Intersection of tool facts across all blocks (H-3).
            Empty when ``ok=False`` for most failure modes.
        agent_id: ``agent("<uuid>")`` fact from the authority block, or None.
        task_id: ``task("<uuid>")`` fact from the authority block, or None.
        tenant_id: ``tenant("<uuid>")`` fact from the authority block, or None.
        expires_at: Earliest expiry across all blocks (H-2), or None.
        reason: Machine-readable denial reason when ``ok=False``.  Matches the
            server's ``denied_reason`` vocabulary:
            ``token_invalid`` | ``task_expired`` |
            ``tool_not_in_scope`` | ``tenant_mismatch``.
    """

    ok: bool
    effective_tools: list[str] = field(default_factory=list)
    agent_id: str | None = None
    task_id: str | None = None
    tenant_id: str | None = None
    expires_at: datetime | None = None
    reason: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Internal BiscuitToken helpers (mirroring Go biscuit.go / biscuit_service.go)
# ──────────────────────────────────────────────────────────────────────────────


def _parse_expiry_from_checks(checks: list[str]) -> datetime | None:
    """Return the RFC3339 expiry from a block's check strings, or None.

    Mirrors Go ``parseExpiryFromChecks`` (biscuit_service.go:786-797).

    Intentional behaviour: unparseable timestamps are SKIPPED (not denied).
    This mirrors ``biscuit_service.go:528-530`` which uses ``continue`` on
    parse error so that tokens from older issuers are not silently broken.
    """
    for check in checks:
        if check.startswith(_CHECK_EXPIRY_PREFIX):
            ts_str = check[len(_CHECK_EXPIRY_PREFIX) :].strip('"')
            try:
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                continue  # skip — mirror Go's continue on parse error
    return None


def _most_restrictive_expiry(blocks: list[TokenBlock]) -> datetime | None:
    """Return the earliest expiry across ALL blocks (H-2 most-restrictive wins).

    Mirrors Go ``mostRestrictiveExpiry`` (biscuit_service.go:769-780).
    """
    result: datetime | None = None
    for block in blocks:
        exp = _parse_expiry_from_checks(block.checks)
        if exp is None:
            continue
        if result is None or exp < result:
            result = exp
    return result


def _extract_tool_names_from_facts(facts: list[str]) -> list[str]:
    """Parse ``tool("<name>")`` facts and return the tool name strings.

    Mirrors Go ``extractAgentToolsFromFacts`` (biscuit_service.go:571-585).
    """
    tools: list[str] = []
    for f in facts:
        if f.startswith("tool(") and f.endswith(")"):
            inner = f[len("tool(") : -1]
            name = inner.strip('"')
            if name:
                tools.append(name)
    return tools


def _extract_identity_from_facts(
    facts: list[str],
) -> tuple[str | None, str | None, str | None]:
    """Extract (tenant_id, agent_id, task_id) from a block's fact list.

    Mirrors Go ``extractTenantFromFacts`` / ``extractAgentTaskFromFacts``
    (biscuit_service.go:604-615, 587-602).
    """
    tenant_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    for f in facts:
        if f.startswith("tenant(") and f.endswith(")"):
            tenant_id = f[len("tenant(") : -1].strip('"')
        elif f.startswith("agent(") and f.endswith(")"):
            agent_id = f[len("agent(") : -1].strip('"')
        elif f.startswith("task(") and f.endswith(")"):
            task_id = f[len("task(") : -1].strip('"')
    return tenant_id, agent_id, task_id


def _compute_effective_tools(blocks: list[TokenBlock]) -> list[str]:
    """Effective tool set = INTERSECTION of tool facts across all blocks (H-3).

    Mirrors Go ``computeEffectiveCaps`` (biscuit.go:259-269,
    biscuit_service.go:656-668). Attenuation can only shrink scope: a tool
    present in block 0 but absent in block 1 is NOT in the effective set.
    """
    if not blocks:
        return []
    effective: set[str] = set(_extract_tool_names_from_facts(blocks[0].facts))
    for block in blocks[1:]:
        effective &= set(_extract_tool_names_from_facts(block.facts))
    return list(effective)


# ──────────────────────────────────────────────────────────────────────────────
# verify_local — the main sub-millisecond local verification path
# ──────────────────────────────────────────────────────────────────────────────


def verify_local(
    token: str,
    keyring: dict[str, bytes],
    *,
    active_kid: str | None = None,
    required_tool: str | None = None,
    expected_tenant: str | None = None,
    now: datetime | None = None,
) -> VerifyResult:
    """Verify a DRM biscuit token locally (<1 ms, no network).

    Wire format: ``base64url_nopad(json_bytes) "." base64url_nopad(ed25519_sig)``.

    Verification order (mirrors Go ``ParseAndVerify`` + ``VerifyCapability``
    in biscuit.go:108-198):

    1. Split on ``"."`` — exactly one separator required.
    2. base64url-decode both halves (raw/no-padding variant).
    3. JSON-parse the payload to extract ``kid``; fail closed if unknown.
    4. Resolve 32-byte ed25519 public key from *keyring* via ``kid``.
    5. ``VerifyKey.verify(payload_bytes, sig_bytes)`` — raises on bad sig.
    6. Parse ``BiscuitToken`` structure; fail if ``blocks`` is empty.
    7. Check expiry across ALL blocks (H-2); most-restrictive (earliest) wins.
    8. Compute effective tools = INTERSECTION across all blocks (H-3).
    9. Check ``expected_tenant`` against authority-block tenant fact.
    10. Check ``required_tool`` is in the effective set.

    **SEC-01 re-marshal NOT reproduced — why it is not needed here:**

    The Go verifier (``biscuit.go:143-148``) performs a canonical round-trip
    check: after ed25519 signature verification it ``json.Marshal``\\s the
    parsed struct and byte-compares the result to the original payload
    (SEC-01). This check is intentionally NOT reproduced in Python for two
    reasons:

    1. **Authenticity is fully covered by the ed25519 signature over the raw
       payload bytes.** Any modification to those bytes — including
       "structurally equivalent" JSON variants such as key reordering or
       whitespace normalisation — produces a *different* byte string. That
       different byte string *fails* the ed25519 signature check, because
       the signature is cryptographically bound to the exact bytes the server
       produced. The SEC-01 re-marshal adds no security when the signature
       already covers the byte stream.

    2. **Python's ``json`` module cannot reproduce Go's byte-identical
       output.** Go's ``encoding/json`` serialises struct fields in
       declaration order (``v``, ``blocks``, ``kid`` for ``BiscuitToken``),
       with specific number formatting and no whitespace. Python's
       ``json.dumps`` sorts keys alphabetically or preserves dict insertion
       order, neither of which matches Go's output. A byte-comparison would
       therefore always fail for valid tokens issued by drm-service.

    Args:
        token: DRM biscuit wire-format token string.
        keyring: Mapping of key-id → 32-byte raw ed25519 public key.
            The token's ``kid`` field selects the verification key.
            If ``kid`` is absent or not in *keyring*, fail closed with
            ``token_invalid``.
        active_kid: When set, kid-less tokens (``kid==""``) are verified
            against ONLY ``keyring[active_kid]``, matching the server's
            active-key-only fallback.  Falls back to trying all keys when
            ``active_kid`` is ``None`` (no-rotation / dev environments).
            Tokens with a non-empty unknown kid always fail closed regardless.
        required_tool: If provided, the fully-qualified tool name
            (``"namespace.name"``) that must appear in the effective
            capability set.  Absent → ``ok=False`` / ``tool_not_in_scope``.
        expected_tenant: If provided, the token's ``tenant("<uuid>")``
            fact must match exactly.  Mismatch → ``tenant_mismatch``.
        now: Override the current UTC time for testing.  Defaults to
            ``datetime.now(tz=timezone.utc)``.

    Returns:
        :class:`VerifyResult` with ``ok=True`` on success, or ``ok=False``
        and a ``reason`` string from the server's denied_reason vocabulary:
        ``token_invalid`` | ``task_expired`` |
        ``tool_not_in_scope`` | ``tenant_mismatch``.
    """
    _now = now if now is not None else datetime.now(tz=timezone.utc)

    # ── Step 1: split on exactly one "." ─────────────────────────────────────
    # Use split(".", 1) to mirror Go's strings.SplitN(token, ".", 2).
    parts = token.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return VerifyResult(ok=False, reason="token_invalid")

    payload_b64, sig_b64 = parts[0], parts[1]

    # ── Step 2: base64url-decode (raw / no-padding) ──────────────────────────
    try:
        payload_bytes = _b64url_decode(payload_b64)
        sig_bytes = _b64url_decode(sig_b64)
    except Exception:  # noqa: BLE001
        return VerifyResult(ok=False, reason="token_invalid")

    # ── Step 3: JSON-parse payload to extract kid ─────────────────────────────
    try:
        raw: dict[str, Any] = json.loads(payload_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return VerifyResult(ok=False, reason="token_invalid")

    kid: str = raw.get("kid", "")

    # ── Step 4: resolve verification key(s) from keyring ─────────────────────
    #
    # Non-empty kid → look up exactly that key; unknown kid → fail closed.
    # kid=="" (legacy / pre-kid tokens):
    #   F3: when active_kid is set, verify against ONLY keyring[active_kid] so
    #   the SDK matches the server's active-key-only fallback.  A token signed
    #   by a rotated-out key still in the SDK keyring is accepted locally but
    #   rejected server-side — this closes that parity gap.
    #   When active_kid is None (no rotation configured), fall back to trying
    #   ALL keys (original L1 behaviour, dev/no-rotation environments only).
    if kid:
        candidate_key = keyring.get(kid)
        if candidate_key is None:
            return VerifyResult(ok=False, reason="token_invalid")
        verify_keys: list[bytes] = [candidate_key]
    else:
        if active_kid is not None:
            # F3: match server authority — only the active key for kid-less tokens.
            active_key = keyring.get(active_kid)
            if active_key is None:
                return VerifyResult(ok=False, reason="token_invalid")
            verify_keys = [active_key]
        else:
            # No active_kid configured — try all keys (fall-back, no rotation).
            verify_keys = list(keyring.values())
            if not verify_keys:
                return VerifyResult(ok=False, reason="token_invalid")

    # ── Step 5: ed25519 signature verification over RAW payload bytes ─────────
    #
    # Authenticity guarantee: the signature is bound to the exact bytes the
    # server produced via json.Marshal.  Any byte-level change — including
    # "structurally equivalent" JSON re-encodings — produces a different byte
    # sequence that fails this check.  No additional canonical re-marshal is
    # needed (see module docstring SEC-01 note above).
    verified = False
    for key_bytes in verify_keys:
        try:
            vk = VerifyKey(key_bytes)
            vk.verify(payload_bytes, sig_bytes)
            verified = True
            break
        except BadSignatureError:
            continue
        except Exception:  # noqa: BLE001  — e.g. invalid key length
            continue

    if not verified:
        return VerifyResult(ok=False, reason="token_invalid")

    # ── Step 6: parse BiscuitToken structure ──────────────────────────────────
    try:
        raw_blocks: list[dict[str, Any]] = raw.get("blocks") or []
        if not raw_blocks:
            return VerifyResult(ok=False, reason="token_invalid")
        blocks: list[TokenBlock] = [
            TokenBlock(
                facts=b.get("facts") or [],
                checks=b.get("checks") or [],
                rid=b.get("rid", ""),
                idx=b.get("idx", i),
            )
            for i, b in enumerate(raw_blocks)
        ]
    except Exception:  # noqa: BLE001
        return VerifyResult(ok=False, reason="token_invalid")

    # ── Step 7: expiry check across ALL blocks (H-2 most-restrictive) ─────────
    expires_at = _most_restrictive_expiry(blocks)
    if expires_at is not None and _now >= expires_at:
        return VerifyResult(ok=False, reason="task_expired", expires_at=expires_at)

    # ── Step 8: effective tools = INTERSECTION across all blocks (H-3) ────────
    effective_tools = _compute_effective_tools(blocks)

    # ── Step 9: identity facts from authority block (block 0) ─────────────────
    tenant_id, agent_id, task_id = _extract_identity_from_facts(blocks[0].facts)

    # ── Step 10: tenant guard ─────────────────────────────────────────────────
    if expected_tenant is not None and tenant_id != expected_tenant:
        return VerifyResult(
            ok=False,
            reason="tenant_mismatch",
            agent_id=agent_id,
            task_id=task_id,
            tenant_id=tenant_id,
            expires_at=expires_at,
        )

    # ── Step 11: required tool scope check ───────────────────────────────────
    if required_tool is not None and required_tool not in effective_tools:
        return VerifyResult(
            ok=False,
            reason="tool_not_in_scope",
            effective_tools=effective_tools,
            agent_id=agent_id,
            task_id=task_id,
            tenant_id=tenant_id,
            expires_at=expires_at,
        )

    return VerifyResult(
        ok=True,
        effective_tools=effective_tools,
        agent_id=agent_id,
        task_id=task_id,
        tenant_id=tenant_id,
        expires_at=expires_at,
    )
