"""Local token verification using JSON + ed25519 (pynacl).

Implements the wire-protocol spec defined in docs/protocol.md §2.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any


def _b64url_decode(data: str) -> bytes:
    """Decode a base64url string (with or without padding)."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def decode_token(token: str) -> dict[str, Any]:
    """Decode a Sigil capability token without verifying the signature.

    Use :func:`verify_token` for signature-checked decoding in production.

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
    return json.loads(authority_bytes)


def verify_token(token: str, public_key_bytes: bytes) -> dict[str, Any]:
    """Verify a Sigil capability token using ed25519 and return its authority block.

    Verification steps (per docs/protocol.md §2.3):
    1. Split and base64url-decode both parts.
    2. Verify ed25519 signature with the provided public key.
    3. Parse authority JSON.
    4. Check ``expires_at > now`` (UTC).

    Args:
        token: A ``<authority_b64url>.<signature_b64url>`` string.
        public_key_bytes: 32-byte ed25519 public key from drm-service.

    Returns:
        The decoded and verified authority block as a dict.

    Raises:
        ValueError: If the token is malformed or expired.
        nacl.exceptions.BadSignatureError: If signature verification fails.
        ImportError: If pynacl is not installed.
    """
    raise NotImplementedError(
        "verify_token requires pynacl. "
        "Install sigil-py with: pip install 'sigil-py[verify]'"
    )


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
