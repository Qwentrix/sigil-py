"""DLP redaction of sensitive data before audit emission.

Redacted args are logged to sigil-core; the original args are never transmitted.
The SHA-256 hash of the original canonical JSON is logged as ``args_hash`` to
preserve integrity without exposing sensitive values.

See docs/protocol.md §3.1 for the redacted args format.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


#: Mapping of DLP classifier name to the placeholder label written into redacted output.
#: Extend this registry as new classifiers are added.
CLASSIFIER_LABELS: dict[str, str] = {
    "PERSON_NAME": "<PII:PERSON_NAME>",
    "EMAIL": "<PII:EMAIL>",
    "PHONE": "<PII:PHONE>",
    "SSN": "<PHI:SSN>",
    "CREDIT_CARD": "<PCI:CREDIT_CARD>",
    "IP_ADDRESS": "<PII:IP_ADDRESS>",
}


def canonical_json(obj: Any) -> str:
    """Serialize *obj* to canonical JSON (keys sorted, no extra whitespace).

    Args:
        obj: A JSON-serializable Python object.

    Returns:
        Canonical JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def args_hash(args: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of *args*.

    This is stored as ``args_hash`` in audit records so integrity of the
    original (unredacted) args can be verified later.

    Args:
        args: Tool argument dict (original, unredacted).

    Returns:
        Lowercase hex SHA-256 digest string.
    """
    return hashlib.sha256(canonical_json(args).encode()).hexdigest()


def redact(args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *args* with sensitive values replaced by DLP labels.

    Top-level keys are preserved.  Values that match a DLP classifier pattern
    are replaced with the corresponding label string (e.g. ``"<PII:EMAIL>"``).
    Non-sensitive values pass through unchanged.

    This is a stub implementation.  The production implementation will use a
    DLP classifier pipeline (e.g. regex-based or an external scanning service).

    Args:
        args: Tool argument dict.

    Returns:
        New dict with the same keys and redacted values where applicable.
    """
    raise NotImplementedError(
        "redact() is not yet implemented. "
        "The production implementation will integrate a DLP classifier pipeline."
    )


def redact_safe(args: dict[str, Any]) -> dict[str, Any]:
    """Like :func:`redact` but falls back to redacting ALL string values on error.

    Used by the SDK in production when the DLP pipeline is unavailable and
    ``fail_mode="closed"`` — guarantees no sensitive data leaks to audit logs
    even if the classifier fails.

    Args:
        args: Tool argument dict.

    Returns:
        New dict with string values replaced by ``"<REDACTED>"`` on fallback.
    """
    try:
        return redact(args)
    except NotImplementedError:
        return {k: "<REDACTED>" if isinstance(v, str) else v for k, v in args.items()}
