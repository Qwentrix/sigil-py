"""DLP redaction pipeline for PII/PHI/secret scrubbing before audit emission.

**args_hash vs args_redacted contract:**

``args_hash`` is computed over the **original, unredacted** canonical JSON of
the tool arguments (see :func:`args_hash`).  This hash is sent to sigil-core in
the preflight and log-batch requests so that the server can verify argument
integrity without storing sensitive data.

``args_redacted`` is the *human-readable* audit copy produced by :func:`redact`.
It replaces sensitive leaf strings with typed placeholders (``<PII:SSN>``,
``<PII:EMAIL>``, etc.) so that the audit log never contains raw PII.

The SDK **must** call ``args_hash(original_args)`` BEFORE calling
``redact(original_args)`` and pass both values separately.  Passing the
redacted copy to ``args_hash`` breaks server-side integrity verification because
the hash would not match the original bytes.

See docs/protocol.md §3.1 for the redacted-args format.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Placeholder labels  (exposed so callers can reference them without hardcoding)
# ──────────────────────────────────────────────────────────────────────────────

#: Mapping of classifier name → placeholder label inserted into redacted output.
#:
#: M4 note: ``PERSON_NAME`` is intentionally absent — person-name detection
#: requires NER (out of scope for a regex pipeline).  ``IP_ADDRESS`` is now
#: backed by a real regex (see ``_IP_RE``).
CLASSIFIER_LABELS: dict[str, str] = {
    "SSN": "<PII:SSN>",
    "EMAIL": "<PII:EMAIL>",
    "PHONE": "<PII:PHONE>",
    "CREDIT_CARD": "<PII:CC>",
    "SECRET": "<PII:SECRET>",
    "IP_ADDRESS": "<PII:IP_ADDRESS>",
}

# ──────────────────────────────────────────────────────────────────────────────
# Compiled DLP regex patterns  (applied in declaration order)
# ──────────────────────────────────────────────────────────────────────────────

# Social Security Number: NNN-NN-NNNN (dashes required — distinguishes from CC)
_SSN_RE: re.Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Email address
_EMAIL_RE: re.Pattern[str] = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# US phone number: optional +1 country code, 10 digits with common separators.
# Negative lookbehind/lookahead prevent matching inside longer digit strings.
_PHONE_RE: re.Pattern[str] = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[\s.\-]?)?"  # optional +1 country code
    r"(?:\(\d{3}\)|\d{3})"  # area code (with or without parens)
    r"[\s.\-]?\d{3}[\s.\-]?\d{4}"  # 7-digit subscriber number
    r"(?!\d)"
)

# Credit card: grouped 4-4-4-{1,4} with space/hyphen separators, OR
# compact 13-16 digits.  Applied AFTER SSN/phone so those are already replaced.
_CC_RE: re.Pattern[str] = re.compile(
    r"\b(?:"
    r"\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{1,4}"  # grouped: 4-4-4-{1,4}
    r"|\d{13,16}"  # compact 13-16 consecutive digits
    r")\b"
)

# AWS access key ID: well-known 4-char prefix followed by 16 uppercase alphanumerics.
_AWS_KEY_RE: re.Pattern[str] = re.compile(r"(?:AKIA|ASIA|AROA|AIPA|AIDA|AIFA)[A-Z0-9]{16}")

# Bearer token in Authorization header value.
_BEARER_RE: re.Pattern[str] = re.compile(
    r"Bearer\s+[A-Za-z0-9\-._~+/=]{20,}",
    re.IGNORECASE,
)

# API key assignments: api_key=VALUE, api-key: VALUE, apikey=VALUE.
_API_KEY_RE: re.Pattern[str] = re.compile(
    r"api[_\-]?key\s*[:=]\s*[A-Za-z0-9\-._~+/=]{16,}",
    re.IGNORECASE,
)

# IP address — IPv4 (strict octet validation) and common IPv6 forms.
_IP_RE: re.Pattern[str] = re.compile(
    r"(?:"
    # IPv4: four octets of 0-255 separated by dots.
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
    r"|"
    # IPv6 full form: eight groups of 1-4 hex digits.
    r"(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}"
    r"|"
    # IPv6 compressed with :: (zero or more groups on each side).
    r"(?:[0-9A-Fa-f]{1,4}:)*::(?:(?:[0-9A-Fa-f]{1,4}:)*[0-9A-Fa-f]{0,4})?"
    r")"
)

# ──────────────────────────────────────────────────────────────────────────────
# Internal string redaction helper
# ──────────────────────────────────────────────────────────────────────────────


def _redact_str(text: str) -> str:
    """Apply the full DLP regex pipeline to a single string value.

    Patterns are applied in order; once a span is replaced the placeholder
    text is not re-scanned by subsequent patterns.
    """
    text = _SSN_RE.sub(CLASSIFIER_LABELS["SSN"], text)
    text = _EMAIL_RE.sub(CLASSIFIER_LABELS["EMAIL"], text)
    text = _PHONE_RE.sub(CLASSIFIER_LABELS["PHONE"], text)
    text = _CC_RE.sub(CLASSIFIER_LABELS["CREDIT_CARD"], text)
    text = _AWS_KEY_RE.sub(CLASSIFIER_LABELS["SECRET"], text)
    text = _BEARER_RE.sub(CLASSIFIER_LABELS["SECRET"], text)
    text = _API_KEY_RE.sub(CLASSIFIER_LABELS["SECRET"], text)
    text = _IP_RE.sub(CLASSIFIER_LABELS["IP_ADDRESS"], text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def canonical_json(obj: Any) -> str:
    """Serialize *obj* to canonical JSON (keys sorted, no extra whitespace).

    Args:
        obj: A JSON-serializable Python object.

    Returns:
        Canonical JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def args_hash(args: Any) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of *args*.

    **Always pass the ORIGINAL, unredacted args.**  Passing the redacted copy
    produces a different hash that does not match the server's integrity check.

    This is stored as ``args_hash`` in preflight and log-batch requests so
    that sigil-core can verify the integrity of the original tool arguments
    without storing sensitive values.

    Args:
        args: Tool argument value — typically a dict, but may be any
            JSON-serializable object (original, unredacted).

    Returns:
        Lowercase hex SHA-256 digest string.
    """
    return hashlib.sha256(canonical_json(args).encode()).hexdigest()


def redact(obj: Any) -> Any:
    """Replace PII/secret leaves with typed placeholders recursively.

    Recurses into ``dict`` values and ``list`` items.  String leaves are
    scanned by the DLP regex pipeline and sensitive spans replaced.
    Non-string, non-container values pass through unchanged.

    Placeholder types produced:

    * ``<PII:SSN>``        — US Social Security Number (NNN-NN-NNNN)
    * ``<PII:EMAIL>``      — email address
    * ``<PII:PHONE>``      — US phone number (10-digit, common separators)
    * ``<PII:CC>``         — credit/debit card number (13-16 digits, Luhn-ish)
    * ``<PII:SECRET>``     — AWS access key ID, Bearer token, API key value
    * ``<PII:IP_ADDRESS>`` — IPv4 address or common IPv6 forms

    **args_redacted vs args_hash:**  this output is the human-readable audit
    copy.  :func:`args_hash` must be computed over the ORIGINAL (unredacted)
    args — see module docstring for the full contract.

    Args:
        obj: Any Python value.  Strings are redacted in-place; dicts and
            lists are traversed recursively; other types pass through.

    Returns:
        A copy of *obj* with sensitive string content replaced.
    """
    if isinstance(obj, str):
        return _redact_str(obj)
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(item) for item in obj]
    return obj


def redact_safe(args: dict[str, Any]) -> dict[str, Any]:
    """Like :func:`redact` but falls back to redacting ALL string values on error.

    Used by the SDK in production when the DLP pipeline fails and
    ``fail_mode="closed"`` — guarantees no sensitive data leaks to audit logs
    even if a classifier raises unexpectedly.

    Args:
        args: Tool argument dict.

    Returns:
        Redacted dict; on any exception from :func:`redact`, all string values
        are replaced with ``"<REDACTED>"``.
    """
    try:
        result = redact(args)
        if isinstance(result, dict):
            return result
        # Should not happen when called with a dict argument.
        return {k: "<REDACTED>" if isinstance(v, str) else v for k, v in args.items()}
    except Exception:  # noqa: BLE001
        return {k: "<REDACTED>" if isinstance(v, str) else v for k, v in args.items()}
