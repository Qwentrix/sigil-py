"""Unit tests for sigil.redaction — DLP regex pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sigil.redaction import (
    CLASSIFIER_LABELS,
    args_hash,
    canonical_json,
    redact,
    redact_safe,
)

# ──────────────────────────────────────────────────────────────────────────────
# String redaction — PII types
# ──────────────────────────────────────────────────────────────────────────────


class TestSSN:
    def test_ssn_replaced(self) -> None:
        assert redact("my SSN is 123-45-6789") == "my SSN is <PII:SSN>"

    def test_ssn_mid_sentence(self) -> None:
        result = redact("SSN: 987-65-4321, please don't share")
        assert "<PII:SSN>" in result
        assert "987-65-4321" not in result

    def test_non_ssn_digits_unchanged(self) -> None:
        # 4-digit group is NOT an SSN
        assert redact("part number 1234-56-7") == "part number 1234-56-7"


class TestEmail:
    def test_email_replaced(self) -> None:
        result = redact("contact foo@example.com for info")
        assert "<PII:EMAIL>" in result
        assert "foo@example.com" not in result

    def test_email_with_plus_tag(self) -> None:
        result = redact("user+tag@domain.org")
        assert "<PII:EMAIL>" in result

    def test_email_uppercase_domain(self) -> None:
        result = redact("User@Example.COM")
        assert "<PII:EMAIL>" in result


class TestPhone:
    def test_dashed_phone(self) -> None:
        result = redact("call 555-867-5309")
        assert "<PII:PHONE>" in result
        assert "867-5309" not in result

    def test_dotted_phone(self) -> None:
        result = redact("call 555.867.5309 now")
        assert "<PII:PHONE>" in result

    def test_phone_with_parens(self) -> None:
        result = redact("dial (555) 867-5309")
        assert "<PII:PHONE>" in result

    def test_phone_with_country_code(self) -> None:
        result = redact("+1 555 867 5309")
        assert "<PII:PHONE>" in result


class TestCreditCard:
    def test_compact_16_digits(self) -> None:
        result = redact("card: 4111111111111111 expires")
        assert "<PII:CC>" in result
        assert "4111111111111111" not in result

    def test_grouped_with_spaces(self) -> None:
        result = redact("visa 4111 1111 1111 1111 ok")
        assert "<PII:CC>" in result

    def test_grouped_with_hyphens(self) -> None:
        result = redact("4111-1111-1111-1111")
        assert "<PII:CC>" in result

    def test_13_digit_compact(self) -> None:
        # 13-digit Visa (old format)
        result = redact("old card 4111111111111 here")
        assert "<PII:CC>" in result

    def test_ssn_applied_before_cc(self) -> None:
        """SSN pattern fires before CC — the dashes prevent a CC match."""
        result = redact("SSN: 123-45-6789")
        # Should be SSN, not CC
        assert "<PII:SSN>" in result
        assert "<PII:CC>" not in result


class TestSecrets:
    def test_aws_access_key(self) -> None:
        result = redact("key = AKIAIOSFODNN7EXAMPLE")
        assert "<PII:SECRET>" in result
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_aws_asia_prefix(self) -> None:
        result = redact("export KEY=ASIAIOSFODNN7EXAMPLE123")
        assert "<PII:SECRET>" in result

    def test_bearer_token(self) -> None:
        result = redact("Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9")
        assert "<PII:SECRET>" in result

    def test_bearer_token_case_insensitive(self) -> None:
        result = redact("authorization: bearer ABCDEFGHIJKLMNOPQRSTUVWX")
        assert "<PII:SECRET>" in result

    def test_api_key_assignment(self) -> None:
        result = redact("api_key=sk-proj-abcdefghijklmnopqrstuv")
        assert "<PII:SECRET>" in result

    def test_api_key_colon_assignment(self) -> None:
        result = redact("api-key: sk-test-abcdefghijklmnopqrst")
        assert "<PII:SECRET>" in result


# ──────────────────────────────────────────────────────────────────────────────
# Container recursion
# ──────────────────────────────────────────────────────────────────────────────


class TestContainerRecursion:
    def test_dict_string_values_redacted(self) -> None:
        data = {"ssn": "123-45-6789", "name": "John"}
        result = redact(data)
        assert isinstance(result, dict)
        assert "<PII:SSN>" in result["ssn"]
        # Non-PII value unchanged
        assert result["name"] == "John"

    def test_dict_non_string_values_unchanged(self) -> None:
        data = {"count": 42, "active": True, "score": 3.14}
        result = redact(data)
        assert result == {"count": 42, "active": True, "score": 3.14}

    def test_dict_nested_dict(self) -> None:
        data = {"contact": {"email": "foo@bar.com", "id": 1}}
        result = redact(data)
        assert "<PII:EMAIL>" in result["contact"]["email"]
        assert result["contact"]["id"] == 1

    def test_list_items_redacted(self) -> None:
        data = ["foo@bar.com", "hello world", 42]
        result = redact(data)
        assert isinstance(result, list)
        assert "<PII:EMAIL>" in result[0]
        assert result[1] == "hello world"
        assert result[2] == 42

    def test_mixed_nested_structure(self) -> None:
        data = {
            "users": [
                {"email": "a@b.com", "ssn": "111-22-3333"},
                {"email": "c@d.com", "count": 5},
            ],
            "version": 1,
        }
        result = redact(data)
        assert "<PII:EMAIL>" in result["users"][0]["email"]
        assert "<PII:SSN>" in result["users"][0]["ssn"]
        assert "<PII:EMAIL>" in result["users"][1]["email"]
        assert result["users"][1]["count"] == 5
        assert result["version"] == 1

    def test_non_container_non_string_passthrough(self) -> None:
        assert redact(42) == 42
        assert redact(3.14) == pytest.approx(3.14)
        assert redact(True) is True
        assert redact(None) is None

    def test_plain_string_redacted(self) -> None:
        assert redact("SSN 999-88-7777") == "SSN <PII:SSN>"

    def test_plain_string_no_pii_unchanged(self) -> None:
        assert redact("hello world") == "hello world"


# ──────────────────────────────────────────────────────────────────────────────
# args_hash — over-original guarantee
# ──────────────────────────────────────────────────────────────────────────────


class TestArgsHashOverOriginal:
    def test_hash_differs_between_original_and_redacted(self) -> None:
        original = {"query": "my SSN is 123-45-6789"}
        h_original = args_hash(original)
        redacted = redact(original)
        h_redacted = args_hash(redacted)
        # Redacted content is different → hashes must differ
        assert h_original != h_redacted

    def test_hash_stable_on_original(self) -> None:
        original = {"query": "my SSN is 123-45-6789", "limit": 10}
        h1 = args_hash(original)
        h2 = args_hash(original)
        assert h1 == h2

    def test_hash_not_changed_by_redact_call(self) -> None:
        original = {"q": "foo@bar.com and 555-123-4567"}
        h_before = args_hash(original)
        _redacted = redact(original)
        # original dict must not be mutated; hash still the same
        assert args_hash(original) == h_before

    def test_hash_is_sha256_hex(self) -> None:
        h = args_hash({"x": 1})
        # SHA-256 hex is 64 characters
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ──────────────────────────────────────────────────────────────────────────────
# canonical_json
# ──────────────────────────────────────────────────────────────────────────────


class TestCanonicalJson:
    def test_keys_sorted(self) -> None:
        s = canonical_json({"z": 1, "a": 2})
        assert s == '{"a":2,"z":1}'

    def test_no_extra_whitespace(self) -> None:
        s = canonical_json({"x": 1})
        assert " " not in s

    def test_nested(self) -> None:
        s = canonical_json({"b": {"d": 4, "c": 3}, "a": 1})
        assert s == '{"a":1,"b":{"c":3,"d":4}}'


# ──────────────────────────────────────────────────────────────────────────────
# redact_safe
# ──────────────────────────────────────────────────────────────────────────────


class TestRedactSafe:
    def test_delegates_to_redact_normally(self) -> None:
        data = {"ssn": "123-45-6789"}
        result = redact_safe(data)
        assert "<PII:SSN>" in result["ssn"]

    def test_falls_back_on_exception(self) -> None:
        """If redact() raises, redact_safe replaces all string values."""
        with patch("sigil.redaction.redact", side_effect=RuntimeError("DLP down")):
            result = redact_safe({"ssn": "123-45-6789", "count": 42})
        assert result["ssn"] == "<REDACTED>"
        assert result["count"] == 42  # non-string unchanged

    def test_non_string_values_preserved_on_fallback(self) -> None:
        with patch("sigil.redaction.redact", side_effect=RuntimeError("oops")):
            result = redact_safe({"x": 1, "y": [1, 2], "z": None})
        # Non-strings pass through in the fallback
        assert result["x"] == 1
        assert result["y"] == [1, 2]
        assert result["z"] is None


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFIER_LABELS registry
# ──────────────────────────────────────────────────────────────────────────────


class TestIPAddress:
    def test_ipv4_replaced(self) -> None:
        """M4: IPv4 addresses are redacted."""
        result = redact("server at 192.168.1.100 responded")
        assert "<PII:IP_ADDRESS>" in result
        assert "192.168.1.100" not in result

    def test_ipv4_boundary(self) -> None:
        result = redact("connect to 10.0.0.1 now")
        assert "<PII:IP_ADDRESS>" in result

    def test_ipv4_localhost_redacted(self) -> None:
        result = redact("host 127.0.0.1 refused")
        assert "<PII:IP_ADDRESS>" in result

    def test_ipv6_full_replaced(self) -> None:
        """M4: Full-form IPv6 addresses are redacted."""
        result = redact("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 here")
        assert "<PII:IP_ADDRESS>" in result

    def test_ipv6_loopback_compressed_replaced(self) -> None:
        """M4: Compressed IPv6 (::1 loopback) is redacted."""
        result = redact("loopback ::1 detected")
        assert "<PII:IP_ADDRESS>" in result


class TestClassifierLabels:
    def test_required_labels_present(self) -> None:
        for key in ("SSN", "EMAIL", "PHONE", "CREDIT_CARD", "SECRET", "IP_ADDRESS"):
            assert key in CLASSIFIER_LABELS

    def test_person_name_removed(self) -> None:
        """M4: PERSON_NAME (needs NER, no regex) is removed from public dict."""
        assert "PERSON_NAME" not in CLASSIFIER_LABELS

    def test_ip_address_label_value(self) -> None:
        assert CLASSIFIER_LABELS["IP_ADDRESS"] == "<PII:IP_ADDRESS>"

    def test_ssn_label_value(self) -> None:
        assert CLASSIFIER_LABELS["SSN"] == "<PII:SSN>"

    def test_cc_label_value(self) -> None:
        assert CLASSIFIER_LABELS["CREDIT_CARD"] == "<PII:CC>"

    def test_secret_label_value(self) -> None:
        assert CLASSIFIER_LABELS["SECRET"] == "<PII:SECRET>"
