"""Unit tests for sigil.verify — local biscuit token verification.

Token construction mirrors the Go wire format exactly:
  base64url_nopad(json_bytes) "." base64url_nopad(ed25519_sig_of_json_bytes)

We sign tokens with a freshly-generated nacl.signing.SigningKey so we control
both the signing side and the keyring, without needing a live drm-service.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest
from nacl.signing import SigningKey

from sigil.verify import (
    decode_token,
    has_tool_fact,
    is_expired,
    verify_local,
    verify_token,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _b64url_encode(data: bytes) -> str:
    """base64url-encode without padding (mirrors Go base64.RawURLEncoding)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_token(payload_dict: dict, signing_key: SigningKey) -> str:
    """Build a wire-format biscuit token from a payload dict + signing key."""
    payload_bytes = json.dumps(payload_dict, separators=(",", ":")).encode()
    sig_bytes = signing_key.sign(payload_bytes).signature
    return f"{_b64url_encode(payload_bytes)}.{_b64url_encode(sig_bytes)}"


def _future_check(hours: int = 1) -> str:
    """RFC3339 expiry check string for a time *hours* in the future."""
    ts = (datetime.now(tz=timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f'check if time($t), $t < "{ts}"'


def _past_check(hours: int = 1) -> str:
    """RFC3339 expiry check string for a time *hours* in the past."""
    ts = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f'check if time($t), $t < "{ts}"'


def _single_block_payload(
    kid: str = "k1",
    tenant: str = "tenant-uuid",
    agent: str = "agent-uuid",
    task_id: str = "task-uuid",
    tools: list[str] | None = None,
    checks: list[str] | None = None,
) -> dict:
    """Build a minimal valid BiscuitToken payload with one authority block."""
    tool_list = tools if tools is not None else ["zep.search"]
    facts = [
        f'tenant("{tenant}")',
        f'agent("{agent}")',
        f'task("{task_id}")',
        *[f'tool("{t}")' for t in tool_list],
    ]
    return {
        "v": 1,
        "blocks": [
            {
                "facts": facts,
                "checks": checks or [],
                "rid": "rev-001",
                "idx": 0,
            }
        ],
        "kid": kid,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sk() -> SigningKey:
    """Fresh ed25519 signing key for each test."""
    return SigningKey.generate()


@pytest.fixture
def keyring(sk: SigningKey) -> dict[str, bytes]:
    """Keyring with a single key 'k1' matching the test signing key."""
    return {"k1": bytes(sk.verify_key)}


# ──────────────────────────────────────────────────────────────────────────────
# verify_local — happy path
# ──────────────────────────────────────────────────────────────────────────────


class TestVerifyLocalHappyPath:
    def test_valid_token_ok_true(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(), sk)
        result = verify_local(token, keyring)
        assert result.ok is True

    def test_extracts_agent_id(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(agent="agent-abc"), sk)
        result = verify_local(token, keyring)
        assert result.agent_id == "agent-abc"

    def test_extracts_task_id(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(task_id="task-xyz"), sk)
        result = verify_local(token, keyring)
        assert result.task_id == "task-xyz"

    def test_extracts_tenant_id(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(tenant="tenant-t1"), sk)
        result = verify_local(token, keyring)
        assert result.tenant_id == "tenant-t1"

    def test_effective_tools_contains_granted_tool(
        self, sk: SigningKey, keyring: dict[str, bytes]
    ) -> None:
        token = _make_token(_single_block_payload(tools=["zep.search", "memory.store"]), sk)
        result = verify_local(token, keyring)
        assert "zep.search" in result.effective_tools
        assert "memory.store" in result.effective_tools

    def test_required_tool_in_scope(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(tools=["zep.search"]), sk)
        result = verify_local(token, keyring, required_tool="zep.search")
        assert result.ok is True

    def test_expected_tenant_matches(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(tenant="t1"), sk)
        result = verify_local(token, keyring, expected_tenant="t1")
        assert result.ok is True

    def test_not_yet_expired_token(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        payload = _single_block_payload(checks=[_future_check(hours=1)])
        token = _make_token(payload, sk)
        result = verify_local(token, keyring)
        assert result.ok is True

    def test_expires_at_populated(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        payload = _single_block_payload(checks=[_future_check(hours=1)])
        token = _make_token(payload, sk)
        result = verify_local(token, keyring)
        assert result.expires_at is not None


# ──────────────────────────────────────────────────────────────────────────────
# verify_local — tampered / invalid tokens
# ──────────────────────────────────────────────────────────────────────────────


class TestVerifyLocalInvalidToken:
    def test_tampered_payload_fails(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(), sk)
        payload_b64, sig_b64 = token.split(".")
        # Decode, add an evil tool, re-encode — sig is now invalid for the new bytes
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
        payload["blocks"][0]["facts"].append('tool("evil.execute")')
        tampered_bytes = json.dumps(payload, separators=(",", ":")).encode()
        bad_token = f"{_b64url_encode(tampered_bytes)}.{sig_b64}"
        result = verify_local(bad_token, keyring)
        assert result.ok is False
        assert result.reason == "token_invalid"

    def test_unknown_kid_fails_closed(self, sk: SigningKey) -> None:
        token = _make_token(_single_block_payload(kid="k1"), sk)
        result = verify_local(token, {"other_kid": bytes(sk.verify_key)})
        assert result.ok is False
        assert result.reason == "token_invalid"

    def test_empty_kid_succeeds_with_keyring_fallback(
        self, sk: SigningKey, keyring: dict[str, bytes]
    ) -> None:
        """L1: kid="" falls back to trying all keyring keys (Go server activeID fallback).

        A token with an empty kid field but a valid signature against a known
        keyring key is accepted, matching the Go server's activeID fallback.
        """
        payload = _single_block_payload()
        payload["kid"] = ""
        token = _make_token(payload, sk)
        result = verify_local(token, keyring)
        assert result.ok is True  # L1: empty kid uses keyring fallback

    def test_empty_kid_empty_keyring_fails_closed(self) -> None:
        """L1: kid="" with an empty keyring → token_invalid (no keys to try)."""
        sk = SigningKey.generate()
        payload = _single_block_payload()
        payload["kid"] = ""
        token = _make_token(payload, sk)
        result = verify_local(token, {})
        assert result.ok is False
        assert result.reason == "token_invalid"

    def test_malformed_token_no_dot(self, keyring: dict[str, bytes]) -> None:
        result = verify_local("nodothere", keyring)
        assert result.ok is False
        assert result.reason == "token_invalid"

    def test_malformed_base64_fails(self, keyring: dict[str, bytes]) -> None:
        result = verify_local("!!!.!!!", keyring)
        assert result.ok is False
        assert result.reason == "token_invalid"

    def test_wrong_key_fails(self, sk: SigningKey) -> None:
        token = _make_token(_single_block_payload(), sk)
        other_sk = SigningKey.generate()
        result = verify_local(token, {"k1": bytes(other_sk.verify_key)})
        assert result.ok is False
        assert result.reason == "token_invalid"

    def test_empty_blocks_fails(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        payload = {"v": 1, "blocks": [], "kid": "k1"}
        token = _make_token(payload, sk)
        result = verify_local(token, keyring)
        assert result.ok is False
        assert result.reason == "token_invalid"


# ──────────────────────────────────────────────────────────────────────────────
# verify_local — expiry
# ──────────────────────────────────────────────────────────────────────────────


class TestVerifyLocalExpiry:
    def test_expired_single_block(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        payload = _single_block_payload(checks=[_past_check(hours=1)])
        token = _make_token(payload, sk)
        result = verify_local(token, keyring)
        assert result.ok is False
        assert result.reason == "task_expired"

    def test_expires_at_returned_on_expiry(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        payload = _single_block_payload(checks=[_past_check(hours=1)])
        token = _make_token(payload, sk)
        result = verify_local(token, keyring)
        assert result.expires_at is not None

    def test_most_restrictive_expiry_wins(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        """Block 0 is not expired but block 1 is — most restrictive (block 1) wins."""
        payload = {
            "v": 1,
            "blocks": [
                {
                    "facts": [
                        'tenant("t1")',
                        'agent("a1")',
                        'task("t1")',
                        'tool("zep.search")',
                    ],
                    "checks": [_future_check(hours=2)],  # not expired
                    "rid": "r0",
                    "idx": 0,
                },
                {
                    "facts": [
                        'tenant("t1")',
                        'agent("a1")',
                        'task("t1")',
                        'tool("zep.search")',
                    ],
                    "checks": [_past_check(hours=1)],  # expired
                    "rid": "r1",
                    "idx": 1,
                },
            ],
            "kid": "k1",
        }
        token = _make_token(payload, sk)
        result = verify_local(token, keyring)
        assert result.ok is False
        assert result.reason == "task_expired"

    def test_unparseable_check_skipped(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        """Mirrors biscuit_service.go:528-530: unparseable checks are skipped, not denied."""
        payload = _single_block_payload(checks=['check if time($t), $t < "not-a-date"'])
        token = _make_token(payload, sk)
        result = verify_local(token, keyring)
        # Unparseable check is skipped → token is NOT expired
        assert result.ok is True

    def test_now_override_causes_expiry(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        """now= override allows testing expiry against a custom clock."""
        future_ts = (datetime.now(tz=timezone.utc) + timedelta(hours=24)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        payload = _single_block_payload(checks=[f'check if time($t), $t < "{future_ts}"'])
        token = _make_token(payload, sk)
        # Token is valid now but pretend it's 48 hours in the future
        far_future = datetime.now(tz=timezone.utc) + timedelta(hours=48)
        result = verify_local(token, keyring, now=far_future)
        assert result.ok is False
        assert result.reason == "task_expired"


# ──────────────────────────────────────────────────────────────────────────────
# verify_local — tool scope
# ──────────────────────────────────────────────────────────────────────────────


class TestVerifyLocalToolScope:
    def test_required_tool_not_in_scope(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(tools=["zep.search"]), sk)
        result = verify_local(token, keyring, required_tool="db.write")
        assert result.ok is False
        assert result.reason == "tool_not_in_scope"

    def test_tool_not_in_scope_returns_effective_tools(
        self, sk: SigningKey, keyring: dict[str, bytes]
    ) -> None:
        token = _make_token(_single_block_payload(tools=["zep.search"]), sk)
        result = verify_local(token, keyring, required_tool="db.write")
        assert "zep.search" in result.effective_tools

    def test_no_required_tool_arg_skips_scope_check(
        self, sk: SigningKey, keyring: dict[str, bytes]
    ) -> None:
        token = _make_token(_single_block_payload(tools=["zep.search"]), sk)
        result = verify_local(token, keyring)
        assert result.ok is True

    def test_attenuation_intersection_removes_tool(
        self, sk: SigningKey, keyring: dict[str, bytes]
    ) -> None:
        """H-3: tool in block 0 but removed in block 1 is NOT in the effective set."""
        payload = {
            "v": 1,
            "blocks": [
                {
                    "facts": [
                        'tenant("t1")',
                        'agent("a1")',
                        'task("tk1")',
                        'tool("zep.search")',
                        'tool("memory.store")',
                    ],
                    "checks": [],
                    "rid": "r0",
                    "idx": 0,
                },
                {
                    # Attenuation block only retains zep.search
                    "facts": [
                        'tenant("t1")',
                        'agent("a1")',
                        'task("tk1")',
                        'tool("zep.search")',
                    ],
                    "checks": [],
                    "rid": "r1",
                    "idx": 1,
                },
            ],
            "kid": "k1",
        }
        token = _make_token(payload, sk)

        # memory.store was removed by attenuation
        result_denied = verify_local(token, keyring, required_tool="memory.store")
        assert result_denied.ok is False
        assert result_denied.reason == "tool_not_in_scope"
        assert "memory.store" not in result_denied.effective_tools

        # zep.search is present in both blocks → still in effective set
        result_allowed = verify_local(token, keyring, required_tool="zep.search")
        assert result_allowed.ok is True
        assert "zep.search" in result_allowed.effective_tools

    def test_attenuation_intersection_empty_block_removes_all(
        self, sk: SigningKey, keyring: dict[str, bytes]
    ) -> None:
        """An attenuation block with no tool facts makes effective set empty."""
        payload = {
            "v": 1,
            "blocks": [
                {
                    "facts": [
                        'tenant("t1")',
                        'agent("a1")',
                        'task("tk1")',
                        'tool("zep.search")',
                    ],
                    "checks": [],
                    "rid": "r0",
                    "idx": 0,
                },
                {
                    # No tool facts — intersection with anything = empty
                    "facts": ['tenant("t1")', 'agent("a1")', 'task("tk1")'],
                    "checks": [],
                    "rid": "r1",
                    "idx": 1,
                },
            ],
            "kid": "k1",
        }
        token = _make_token(payload, sk)
        result = verify_local(token, keyring, required_tool="zep.search")
        assert result.ok is False
        assert result.reason == "tool_not_in_scope"


# ──────────────────────────────────────────────────────────────────────────────
# verify_local — tenant guard
# ──────────────────────────────────────────────────────────────────────────────


class TestVerifyLocalTenantGuard:
    def test_tenant_mismatch_denied(self, sk: SigningKey, keyring: dict[str, bytes]) -> None:
        token = _make_token(_single_block_payload(tenant="tenant-A"), sk)
        result = verify_local(token, keyring, expected_tenant="tenant-B")
        assert result.ok is False
        assert result.reason == "tenant_mismatch"

    def test_tenant_mismatch_returns_identity(
        self, sk: SigningKey, keyring: dict[str, bytes]
    ) -> None:
        token = _make_token(_single_block_payload(tenant="tenant-A", agent="agent-1"), sk)
        result = verify_local(token, keyring, expected_tenant="tenant-B")
        assert result.agent_id == "agent-1"
        assert result.tenant_id == "tenant-A"

    def test_none_expected_tenant_skips_check(
        self, sk: SigningKey, keyring: dict[str, bytes]
    ) -> None:
        token = _make_token(_single_block_payload(tenant="tenant-A"), sk)
        result = verify_local(token, keyring, expected_tenant=None)
        assert result.ok is True


# ──────────────────────────────────────────────────────────────────────────────
# Legacy helpers — backward compat
# ──────────────────────────────────────────────────────────────────────────────


class TestLegacyHelpers:
    def test_decode_token_returns_dict(self, sk: SigningKey) -> None:
        token = _make_token(_single_block_payload(), sk)
        d = decode_token(token)
        assert isinstance(d, dict)
        assert "blocks" in d

    def test_decode_token_rejects_malformed(self) -> None:
        with pytest.raises(ValueError):
            decode_token("nodot")

    def test_has_tool_fact_old_format(self) -> None:
        authority = {"facts": ['tool("zep.search")', 'agent("a1")']}
        assert has_tool_fact(authority, "zep.search") is True
        assert has_tool_fact(authority, "db.write") is False

    def test_is_expired_past(self) -> None:
        past = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
        assert is_expired({"expires_at": past}) is True

    def test_is_expired_future(self) -> None:
        future = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        assert is_expired({"expires_at": future}) is False

    def test_is_expired_missing_raises(self) -> None:
        with pytest.raises(ValueError):
            is_expired({})

    def test_verify_token_valid(self, sk: SigningKey) -> None:
        """verify_token (legacy single-key) works for old flat-format tokens."""
        old_payload = {
            "facts": ['tool("zep.search")'],
            "issued_at": "2026-01-01T00:00:00Z",
            "expires_at": (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        payload_bytes = json.dumps(old_payload, separators=(",", ":")).encode()
        sig = sk.sign(payload_bytes).signature
        token = f"{_b64url_encode(payload_bytes)}.{_b64url_encode(sig)}"
        result = verify_token(token, bytes(sk.verify_key))
        assert result["facts"] == ['tool("zep.search")']

    def test_verify_token_bad_sig_raises(self, sk: SigningKey) -> None:
        from nacl.exceptions import BadSignatureError as NaclBSE

        payload_bytes = json.dumps({"x": 1}, separators=(",", ":")).encode()
        bad_sig = b"\x00" * 64
        token = f"{_b64url_encode(payload_bytes)}.{_b64url_encode(bad_sig)}"
        with pytest.raises(NaclBSE):
            verify_token(token, bytes(sk.verify_key))
