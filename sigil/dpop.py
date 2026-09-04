"""DPoP (RFC 9449) proof generation for the MCP token-exchange helper.

A per-client ES256 (P-256) keypair. Its RFC 7638 JWK thumbprint is the cnf.jkt the
server binds; the same key signs the token-request proof and every per-request proof,
so all DPoP tokens from one client share a stable jkt (DPoP permits key reuse — the
binding is proof-of-possession, not per-token uniqueness). Keys are ephemeral per
process (fine for 5-minute tokens).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _int_b64url(n: int, size: int = 32) -> str:
    return _b64url(n.to_bytes(size, "big"))


def ath_for(access_token: str) -> str:
    """base64url(SHA-256(access_token)) — the DPoP 'ath' binding for resource requests.

    UTF-8 (a superset of the ASCII a JWT uses) so this is byte-identical to the TS SDK's athFor
    for every input, not only well-formed ASCII tokens.
    """
    return _b64url(hashlib.sha256(access_token.encode("utf-8")).digest())


class DPoPKey:
    """A P-256 keypair that produces DPoP proof JWTs."""

    def __init__(self) -> None:
        self._priv = ec.generate_private_key(ec.SECP256R1())
        nums = self._priv.public_key().public_numbers()
        self._jwk = {
            "crv": "P-256",
            "kty": "EC",
            "x": _int_b64url(nums.x),
            "y": _int_b64url(nums.y),
        }
        # RFC 7638 thumbprint: SHA-256 over the required members in lexicographic order.
        canon = json.dumps(
            {"crv": "P-256", "kty": "EC", "x": self._jwk["x"], "y": self._jwk["y"]},
            separators=(",", ":"),
            sort_keys=True,
        )
        self.thumbprint = _b64url(hashlib.sha256(canon.encode("ascii")).digest())

    def _sign(self, signing_input: bytes) -> bytes:
        der = self._priv.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")  # JWS ES256 = raw r||s

    def proof(
        self,
        htu: str,
        htm: str = "POST",
        *,
        ath: str | None = None,
        now: float | None = None,
    ) -> str:
        ts = int(now if now is not None else time.time())
        header = {"typ": "dpop+jwt", "alg": "ES256", "jwk": self._jwk}
        payload: dict[str, object] = {
            "htu": htu,
            "htm": htm,
            "iat": ts,
            "jti": _b64url(os.urandom(16)),
        }
        if ath is not None:
            payload["ath"] = ath
        # Insertion order (NOT sort_keys) so the encoded header/payload are byte-identical to the TS
        # SDK's JSON.stringify output. Key order is JWS-irrelevant; this is purely for cross-SDK
        # parity. (The RFC 7638 thumbprint above keeps its required lexicographic canonicalization.)
        h = _b64url(json.dumps(header, separators=(",", ":")).encode())
        p = _b64url(json.dumps(payload, separators=(",", ":")).encode())
        sig = _b64url(self._sign(f"{h}.{p}".encode()))
        return f"{h}.{p}.{sig}"
