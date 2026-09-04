import base64
import json

from sigil.dpop import DPoPKey, ath_for


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _parts(jwt: str):
    h, p, sig = jwt.split(".")
    return json.loads(_b64url_decode(h)), json.loads(_b64url_decode(p)), _b64url_decode(sig)


def test_proof_has_dpop_header_and_claims():
    key = DPoPKey()
    proof = key.proof("https://sigil.example/oauth/token", "POST", now=1000)
    header, payload, sig = _parts(proof)
    assert header["typ"] == "dpop+jwt"
    assert header["alg"] == "ES256"
    assert header["jwk"]["crv"] == "P-256" and header["jwk"]["kty"] == "EC"
    assert payload["htu"] == "https://sigil.example/oauth/token"
    assert payload["htm"] == "POST"
    assert payload["iat"] == 1000
    assert payload["jti"]  # present, non-empty
    assert "ath" not in payload
    assert len(sig) == 64  # ES256 raw r||s


def test_proof_for_resource_includes_ath():
    key = DPoPKey()
    proof = key.proof("https://gw.example/mcp", "GET", ath=ath_for("the-access-token"), now=1)
    _, payload, _ = _parts(proof)
    assert payload["htm"] == "GET"
    assert payload["ath"] == ath_for("the-access-token")


def test_signature_verifies_against_embedded_jwk():
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    key = DPoPKey()
    proof = key.proof("https://sigil.example/oauth/token", now=5)
    header, _, sig = _parts(proof)
    x = int.from_bytes(_b64url_decode(header["jwk"]["x"]), "big")
    y = int.from_bytes(_b64url_decode(header["jwk"]["y"]), "big")
    pub = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()
    signing_input = ".".join(proof.split(".")[:2]).encode()
    der = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
    pub.verify(der, signing_input, ec.ECDSA(hashes.SHA256()))  # raises if invalid


def test_thumbprint_is_stable_and_urlsafe():
    key = DPoPKey()
    assert key.thumbprint == key.thumbprint
    assert "=" not in key.thumbprint and "+" not in key.thumbprint and "/" not in key.thumbprint
