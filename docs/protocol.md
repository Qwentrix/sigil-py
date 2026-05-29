# Sigil Wire Protocol — v1 Specification

**Status:** Draft — SG-2 (2026-11-16)
**Canonical location:** This document is the shared protocol spec for `sigil-py` and `sigil-sdk-node`. Both SDKs implement against this spec. Changes require a PR to this file in `sigil-py`; the Node SDK mirrors by cross-referencing.

---

## 1. Overview

The Sigil wire protocol defines the HTTP exchange between SDK clients and `sigil-core`. All calls are over HTTPS in production (plaintext HTTP acceptable on localhost/loopback only). All request/response bodies are `application/json`.

Authentication on internal routes uses `X-Internal-Secret` (shared secret from `INTERNAL_API_SECRET` env var) plus `X-Sigil-Agent-Token` (Biscuit task token) for SDK-originated calls.

---

## 2. Token Format

Sigil uses a **homegrown JSON + ed25519 capability token** format (not an external Biscuit library). This matches the existing `drm-service` token format in `services/drm-service/pkg/biscuit/`.

### 2.1 Structure

A token is a base64url-encoded concatenation of two components:

```
<authority_block_b64url>.<signature_b64url>
```

Where `<authority_block_b64url>` is a base64url-encoded JSON object:

```json
{
  "version": 1,
  "identity_type": "agent",
  "agent_id": "<uuid>",
  "task_id": "<uuid>",
  "tenant_id": "<uuid>",
  "facts": [
    "tool(\"zep.search\")",
    "tool(\"skovo.fetch\")",
    "task(\"<task_id_uuid>\")",
    "agent(\"<agent_id_uuid>\")"
  ],
  "issued_at": "<ISO-8601>",
  "expires_at": "<ISO-8601>",
  "revocation_id": "<uuid>"
}
```

Facts use a `predicate("value")` string encoding. Each allowed tool produces one `tool(...)` fact with a `namespace.name` string.

### 2.2 Signature

The authority block bytes are signed with **ed25519** using the drm-service's root signing key. The public key is distributed to SDKs at credential bootstrap time (service account provisioning response) and rotated quarterly. SDKs cache the public key in memory for the token lifetime.

Verification uses `pynacl` (Python) or `tweetnacl` (TypeScript/Node.js) — both implement the same ed25519 signing primitive with compatible wire representations.

### 2.3 Verification Steps (SDK local check)

1. Split token on `.` into `[authority_b64, sig_b64]`.
2. base64url-decode both parts.
3. `nacl.signing.VerifyKey(public_key_bytes).verify(authority_bytes, sig_bytes)` — raises if invalid.
4. Parse authority JSON.
5. Check `expires_at > now`. If expired, treat as deny and fire a preflight to sigil-core.
6. For a tool call check: confirm `"tool(\"<namespace>.<name>\")"` is present in `facts`.
7. Check `revocation_id` against local revocation cache (populated by Redis pub/sub on `drm:revocation-events`). If found in cache, deny immediately.

Local verification is used for all `risk_tier=low` tool calls not in the freshness probe cycle (every 10 calls). `risk_tier >= high` always triggers a preflight HTTP call to sigil-core regardless of local verify result.

---

## 3. Preflight Request/Response

**Endpoint:** `POST /internal/v1/sigil/toolgate/preflight`

**Auth headers:** `X-Internal-Secret: <INTERNAL_API_SECRET>`, `X-Sigil-Agent-Token: <biscuit_token>`

### 3.1 Request

```json
{
  "agent_id": "<uuid>",
  "task_id": "<uuid>",
  "tool_name": "<namespace>.<name>",
  "args_hash": "<sha256hex of canonical JSON of unredacted args>",
  "args_redacted": {
    "<key>": "<value or PII-redacted placeholder>"
  }
}
```

`args_hash` is SHA-256 of the canonical JSON (keys sorted, no extra whitespace) of the **original unredacted args**. The hash proves integrity without logging sensitive data.

`args_redacted` contains the same keys as the original args, with values replaced by DLP classifier labels (e.g., `"<PII:PERSON_NAME>"`, `"<PHI:SSN>"`) when a classifier fires. Non-sensitive values pass through unchanged.

### 3.2 Response

```json
{
  "verdict": "allow" | "deny" | "pending_approval",
  "denied_reason": "<string or null>",
  "approval_id": "<uuid or null>",
  "latency_budget_ms": "<int or null>"
}
```

| Field | When present |
|---|---|
| `verdict` | Always |
| `denied_reason` | When `verdict == "deny"`. Values: `tool_not_in_scope`, `task_expired`, `agent_revoked`, `token_invalid`, `approval_service_unavailable`, `policy_deny` |
| `approval_id` | When `verdict == "pending_approval"` (v2 only; v1 always returns `allow` or `deny`) |
| `latency_budget_ms` | When `verdict == "pending_approval"`: milliseconds SDK should wait before timing out |

P99 target: **20 ms** (sigil-core performs local token verify + one Redis read for revocation check).

---

## 4. Audit Event Envelope

### 4.1 SDK Batch Log

**Endpoint:** `POST /internal/v1/sigil/toolgate/log-batch`

**Auth headers:** same as preflight.

**Request:**

```json
{
  "events": [
    {
      "agent_id": "<uuid>",
      "task_id": "<uuid>",
      "tool_name": "<namespace>.<name>",
      "tool_namespace": "<namespace>",
      "args_hash": "<sha256hex>",
      "args_redacted": { "<key>": "<value>" },
      "result_hash": "<sha256hex of canonical JSON of tool return value>",
      "result_sampled": { "<key>": "<value>" },
      "latency_ms": "<int>",
      "outcome": "allowed" | "denied",
      "denied_reason": "<string or null>",
      "risk_tier": "low" | "med" | "high" | "critical"
    }
  ]
}
```

Max **100 events** per batch. SDK flushes every 500 ms or 50 events, whichever comes first.

**Response:**

```json
{
  "invocation_ids": ["<uuid>", "..."]
}
```

Synthetic UUIDv7 ids assigned server-side. sigil-core XADDs to `sigil:writes` Redis Stream and ACKs immediately; a `sigil-writer` goroutine pool drains to PostgreSQL via COPY. P99 ACK target: **50 ms**.

### 4.2 Security Events Stream Payload

sigil-core also XADDs to `security-events:agent` Redis Stream after each logged batch. This is one-way fan-out to the compliance-reporting and behavioral-risk-engine consumers:

```json
{
  "event_type": "agent_tool_invoked" | "agent_task_opened" | "agent_task_closed" | "agent_tool_denied" | "agent_revoked" | "agent_handoff_recorded",
  "tenant_id": "<uuid>",
  "agent_id": "<uuid>",
  "task_id": "<uuid>",
  "tool_name": "<namespace>.<name>",
  "outcome": "allowed" | "denied",
  "risk_tier": "low" | "med" | "high" | "critical",
  "latency_ms": "<int>",
  "occurred_at": "<ISO-8601>",
  "correlation_id": "<uuid>"
}
```

---

## 5. Revocation

### 5.1 Kill Switch Flow

1. User POSTs `POST /api/v1/sigil/agents/:id/kill` to the API gateway (JWT auth, `sigil_admin` role).
2. sigil-core writes a `sigil_revocation_events` record and calls drm-service `POST /internal/v1/grants/biscuit/revoke` with the token's `revocation_id`.
3. drm-service publishes the revocation event to the Redis pub/sub channel `drm:revocation-events`.
4. All SDK instances subscribed to `drm:revocation-events` receive the event and update their local revocation cache within **1 second** (matches Sprint 35 implementation).

### 5.2 SDK Revocation Cache

The in-memory revocation cache maps `revocation_id -> revoked_at`. Cache is warmed at SDK startup (HTTP GET to sigil-core to pull current revocations for the agent). On cache hit during local verify, the SDK immediately raises `SigilDeniedError(denied_reason="agent_revoked")` without calling sigil-core.

### 5.3 Fail-Closed on Revocation

If the Redis subscription is disconnected, the SDK falls back to checking revocation via the preflight HTTP call (forced for every tool call, not just `risk_tier >= high`). The freshness probe interval drops to 1 call (every call triggers a preflight) until the subscription is re-established.

---

## 6. Versioning

This spec is versioned at the URL path level (`/internal/v1/`). Breaking changes require a new version prefix. The SDK negotiates version at startup via `GET /internal/v1/sigil/capabilities` which returns the set of supported API versions and feature flags (e.g., `approval_gate: false` in v1, `approval_gate: true` in v2).

---

*This document is the canonical wire-protocol spec. The Node SDK (`@qwentrix/sigil`, `sigil-sdk-node`) implements against this same spec and cross-references this file in its own `docs/protocol.md`.*
