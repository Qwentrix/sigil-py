# Sigil Wire Protocol — v1 Specification

**Status:** Draft — SG-2 (2026-11-16)
**Canonical location:** This document is the shared protocol spec for `sigil-py` and `sigil-sdk-node`. Both SDKs implement against this spec. Changes require a PR to this file in `sigil-py`; the Node SDK mirrors by cross-referencing.

---

## 1. Overview

The Sigil wire protocol defines the HTTP exchange between SDK clients and `sigil-core`. All calls are over HTTPS in production (plaintext HTTP acceptable on localhost/loopback only). All request/response bodies are `application/json`.

Authentication on internal routes uses four headers (see §3 and §4 for which apply where):

| Header | Description |
|---|---|
| `X-Internal-Service-Token` | Shared secret from `SIGIL_SDK_TOKEN` env var (internalauth credential) |
| `X-Internal-Service-Account` | Service account name string (e.g. `"sigil-agent-sa"`) |
| `X-Tenant-ID` | Tenant UUID |
| `X-Sigil-Agent-ID` | Agent UUID (required by the rate-limited toolgate group) |

---

## 2. Token Format

Sigil uses a **homegrown JSON + ed25519 capability token** format (not an external Biscuit library). This matches the existing `drm-service` token format in `services/drm-service/pkg/biscuit/`.

### 2.1 Structure

A token is two base64url-encoded components separated by `.`:

```
<payload_b64url>.<signature_b64url>
```

Where `<payload_b64url>` is a base64url-encoded JSON object:

```json
{
  "v": 1,
  "kid": "<key-id>",
  "blocks": [
    {
      "facts": [
        "tool(\"zep.search\")",
        "tool(\"skovo.fetch\")",
        "tenant(\"<tenant_uuid>\")",
        "agent(\"<agent_uuid>\")",
        "task(\"<task_id_uuid>\")"
      ],
      "checks": ["check if time($t), $t < \"<ISO-8601-expiry>\""],
      "rid": "<revocation_id>",
      "idx": 0
    }
  ]
}
```

Facts use a `predicate("value")` string encoding. Each allowed tool produces one `tool(...)` fact with a `namespace.name` string. The `tenant(...)`, `agent(...)`, and `task(...)` facts bind the token to a specific execution context.

### 2.2 Signature

The payload bytes are signed with **ed25519** using the drm-service's root signing key identified by `kid`. The public key is distributed to SDKs at credential bootstrap time (service account provisioning response) and rotated quarterly. SDKs cache the public key in memory for the token lifetime.

Verification uses `pynacl` (Python) or `tweetnacl` (TypeScript/Node.js).

### 2.3 Verification Steps (SDK local check)

1. Split token on `.` into `[payload_b64, sig_b64]`.
2. base64url-decode both parts.
3. Read `kid` from the decoded payload JSON.
4. If `kid` is non-empty, look up the key in the SDK keyring by `kid`. If not found, reject.
   If `kid` is empty (Go server activeID fallback), try all keys in the keyring; accept if any verifies.
5. `nacl.signing.VerifyKey(key_bytes).verify(payload_bytes, sig_bytes)` — raises `BadSignatureError` if invalid.
6. Parse payload JSON; iterate `blocks[0].facts`.
7. Check the time `check` expression. If expired, treat as deny.
8. For a tool call check: confirm `"tool(\"<namespace>.<name>\")"` is present in `facts`.
9. Check `blocks[0].rid` against the local revocation cache. If found, deny immediately.

Local verification is used for all `risk_tier=low` tool calls not in the freshness probe cycle (every 10 calls). `risk_tier >= high` always triggers a preflight HTTP call to sigil-core regardless of local verify result.

---

## 3. Preflight Request/Response

**Endpoint:** `POST /internal/v1/sigil/toolgate/preflight`

**Required headers:** `X-Internal-Service-Token`, `X-Internal-Service-Account`, `X-Tenant-ID`, `X-Sigil-Agent-ID`

### 3.1 Request

```json
{
  "agent_id": "<uuid>",
  "task_id": "<uuid>",
  "tool_namespace": "<namespace>",
  "tool_name": "<name>",
  "args_hash": "<sha256hex of canonical JSON of unredacted args>",
  "args_redacted": {
    "<key>": "<value or PII-redacted placeholder>"
  }
}
```

`tool_namespace` and `tool_name` are sent as **separate fields** (not combined). The combined FQN `<namespace>.<name>` is used only within SDK internals.

`args_hash` is SHA-256 of the canonical JSON (keys sorted, no extra whitespace) of the **original unredacted args**. The hash proves integrity without logging sensitive data.

`args_redacted` contains the same keys as the original args, with string values replaced by DLP classifier labels when a classifier fires (e.g., `"<PII:SSN>"`, `"<PII:EMAIL>"`, `"<PII:IP_ADDRESS>"`). Non-sensitive values pass through unchanged.

### 3.2 Response

```json
{
  "verdict": "allow" | "deny" | "approve",
  "denied_reason": "<string or null>",
  "approval_id": "<uuid or null>",
  "latency_budget_ms": "<int or null>"
}
```

| Field | When present |
|---|---|
| `verdict` | Always |
| `denied_reason` | When `verdict == "deny"`. Values: `tool_not_in_scope`, `task_expired`, `agent_revoked`, `token_invalid`, `approval_service_unavailable`, `policy_deny` |
| `approval_id` | When `verdict == "approve"` (v2 only; v1 sigil-core always returns `allow` or `deny`) |
| `latency_budget_ms` | When `verdict == "approve"`: milliseconds SDK should wait before timing out |

**SDK verdict handling:**

- `"allow"` → proceed with execution.
- `"deny"` → raise `SigilDeniedError` with `denied_reason` from response.
- `"approve"` → approval gates are v2 only; v1 SDK treats `approve` as `deny` with reason `approval_service_unavailable`.
- `null`, `""`, or any other value → SDK treats as `"deny"` (fail-closed).

P99 target: **20 ms** (sigil-core performs local token verify + one Redis read for revocation check).

---

## 4. Audit Event Envelope

### 4.1 SDK Batch Log

**Endpoint:** `POST /internal/v1/sigil/toolgate/log-batch`

**Required headers:** `X-Internal-Service-Token`, `X-Internal-Service-Account`, `X-Tenant-ID`, `X-Sigil-Agent-ID`

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
      "latency_ms": "<int>",
      "outcome": "allowed" | "denied" | "error",
      "denied_reason": "<string or null>",
      "risk_tier": "low" | "med" | "high" | "critical",
      "fail_open": "<bool — present and true only when fail_mode=open overrode a sigil-core unreachable error>"
    }
  ]
}
```

> **Not yet implemented by SDK v0.1:** `result_hash` and `result_sampled` fields are reserved for a future release. The SDK does not populate them; sigil-core ignores absent fields.

Max **100 events** per batch. SDK flushes every 500 ms or 50 events, whichever comes first.

**Response:**

```json
{
  "accepted": 12
}
```

`accepted` is the integer count of events accepted by sigil-core. sigil-core XADDs to `sigil:writes` Redis Stream and ACKs immediately; a `sigil-writer` goroutine pool drains to PostgreSQL via COPY. P99 ACK target: **50 ms**.

### 4.2 Security Events Stream Payload

sigil-core also XADDs to `security-events:agent` Redis Stream after each logged batch. This is one-way fan-out to the compliance-reporting and behavioral-risk-engine consumers:

```json
{
  "event_type": "agent_tool_invoked" | "agent_task_opened" | "agent_task_closed" | "agent_tool_denied" | "agent_revoked" | "agent_handoff_recorded",
  "tenant_id": "<uuid>",
  "agent_id": "<uuid>",
  "task_id": "<uuid>",
  "tool_name": "<namespace>.<name>",
  "outcome": "allowed" | "denied" | "error",
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

The in-memory revocation cache maps `revocation_id -> revoked_at`. Cache is warmed at SDK startup (HTTP GET to sigil-core to pull current revocations for the agent). On cache hit during local verify, the SDK immediately raises `SigilDeniedError(denied_reason="agent_revoked")` without calling sigil-core. Cache is bounded at 10 000 entries (FIFO eviction).

**Tenant guard:** The SDK applies revocation events only when `tenant_id` in the event matches the client's own tenant, or when `tenant_id` is absent (backward compatibility with pre-tenant-field messages). Cross-tenant revocation events are silently ignored.

### 5.3 Fail-Closed on Revocation

If the Redis subscription is disconnected, the SDK falls back to checking revocation via the preflight HTTP call (forced for every tool call, not just `risk_tier >= high`). The freshness probe interval drops to 1 call (every call triggers a preflight) until the subscription is re-established.

---

## 6. Versioning

This spec is versioned at the URL path level (`/internal/v1/`). Breaking changes require a new version prefix. The SDK negotiates version at startup via `GET /internal/v1/sigil/capabilities` which returns the set of supported API versions and feature flags (e.g., `approval_gate: false` in v1, `approval_gate: true` in v2).

---

*This document is the canonical wire-protocol spec. The Node SDK (`@qwentrix/sigil`, `sigil-sdk-node`) implements against this same spec and cross-references this file in its own `docs/protocol.md`.*
