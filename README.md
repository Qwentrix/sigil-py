# sigil-py

**Micelium Sigil** — Python SDK for embedding governance into AI agents.

Every AI agent your code deploys can be registered, scoped, audited, and
kill-switched in under-1 ms overhead via the Sigil control plane.

- Full docs: [sigil.micelium.com/docs](https://sigil.micelium.com/docs)
- Wire protocol spec: [docs/protocol.md](docs/protocol.md)
- Node.js SDK: [github.com/Qwentrix/sigil-sdk-node](https://github.com/Qwentrix/sigil-sdk-node)

---

## Requirements

- Python 3.10+
- A running [sigil-core](https://github.com/Qwentrix/sigil-core) instance
- A registered agent ID + service-account API key (obtained from the Micelium
  One dashboard or via `POST /api/v1/sigil/agents`)

---

## Installation

```bash
pip install sigil-py
```

---

## Quickstart

```python
import os
from sigil import SigilClient, instrumented_tool, SigilDeniedError

# Initialise once (thread-safe, shares a connection pool)
client = SigilClient(
    agent_id=os.environ["SIGIL_AGENT_ID"],
    api_key=os.environ["SIGIL_API_KEY"],
    base_url=os.environ["SIGIL_BASE_URL"],   # e.g. "http://sigil-core:8120"
    fail_mode="closed",                      # default — deny if unreachable
)

# Decorate any tool function
@instrumented_tool(name="zep.search", risk_tier="low")
def search_memory(query: str) -> list[dict]:
    # ... your tool implementation ...
    return []

# Open a task scope (context manager)
with client.task("summarize-document", scope={"tools": ["zep.search"], "ttl_seconds": 600}):
    try:
        results = search_memory("Q4 financial results")
    except SigilDeniedError as exc:
        print(f"Denied: {exc.denied_reason} on tool {exc.tool_name}")
```

### Async variant

```python
from sigil import SigilClient

async def run() -> None:
    async with client.task("summarize-document", scope={...}) as ctx:
        results = await ctx.run(my_async_tool, "query")
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SIGIL_AGENT_ID` | Yes | UUID of the registered agent |
| `SIGIL_API_KEY` | Yes | Service-account credential |
| `SIGIL_BASE_URL` | Yes | sigil-core base URL |
| `SIGIL_FAIL_MODE` | No | `"closed"` (default) or `"open"` (dev only) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributors must sign the
Qwentrix CLA before their first PR is merged.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Copyright (c) 2026 Qwentrix Inc.
