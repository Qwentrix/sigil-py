"""SG-9 SP-1 (ENT-91) — agent handoff producer (sigil-py).

Covers client.record_handoff transport (201 + non-201) and the SigilTaskContext.handoff
convenience that fills parent task/agent from the active context (and the no-task guard).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sigil.client import SigilClient
from sigil.errors import SigilAPIError, SigilDeniedError

_AGENT = "agent-1"
_TENANT = "tenant-1"


def _client(tmp_path: Any) -> SigilClient:
    return SigilClient(
        base_url="http://sigil-test:8120",
        internal_token="tok",
        tenant_id=_TENANT,
        agent_id=_AGENT,
        service_account_id="sa-1",
        biscuit_keyring={},
        overflow_dir=str(tmp_path),
    )


def _resp(status: int, body: Any) -> MagicMock:
    r: MagicMock = MagicMock()
    r.status_code = status
    r.json.return_value = body
    return r


def test_record_handoff_posts_to_handoffs_endpoint(tmp_path: Any) -> None:
    client = _client(tmp_path)
    try:
        with patch.object(
            client._session, "post", return_value=_resp(201, {"handoff_id": "h-1"})
        ) as mp:
            out = client.record_handoff(
                parent_task_id="t",
                parent_agent_id="a",
                child_agent_id="b",
                child_task_id="c",
                scope_delta={"tools": ["zep.search"]},
            )
        assert out == {"handoff_id": "h-1"}
        assert mp.call_args.args[0].endswith("/internal/v1/sigil/handoffs")
        body = mp.call_args.kwargs["json"]
        assert body["parent_task_id"] == "t"
        assert body["child_agent_id"] == "b"
        assert body["scope_delta"] == {"tools": ["zep.search"]}
        assert "authorized_by_user_id" not in body  # omitted when None
    finally:
        client.close()


def test_record_handoff_non_201_raises(tmp_path: Any) -> None:
    client = _client(tmp_path)
    try:
        with (
            patch.object(client._session, "post", return_value=_resp(500, {})),
            pytest.raises(SigilAPIError),
        ):
            client.record_handoff(
                parent_task_id="t",
                parent_agent_id="a",
                child_agent_id="b",
                child_task_id="c",
                scope_delta={},
            )
    finally:
        client.close()


def test_task_handoff_fills_parent_from_context(tmp_path: Any) -> None:
    client = _client(tmp_path)
    try:
        task = client.task(["zep.search"])
        task._task_id = "task-123"  # simulate an active task without a real biscuit issue
        with patch.object(client, "record_handoff", return_value={"handoff_id": "h-1"}) as mp:
            out = task.handoff("child-agent", {"tools": ["zep.search"]})
        assert out["handoff_id"] == "h-1"
        assert out["child_task_id"]  # generated
        kw = mp.call_args.kwargs
        assert kw["parent_task_id"] == "task-123"
        assert kw["parent_agent_id"] == _AGENT
        assert kw["child_agent_id"] == "child-agent"
        assert kw["child_task_id"] == out["child_task_id"]
    finally:
        client.close()


def test_task_handoff_without_active_task_raises(tmp_path: Any) -> None:
    client = _client(tmp_path)
    try:
        task = client.task(["zep.search"])  # never entered → _task_id is None
        with pytest.raises(SigilDeniedError):
            task.handoff("child-agent", {})
    finally:
        client.close()
