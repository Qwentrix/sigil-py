"""Contract tests for the preflight endpoint.

These tests run against a sigil-core fixture (real or mock server).  They are
skipped in plain unit-test runs when the fixture is unavailable.

Fixture expectations:
- sigil-core listening on the URL in env var ``SIGIL_TEST_BASE_URL`` (default:
  ``http://localhost:8120``).
- A pre-registered agent with id ``SIGIL_TEST_AGENT_ID`` and an open task
  ``SIGIL_TEST_TASK_ID`` whose scope includes ``"zep.search"`` and excludes
  ``"db.write"``.
- ``SIGIL_TEST_API_KEY`` set to a valid service account credential.

Set ``SIGIL_CONTRACT_TESTS=1`` to enable this test suite.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Pytest markers / skip guard
# ---------------------------------------------------------------------------

CONTRACT_TESTS_ENABLED = os.environ.get("SIGIL_CONTRACT_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not CONTRACT_TESTS_ENABLED,
    reason="Contract tests disabled. Set SIGIL_CONTRACT_TESTS=1 to enable.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_url() -> str:
    return os.environ.get("SIGIL_TEST_BASE_URL", "http://localhost:8120")


@pytest.fixture()
def agent_id() -> str:
    value = os.environ.get("SIGIL_TEST_AGENT_ID", "")
    if not value:
        pytest.skip("SIGIL_TEST_AGENT_ID not set")
    return value


@pytest.fixture()
def task_id() -> str:
    value = os.environ.get("SIGIL_TEST_TASK_ID", "")
    if not value:
        pytest.skip("SIGIL_TEST_TASK_ID not set")
    return value


@pytest.fixture()
def api_key() -> str:
    value = os.environ.get("SIGIL_TEST_API_KEY", "")
    if not value:
        pytest.skip("SIGIL_TEST_API_KEY not set")
    return value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreflightAllowed:
    """Preflight returns ``allow`` for a tool that is in scope."""

    def test_allowed_tool_returns_allow_verdict(
        self, base_url: str, agent_id: str, task_id: str, api_key: str
    ) -> None:
        """``zep.search`` is in the test task scope — expect verdict=allow."""
        pytest.skip("Not yet implemented — pending live sigil-core fixture")


class TestPreflightDenied:
    """Preflight returns ``deny`` for a tool that is NOT in scope."""

    def test_out_of_scope_tool_returns_deny_verdict(
        self, base_url: str, agent_id: str, task_id: str, api_key: str
    ) -> None:
        """``db.write`` is excluded from the test task scope — expect verdict=deny."""
        pytest.skip("Not yet implemented — pending live sigil-core fixture")

    def test_deny_verdict_includes_denied_reason(
        self, base_url: str, agent_id: str, task_id: str, api_key: str
    ) -> None:
        """Denial response must include a non-empty ``denied_reason`` string."""
        pytest.skip("Not yet implemented — pending live sigil-core fixture")


class TestPreflightRevokedAgent:
    """Preflight returns ``deny`` with ``agent_revoked`` after kill switch."""

    def test_revoked_agent_returns_agent_revoked_reason(
        self, base_url: str, agent_id: str, task_id: str, api_key: str
    ) -> None:
        """After issuing a kill switch, any subsequent preflight must deny."""
        pytest.skip("Not yet implemented — requires kill-switch endpoint access")


class TestPreflightResponseShape:
    """All preflight responses must conform to the protocol schema."""

    def test_response_has_required_fields(
        self, base_url: str, agent_id: str, task_id: str, api_key: str
    ) -> None:
        """Response dict must contain at least ``verdict`` key."""
        pytest.skip("Not yet implemented — pending live sigil-core fixture")

    def test_allow_verdict_has_null_denied_reason(
        self, base_url: str, agent_id: str, task_id: str, api_key: str
    ) -> None:
        """``denied_reason`` must be null when verdict is ``allow``."""
        pytest.skip("Not yet implemented — pending live sigil-core fixture")
