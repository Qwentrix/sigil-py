"""Disk overflow writer and replayer for audit events.

Events are written here when sigil-core is unreachable during a flush, when
the ring buffer is full, or on a fail-closed denial (for later replay).

Each overflow file is newline-delimited JSON (NDJSON), one event per line,
named ``{agent_id}_{YYYY-MM-DD}.ndjson`` under ``SIGIL_OVERFLOW_DIR``
(default ``~/.sigil/overflow/``).

On a subsequent *successful* flush the writer drains and replays the files
in sorted (chronological) order, chunking to ≤100 events per log_batch call.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sigil.client import SigilClient

_log = logging.getLogger("sigil")

_DEFAULT_OVERFLOW_DIR: str = os.path.join(os.path.expanduser("~"), ".sigil", "overflow")

# M2/L4: security and disk-usage bounds.
# Directory created 0o700 (owner-only); files created 0o600 (owner read/write).
# Per-file cap: once the daily file reaches this size, new events are dropped
# (with a warning) rather than growing without bound.
_MAX_OVERFLOW_FILE_BYTES: int = 50 * 1024 * 1024  # 50 MiB


class _OverflowWriter:
    """Thread-safe (append-only) NDJSON writer with best-effort replay."""

    def __init__(self, overflow_dir: str | None, agent_id: str) -> None:
        self._dir: str = (
            overflow_dir or os.environ.get("SIGIL_OVERFLOW_DIR") or _DEFAULT_OVERFLOW_DIR
        )
        self._agent_id: str = agent_id or "unknown"
        self._writable: bool = self._init_dir()
        # F5: count dropped events so host applications can observe overflow-full
        # conditions and alert.  Exposed via the drop_count property.
        self._drop_count: int = 0

    def _init_dir(self) -> bool:
        try:
            # M2: owner-only directory (0o700) so overflow files are not
            # world-readable.  exist_ok=True preserves mode of existing dirs.
            os.makedirs(self._dir, mode=0o700, exist_ok=True)
            return True
        except OSError:
            _log.warning(
                "sigil: overflow dir %r is not writable — disk overflow disabled; "
                "events may be lost when sigil-core is unreachable",
                self._dir,
            )
            return False

    @property
    def directory(self) -> str:
        """Configured overflow directory (may or may not be writable)."""
        return self._dir

    @property
    def drop_count(self) -> int:
        """Number of events dropped because the overflow file hit the size cap.

        Expose this so host applications can alert when sigil-core has been
        unreachable long enough to exhaust the overflow budget.
        """
        return self._drop_count

    def _current_path(self) -> str:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self._dir, f"{self._agent_id}_{today}.ndjson")

    def write(self, event: dict[str, Any]) -> None:
        """Append one event to the current day's overflow file (best-effort).

        M2: Files are created with mode 0o600 (owner read/write only) via
        ``os.open`` so the underlying OS respects the explicit permission bits.

        L4: If the file would exceed ``_MAX_OVERFLOW_FILE_BYTES`` the event is
        dropped (with a warning) to bound disk usage.
        """
        if not self._writable:
            return
        try:
            path = self._current_path()
            # L4: enforce per-file size cap.
            # m-4: the getsize→write sequence is a benign TOCTOU — the flusher
            # is single-threaded so no concurrent writer can interleave between
            # the size check and the append; the worst case is a slightly
            # oversized file, not a security issue.
            try:
                if os.path.getsize(path) >= _MAX_OVERFLOW_FILE_BYTES:
                    _log.warning(
                        "sigil: overflow file %r is full (>= %d bytes); "
                        "event dropped — sigil-core has been unreachable too long",
                        path,
                        _MAX_OVERFLOW_FILE_BYTES,
                    )
                    self._drop_count += 1  # F5: track drops for host-app observability
                    return
            except OSError:
                pass  # file does not exist yet — proceed normally

            # M2: create / append with restricted permissions (0o600).
            line = (json.dumps(event, separators=(",", ":")) + "\n").encode()
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError:
            pass  # graceful degradation — never crash the host app

    def replay(self, client: SigilClient) -> None:
        """Drain all overflow files and send to sigil-core (best-effort, idempotent-ish).

        Called after a successful log_batch to recover previously undelivered events.
        Files are only deleted when ALL their events have been accepted.
        If a partial replay fails mid-file, the file is left intact for the next attempt.
        """
        if not self._writable:
            return
        try:
            entries = sorted(e for e in os.listdir(self._dir) if e.endswith(".ndjson"))
        except OSError:
            return

        for filename in entries:
            self._replay_file(os.path.join(self._dir, filename), client)

    def _replay_file(self, path: str, client: SigilClient) -> None:
        from sigil.errors import SigilAPIError, SigilTransportError

        try:
            with open(path, encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
        except OSError:
            return

        if not lines:
            try:
                os.unlink(path)
            except OSError:
                pass
            return

        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip corrupt lines

        all_ok = True
        for i in range(0, len(events), 100):
            try:
                client.log_batch(events[i : i + 100])
            except (SigilTransportError, SigilAPIError):
                all_ok = False
                break  # leave file for next successful connection

        if all_ok:
            try:
                os.unlink(path)
            except OSError:
                pass
