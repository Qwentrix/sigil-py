"""Bounded ring buffer and background flusher thread for audit events.

Design:
- _LogBuffer: thread-safe deque with maxlen=1000 (oldest evicted when full).
  Signals a threading.Event when the buffer reaches 50 events so the flusher
  wakes up immediately rather than waiting for the 500 ms timer.
- _Flusher: daemon thread that drains the buffer every 500 ms (or immediately
  on the 50-event signal), calls client.log_batch in chunks of ≤100, and
  falls back to disk overflow on SigilTransportError.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sigil._overflow import _OverflowWriter
    from sigil.client import SigilClient

_log = logging.getLogger("sigil")

_FLUSH_THRESHOLD: int = 50  # events — immediate flush trigger
_FLUSH_INTERVAL: float = 0.5  # seconds — timer-based flush
_MAX_SEND_BATCH: int = 100  # mirrors _MAX_BATCH_SIZE in client.py
_RING_CAP: int = 1_000  # bounded ring — oldest silently evicted


class _LogBuffer:
    """Thread-safe bounded ring buffer."""

    def __init__(self) -> None:
        self._buf: deque[dict[str, Any]] = deque(maxlen=_RING_CAP)
        self._lock = threading.Lock()
        # Set by push() when len >= _FLUSH_THRESHOLD; cleared by _Flusher.
        self._flush_now: threading.Event = threading.Event()

    def push(self, event: dict[str, Any]) -> None:
        """Append an event; signals the flusher if threshold is reached."""
        with self._lock:
            self._buf.append(event)
            if len(self._buf) >= _FLUSH_THRESHOLD:
                self._flush_now.set()

    def drain(self, n: int) -> list[dict[str, Any]]:
        """Pop and return up to *n* events (FIFO); returns [] if empty."""
        with self._lock:
            count = min(n, len(self._buf))
            return [self._buf.popleft() for _ in range(count)]

    def size(self) -> int:
        """Current number of buffered events."""
        with self._lock:
            return len(self._buf)

    def wake(self) -> None:
        """Signal the flusher to run immediately (used by close/exit)."""
        self._flush_now.set()

    @property
    def flush_event(self) -> threading.Event:
        """The threading.Event used to trigger an immediate flush."""
        return self._flush_now


class _Flusher(threading.Thread):
    """Background daemon thread: drains _LogBuffer → sigil-core → overflow."""

    def __init__(
        self,
        buffer: _LogBuffer,
        client: SigilClient,
        overflow: _OverflowWriter,
    ) -> None:
        super().__init__(daemon=True, name="sigil-log-flusher")
        self._buffer = buffer
        self._client = client
        self._overflow = overflow
        self._stop = threading.Event()

    def run(self) -> None:
        """Main loop: wait up to 500 ms (or until woken), then flush."""
        while not self._stop.is_set():
            triggered = self._buffer.flush_event.wait(timeout=_FLUSH_INTERVAL)
            if triggered:
                self._buffer.flush_event.clear()
            self._do_flush()

    def flush_all(self) -> None:
        """Synchronously drain all buffered events.

        Called by SigilTaskContext.__exit__ and SigilClient.close() so that
        events are delivered before the context exits or the process shuts down.
        """
        self._do_flush()

    def stop(self) -> None:
        """Signal the flusher to stop and unblock its wait."""
        self._stop.set()
        self._buffer.wake()

    def _do_flush(self) -> None:
        from sigil.errors import SigilAPIError, SigilTransportError

        while True:
            events = self._buffer.drain(_MAX_SEND_BATCH)
            if not events:
                break
            try:
                self._client.log_batch(events)
                # On a successful delivery, try to drain any overflow files.
                self._overflow.replay(self._client)
            except SigilTransportError:
                # sigil-core unreachable — persist to disk for later replay.
                for ev in events:
                    self._overflow.write(ev)
                break  # stop draining; server is down
            except SigilAPIError:
                # Unexpected status code — not retryable; log and drop.
                _log.warning(
                    "sigil: log_batch returned unexpected HTTP status; %d event(s) dropped",
                    len(events),
                )
                break
            except Exception:  # noqa: BLE001
                _log.debug("sigil: unexpected error in log flusher", exc_info=True)
                break
