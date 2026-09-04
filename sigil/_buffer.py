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

# SG-9 SP-1 (ENT-91): the buffer carries two event kinds. Tool-invocation events keep
# their exact on-wire shape and have NO kind key (absence ⇒ tool). Prompt-log events
# (@instrumented_llm enrichment) are tagged with _KIND_KEY == _PROMPT_LOG_KIND so the
# flusher and the overflow replayer route them to client.log_prompt instead of
# client.log_batch. Keeping tool events untagged means their wire format is unchanged.
_KIND_KEY: str = "_sigil_kind"
_PROMPT_LOG_KIND: str = "prompt_log"


def _partition_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split drained/overflow events into ``(tool_events, prompt_log_events)``.

    Tool events (the untagged majority) are returned first so the caller can send them
    through the existing ``log_batch`` path unchanged; prompt-log events are routed
    individually via :func:`_send_prompt_log`.
    """
    tool_events: list[dict[str, Any]] = []
    prompt_events: list[dict[str, Any]] = []
    for ev in events:
        if ev.get(_KIND_KEY) == _PROMPT_LOG_KIND:
            prompt_events.append(ev)
        else:
            tool_events.append(ev)
    return tool_events, prompt_events


def _send_prompt_log(client: SigilClient, ev: dict[str, Any]) -> None:
    """Dispatch one buffered/overflowed prompt-log event to ``client.log_prompt``.

    Raises whatever ``client.log_prompt`` raises (Sigil{Transport,API}Error) so the
    caller can apply the same overflow-persist / drop semantics used for tool events.
    """
    client.log_prompt(
        ev["task_id"],
        ev["agent_id"],
        prompt_hash=ev["prompt_hash"],
        prompt_redacted=ev.get("prompt_redacted") or {},
        response_hash=ev["response_hash"],
        response_sampled=ev.get("response_sampled"),
        model=ev["model"],
        model_provider=ev["model_provider"],
        token_count_input=ev.get("token_count_input", 0),
        token_count_output=ev.get("token_count_output", 0),
        latency_ms=ev.get("latency_ms"),
    )


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
        while True:
            events = self._buffer.drain(_MAX_SEND_BATCH)
            if not events:
                break
            # SG-9 SP-1: a drained page may mix tool-invocation events (→ log_batch) and
            # prompt-log events (→ log_prompt, one call each). Each kind persists its own
            # failures to overflow. Preserve the original loop semantics: stop draining on
            # ANY failure (server likely down / non-retryable), and only opportunistically
            # replay overflow after a fully-clean page.
            tool_events, prompt_events = _partition_events(events)
            ok = True
            if tool_events and not self._flush_tool_events(tool_events):
                ok = False
            # Once a prompt-log hits a transport error, sigil-core is unreachable — persist the
            # REST of the page straight to overflow without hammering the network with up to
            # _MAX_SEND_BATCH more per-event calls (each up to the client timeout).
            prompt_down = False
            for ev in prompt_events:
                if prompt_down:
                    self._overflow.write(ev)
                    ok = False
                    continue
                status = self._flush_one_prompt_log(ev)
                if status == "down":
                    prompt_down = True
                    ok = False
                elif status == "drop":
                    ok = False
            if not ok:
                break
            self._overflow.replay(self._client)

    def _flush_tool_events(self, events: list[dict[str, Any]]) -> bool:
        """Send a batch of tool-invocation events. Returns True on clean delivery.

        On a transport error the events are persisted to overflow for later replay;
        an unexpected HTTP status is non-retryable (logged + dropped). Either failure
        returns False so the caller stops draining this cycle.
        """
        from sigil.errors import SigilAPIError, SigilTransportError

        try:
            self._client.log_batch(events)
            return True
        except SigilTransportError:
            for ev in events:
                self._overflow.write(ev)
            return False
        except SigilAPIError:
            _log.warning(
                "sigil: log_batch returned unexpected HTTP status; %d event(s) dropped",
                len(events),
            )
            return False
        except Exception:  # noqa: BLE001
            _log.debug("sigil: unexpected error flushing tool events", exc_info=True)
            return False

    def _flush_one_prompt_log(self, ev: dict[str, Any]) -> str:
        """Send one prompt-log event (SG-9 SP-1). Returns ``"ok"`` | ``"down"`` | ``"drop"``.

        ``"down"`` (transport error) means sigil-core is unreachable — the event is persisted
        to overflow and the caller should stop attempting the network for the rest of the page.
        ``"drop"`` (unexpected status) is non-retryable and dropped. A prompt-log is enrichment
        on top of the authoritative tool-invocation event, so a dropped one never means the call
        went unaudited.
        """
        from sigil.errors import SigilAPIError, SigilTransportError

        try:
            _send_prompt_log(self._client, ev)
            return "ok"
        except SigilTransportError:
            self._overflow.write(ev)
            return "down"
        except SigilAPIError:
            _log.warning("sigil: log_prompt returned unexpected HTTP status; 1 prompt-log dropped")
            return "drop"
        except Exception:  # noqa: BLE001
            _log.debug("sigil: unexpected error flushing prompt-log", exc_info=True)
            return "drop"
