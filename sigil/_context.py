"""Contextvar holding the active SigilTaskContext for the current async/thread context.

Imported by client.py (to set/reset) and by decorators.py (to read).
Kept in its own module to break the potential circular import between
client.py and decorators.py.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Typed as Any at runtime; callers that need the full SigilTaskContext type
# should import it under TYPE_CHECKING and cast accordingly.
_current_task: ContextVar[Any] = ContextVar("sigil_current_task", default=None)
