"""
Repository Protocol — the storage contract that ingestion.py depends on.

Stage 2 adds the SQLite implementation in this same file.
Having the Protocol here (not in ingestion.py) means both the API and
future CLI tools can depend on it without importing ingestion logic.

Why runtime_checkable: lets tests do `assert isinstance(repo, EventRepository)`
without needing an explicit base class.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models import CanonicalEvent


@runtime_checkable
class EventRepository(Protocol):
    def exists(self, event_id: str) -> bool:
        """Return True if an event with this event_id is already stored."""
        ...

    def save(self, event: CanonicalEvent) -> None:
        """Persist a canonical event. Caller guarantees event_id is unique."""
        ...
