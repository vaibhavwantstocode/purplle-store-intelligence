"""
Ingest service: normalize → deduplicate → persist.

Accepts a batch of raw event dicts (any supported dialect), normalizes each
to CanonicalEvent(s), skips already-seen event_ids (idempotent), and returns
a structured partial-success response so callers know exactly which raw events
failed and why.

Intentionally framework-agnostic: takes a Repository, returns IngestResponse.
main.py wires in the concrete repo via FastAPI dependency injection.
"""
from __future__ import annotations

from typing import Any

from app.models import CanonicalEvent, IngestError, IngestResponse
from app.normalize import normalize
from app.repository import EventRepository

_MAX_BATCH_SIZE = 500


def ingest_batch(
    raw_events: list[dict[str, Any]],
    repo: EventRepository,
) -> IngestResponse:
    """
    Process a batch of raw event dicts.

    Rules:
    - Batch > 500 items → ValueError (caller turns this into HTTP 400).
    - Each raw event is normalized independently; failure of one never blocks others.
    - A collapsed queue record (queue_completed/queue_abandoned) produces 2 canonical
      events; both are stored under separate event_ids.
    - event_id collision (duplicate) → increment duplicates, do not re-store.
    - accepted counts canonical events stored, not raw events received.
      (One raw queue record → accepted += 2 if both are new.)
    """
    if len(raw_events) > _MAX_BATCH_SIZE:
        raise ValueError(
            f"batch size {len(raw_events)} exceeds the limit of {_MAX_BATCH_SIZE}"
        )

    accepted = 0
    duplicates = 0
    errors: list[IngestError] = []

    for i, raw in enumerate(raw_events):
        try:
            canonical_events: list[CanonicalEvent] = normalize(raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(IngestError(index=i, error=str(exc)))
            continue

        for event in canonical_events:
            if repo.exists(event.event_id):
                duplicates += 1
            else:
                repo.save(event)
                accepted += 1

    return IngestResponse(
        accepted=accepted,
        duplicates=duplicates,
        errors=errors,
        total_received=len(raw_events),
    )
