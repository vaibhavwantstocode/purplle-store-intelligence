"""
Normalizes raw event dicts from 4 observed input dialects into CanonicalEvent(s).

Dialect detection priority (ordered to handle overlapping key sets):
  1. event_type in {queue_completed, queue_abandoned}  → collapsed queue lifecycle
  2. id_token present                                  → sample entry/exit dialect
  3. track_id present AND zone event_type              → sample zone dialect
  4. everything else                                   → spec canonical / best-effort

The normalizer is a pure function: it never touches the database or the
filesystem. store_resolver is injected so tests can run without config files.

Queue lifecycle records expand to TWO CanonicalEvents:
  queue_completed  → BILLING_QUEUE_JOIN  +  BILLING_QUEUE_COMPLETE
  queue_abandoned  → BILLING_QUEUE_JOIN  +  BILLING_QUEUE_ABANDON

All other dialects produce exactly one CanonicalEvent.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.models import CanonicalEvent, EventMetadata, EventType

# ── constants ─────────────────────────────────────────────────────────────────

_QUEUE_LIFECYCLE_TYPES: frozenset[str] = frozenset(["queue_completed", "queue_abandoned"])
_ZONE_EVENT_TYPES: frozenset[str] = frozenset(["zone_entered", "zone_exited", "zone_dwell"])

# Every event_type string ever observed maps to a canonical EventType.
# Unknown strings cause ValueError (rather than silent corruption of analytics).
_EVENT_TYPE_MAP: dict[str, EventType] = {
    # spec canonical — UPPER_SNAKE
    "ENTRY": EventType.ENTRY,
    "EXIT": EventType.EXIT,
    "ZONE_ENTER": EventType.ZONE_ENTER,
    "ZONE_EXIT": EventType.ZONE_EXIT,
    "ZONE_DWELL": EventType.ZONE_DWELL,
    "BILLING_QUEUE_JOIN": EventType.BILLING_QUEUE_JOIN,
    "BILLING_QUEUE_ABANDON": EventType.BILLING_QUEUE_ABANDON,
    "BILLING_QUEUE_COMPLETE": EventType.BILLING_QUEUE_COMPLETE,
    "REENTRY": EventType.REENTRY,
    # sample entry/exit dialect — lower
    "entry": EventType.ENTRY,
    "exit": EventType.EXIT,
    # sample zone dialect — lower
    "zone_entered": EventType.ZONE_ENTER,
    "zone_exited": EventType.ZONE_EXIT,
    "zone_dwell": EventType.ZONE_DWELL,
    # sample queue lifecycle — these are the collapsed forms (expanded below)
    "queue_completed": EventType.BILLING_QUEUE_COMPLETE,
    "queue_abandoned": EventType.BILLING_QUEUE_ABANDON,
}

# Top-level fields that belong to the canonical schema (not forwarded to metadata)
_CANONICAL_TOP_LEVEL: frozenset[str] = frozenset([
    "event_id", "store_id", "camera_id", "visitor_id", "event_type",
    "timestamp", "zone_id", "dwell_ms", "is_staff", "confidence", "metadata",
])

# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(val: Any) -> datetime:
    """Parse an ISO-8601 timestamp string to an aware UTC datetime."""
    if val is None:
        raise ValueError("timestamp is required")
    s = str(val).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp {val!r}: {exc}") from exc
    # Treat naive timestamps as UTC (all observed samples are UTC without tz marker)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _make_event_id(*parts: Any) -> str:
    """
    Deterministic synthetic event_id from natural keys.
    Same inputs always produce the same ID → safe to call ingest twice (idempotent).
    Prefixed SYN- so logs can distinguish synthesized from provided IDs.
    """
    raw = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"SYN-{digest}"


def _map_event_type(raw: str) -> EventType:
    et = _EVENT_TYPE_MAP.get(raw)
    if et is None:
        raise ValueError(f"unrecognized event_type: {raw!r}")
    return et


def _to_bool(val: Any) -> Optional[bool]:
    """Coerce "Yes"/"No"/True/False/None to Optional[bool]."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("yes", "true", "1")


def _clamp_confidence(val: Any) -> float:
    """Clamp to [0, 1] rather than rejecting — detectors can emit slightly OOB values."""
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return 1.0


# ── public interface ──────────────────────────────────────────────────────────

def normalize(
    raw: dict[str, Any],
    store_resolver: Optional[Callable[[str], str]] = None,
) -> list[CanonicalEvent]:
    """
    Normalize one raw event dict into one or more CanonicalEvents.

    Returns a list because collapsed queue records expand to 2 events.
    Raises ValueError with a descriptive message on unrecoverable input
    (missing event_type, missing timestamp, unknown event_type string).
    Unknown *fields* are preserved in EventMetadata, never rejected.
    """
    if store_resolver is None:
        from app.config import resolve_store_id
        store_resolver = resolve_store_id

    event_type_raw = raw.get("event_type")
    if event_type_raw is None:
        raise ValueError("missing required field: event_type")

    if event_type_raw in _QUEUE_LIFECYCLE_TYPES:
        return _normalize_queue_lifecycle(raw, store_resolver)
    if "id_token" in raw:
        return [_normalize_entry_exit(raw, store_resolver)]
    if "track_id" in raw and event_type_raw in _ZONE_EVENT_TYPES:
        return [_normalize_zone(raw, store_resolver)]
    return [_normalize_canonical(raw, store_resolver)]


# ── dialect adapters ──────────────────────────────────────────────────────────

def _resolve_store(raw: dict[str, Any], store_resolver: Callable[[str], str]) -> str:
    """Try store_code first (entry/exit dialect), then store_id."""
    raw_id = raw.get("store_code") or raw.get("store_id") or ""
    return store_resolver(raw_id) if raw_id else "UNKNOWN"


def _normalize_entry_exit(
    raw: dict[str, Any],
    store_resolver: Callable[[str], str],
) -> CanonicalEvent:
    """Sample dialect: entry/exit events carrying id_token + store_code."""
    id_token: str = raw.get("id_token") or "UNKNOWN"
    event_type_raw: str = raw.get("event_type", "")
    ts_raw = raw.get("event_timestamp")

    event_id = raw.get("event_id") or _make_event_id(id_token, event_type_raw, ts_raw)

    return CanonicalEvent(
        event_id=event_id,
        store_id=_resolve_store(raw, store_resolver),
        camera_id=str(raw.get("camera_id", "UNKNOWN")),
        visitor_id=id_token,
        event_type=_map_event_type(event_type_raw),
        timestamp=_parse_ts(ts_raw),
        zone_id=raw.get("zone_id"),
        dwell_ms=int(raw.get("dwell_ms") or 0),
        is_staff=bool(raw.get("is_staff", False)),
        confidence=_clamp_confidence(raw.get("confidence", 1.0)),
        metadata=EventMetadata(
            gender=raw.get("gender_pred"),
            age=int(raw["age_pred"]) if raw.get("age_pred") is not None else None,
            age_bucket=raw.get("age_bucket"),
            is_face_hidden=raw.get("is_face_hidden"),
            group_id=raw.get("group_id"),
            group_size=raw.get("group_size"),
        ),
    )


def _normalize_zone(
    raw: dict[str, Any],
    store_resolver: Callable[[str], str],
) -> CanonicalEvent:
    """Sample dialect: zone_entered/zone_exited/zone_dwell events carrying track_id."""
    track_id = raw.get("track_id")
    event_type_raw: str = raw.get("event_type", "")
    # Zone events use event_time; fall back to timestamp for forward-compat
    ts_raw = raw.get("event_time") or raw.get("timestamp")

    visitor_id = f"TRK-{track_id}" if track_id is not None else "UNKNOWN"
    event_id = raw.get("event_id") or _make_event_id(
        track_id, raw.get("store_id"), event_type_raw, ts_raw
    )

    return CanonicalEvent(
        event_id=event_id,
        store_id=_resolve_store(raw, store_resolver),
        camera_id=str(raw.get("camera_id", "UNKNOWN")),
        visitor_id=visitor_id,
        event_type=_map_event_type(event_type_raw),
        timestamp=_parse_ts(ts_raw),
        zone_id=raw.get("zone_id"),
        dwell_ms=int(raw.get("dwell_ms") or 0),
        is_staff=bool(raw.get("is_staff", False)),
        confidence=_clamp_confidence(raw.get("confidence", 1.0)),
        metadata=EventMetadata(
            zone_name=raw.get("zone_name"),
            zone_type=raw.get("zone_type"),
            is_revenue_zone=_to_bool(raw.get("is_revenue_zone")),
            zone_hotspot_x=raw.get("zone_hotspot_x"),
            zone_hotspot_y=raw.get("zone_hotspot_y"),
            gender=raw.get("gender"),
            age=int(raw["age"]) if raw.get("age") is not None else None,
            age_bucket=raw.get("age_bucket"),
        ),
    )


def _normalize_queue_lifecycle(
    raw: dict[str, Any],
    store_resolver: Callable[[str], str],
) -> list[CanonicalEvent]:
    """
    Sample dialect: collapsed queue lifecycle (queue_completed / queue_abandoned).
    One raw record → two CanonicalEvents:
      [0] BILLING_QUEUE_JOIN   at queue_join_ts
      [1] BILLING_QUEUE_COMPLETE or BILLING_QUEUE_ABANDON at queue_exit_ts

    queue_depth = queue_position_at_join - 1  (position 1 means no queue ahead)
    """
    track_id = raw.get("track_id")
    queue_event_id: str = raw.get("queue_event_id") or str(track_id) or "UNKNOWN"
    abandoned: bool = bool(raw.get("abandoned", False))

    visitor_id = f"TRK-{track_id}" if track_id is not None else "UNKNOWN"
    store_id = _resolve_store(raw, store_resolver)
    camera_id = str(raw.get("camera_id", "UNKNOWN"))
    zone_id = raw.get("zone_id")
    confidence = _clamp_confidence(raw.get("confidence", 1.0))

    pos_at_join: int = int(raw.get("queue_position_at_join") or 1)
    queue_depth: int = max(0, pos_at_join - 1)
    wait_seconds = raw.get("wait_seconds")

    join_ts_raw = raw.get("queue_join_ts")
    exit_ts_raw = raw.get("queue_exit_ts")

    # Shared zone enrichment fields
    zone_meta = dict(
        zone_name=raw.get("zone_name"),
        zone_type=raw.get("zone_type"),
        is_revenue_zone=_to_bool(raw.get("is_revenue_zone")),
        zone_hotspot_x=raw.get("zone_hotspot_x"),
        zone_hotspot_y=raw.get("zone_hotspot_y"),
        gender=raw.get("gender"),
        age=int(raw["age"]) if raw.get("age") is not None else None,
        age_bucket=raw.get("age_bucket"),
    )

    join_event = CanonicalEvent(
        event_id=_make_event_id(queue_event_id, "JOIN", join_ts_raw),
        store_id=store_id,
        camera_id=camera_id,
        visitor_id=visitor_id,
        event_type=EventType.BILLING_QUEUE_JOIN,
        timestamp=_parse_ts(join_ts_raw),
        zone_id=zone_id,
        dwell_ms=0,
        is_staff=False,
        confidence=confidence,
        metadata=EventMetadata(
            queue_depth=queue_depth,
            queue_position_at_join=pos_at_join,
            **zone_meta,
        ),
    )

    close_type = EventType.BILLING_QUEUE_ABANDON if abandoned else EventType.BILLING_QUEUE_COMPLETE
    close_event = CanonicalEvent(
        event_id=_make_event_id(queue_event_id, "ABANDON" if abandoned else "COMPLETE", exit_ts_raw),
        store_id=store_id,
        camera_id=camera_id,
        visitor_id=visitor_id,
        event_type=close_type,
        timestamp=_parse_ts(exit_ts_raw),
        zone_id=zone_id,
        # dwell_ms from the wait period (ms precision overkill; seconds is what the sample gives)
        dwell_ms=int(wait_seconds or 0) * 1000,
        is_staff=False,
        confidence=confidence,
        metadata=EventMetadata(
            wait_seconds=int(wait_seconds) if wait_seconds is not None else None,
            queue_position_at_join=pos_at_join,
            **zone_meta,
        ),
    )

    return [join_event, close_event]


def _normalize_canonical(
    raw: dict[str, Any],
    store_resolver: Callable[[str], str],
) -> CanonicalEvent:
    """
    Spec canonical dialect (or any unrecognized format).
    Field names match the spec 1:1. Unknown top-level keys are folded into
    metadata so they survive ingestion rather than being silently dropped.
    """
    event_type_raw: str = raw.get("event_type", "")
    store_raw: str = raw.get("store_id", "")
    ts_raw = raw.get("timestamp")

    event_id = raw.get("event_id") or _make_event_id(
        store_raw, raw.get("camera_id"), raw.get("visitor_id"), event_type_raw, ts_raw
    )

    # Start with the nested metadata dict (spec canonical), then fold in any
    # unknown top-level keys so no input field is ever silently discarded.
    meta_dict: dict[str, Any] = dict(raw.get("metadata") or {})
    for k, v in raw.items():
        if k not in _CANONICAL_TOP_LEVEL:
            meta_dict.setdefault(k, v)

    return CanonicalEvent(
        event_id=event_id,
        store_id=store_resolver(store_raw) if store_raw else "UNKNOWN",
        camera_id=str(raw.get("camera_id", "UNKNOWN")),
        visitor_id=str(raw.get("visitor_id", "UNKNOWN")),
        event_type=_map_event_type(event_type_raw),
        timestamp=_parse_ts(ts_raw),
        zone_id=raw.get("zone_id"),
        dwell_ms=int(raw.get("dwell_ms") or 0),
        is_staff=bool(raw.get("is_staff", False)),
        confidence=_clamp_confidence(raw.get("confidence", 1.0)),
        metadata=EventMetadata(**meta_dict) if meta_dict else EventMetadata(),
    )
