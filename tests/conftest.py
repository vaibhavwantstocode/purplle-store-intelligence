"""
Shared fixtures for all test modules.

InMemoryRepository: satisfies the EventRepository Protocol without touching
the filesystem. Used in both test_normalize.py and test_ingestion.py so
the service layer can be tested without a database.

test_store_resolver: a deterministic alias map that tests can inject into
normalize() to avoid loading config/stores.yaml during the test run.
"""
from __future__ import annotations

import pytest

from app.models import CanonicalEvent


# ── in-memory repository ──────────────────────────────────────────────────────

class InMemoryRepository:
    def __init__(self) -> None:
        self._store: dict[str, CanonicalEvent] = {}

    def exists(self, event_id: str) -> bool:
        return event_id in self._store

    def save(self, event: CanonicalEvent) -> None:
        self._store[event.event_id] = event

    # Test helpers (not part of the Protocol — tests only)
    def all_events(self) -> list[CanonicalEvent]:
        return list(self._store.values())

    def count(self) -> int:
        return len(self._store)

    def get(self, event_id: str) -> CanonicalEvent | None:
        return self._store.get(event_id)


@pytest.fixture
def repo() -> InMemoryRepository:
    return InMemoryRepository()


# ── store resolver ────────────────────────────────────────────────────────────

_ALIAS_MAP: dict[str, str] = {
    "STORE_BLR_002": "ST1008",
    "store_1076": "ST1076",
    "store_1008": "ST1008",
    "ST1008": "ST1008",
    "ST1076": "ST1076",
}


def store_resolver(raw_id: str) -> str:
    """Deterministic alias resolution for tests. No file I/O."""
    return _ALIAS_MAP.get(raw_id, raw_id)


# ── raw event fixtures (exact bytes from sample_events.jsonl) ─────────────────

SAMPLE_ENTRY = {
    "event_type": "entry",
    "id_token": "ID_60001",
    "store_code": "store_1076",
    "camera_id": "cam1",
    "event_timestamp": "2026-03-08T18:10:05.120000",
    "is_staff": False,
    "gender_pred": "F",
    "age_pred": 28,
    "age_bucket": "25-34",
    "is_face_hidden": False,
    "group_id": None,
    "group_size": None,
}

SAMPLE_EXIT = {
    "event_type": "exit",
    "id_token": "ID_60001",
    "store_code": "store_1076",
    "camera_id": "cam1",
    "event_timestamp": "2026-03-08T18:12:44.360000",
    "is_staff": False,
    "gender_pred": "F",
    "age_pred": 28,
    "age_bucket": "25-34",
    "is_face_hidden": False,
    "group_id": None,
    "group_size": None,
}

SAMPLE_GROUP_ENTRY = {
    "event_type": "entry",
    "id_token": "ID_60002",
    "store_code": "store_1076",
    "camera_id": "cam1",
    "event_timestamp": "2026-03-08T18:10:22.480000",
    "is_staff": False,
    "gender_pred": "M",
    "age_pred": 31,
    "age_bucket": "25-34",
    "is_face_hidden": False,
    "group_id": "G_10",
    "group_size": 2,
}

SAMPLE_ZONE_ENTERED = {
    "event_type": "zone_entered",
    "track_id": 101,
    "store_id": "ST1076",
    "camera_id": "CAM2",
    "zone_id": "PURPLLE_MUM_1076_Z01",
    "zone_name": "Left Shelf",
    "zone_type": "SHELF",
    "is_revenue_zone": "Yes",
    "event_time": "2026-03-08T18:10:45.280000",
    "zone_hotspot_x": 412.6,
    "zone_hotspot_y": 238.4,
    "gender": "F",
    "age": 28,
    "age_bucket": "25-34",
}

SAMPLE_ZONE_EXITED = {
    "event_type": "zone_exited",
    "track_id": 101,
    "store_id": "ST1076",
    "camera_id": "CAM2",
    "zone_id": "PURPLLE_MUM_1076_Z01",
    "zone_name": "Left Shelf",
    "zone_type": "SHELF",
    "is_revenue_zone": "Yes",
    "event_time": "2026-03-08T18:11:18.720000",
    "zone_hotspot_x": 418.2,
    "zone_hotspot_y": 241.0,
    "gender": "F",
    "age": 28,
    "age_bucket": "25-34",
}

SAMPLE_QUEUE_COMPLETED = {
    "queue_event_id": "cfd8e3c5-7aa0-4ea3-9b59-692d50da8308",
    "event_type": "queue_completed",
    "track_id": 102,
    "store_id": "ST1076",
    "camera_id": "PURPLLE_MUM_1076_CAM6",
    "zone_id": "PURPLLE_MUM_1076_Z_BILLING_01",
    "zone_name": "Billing Counter Queue",
    "zone_type": "BILLING",
    "is_revenue_zone": "Yes",
    "queue_join_ts": "2026-03-08T18:13:05.080000",
    "queue_served_ts": "2026-03-08T18:13:13.240000",
    "queue_exit_ts": "2026-03-08T18:15:31.840000",
    "wait_seconds": 8,
    "queue_position_at_join": 2,
    "abandoned": False,
    "zone_hotspot_x": 602.8,
    "zone_hotspot_y": 183.4,
    "gender": "M",
    "age": 31,
    "age_bucket": "25-34",
}

SAMPLE_QUEUE_ABANDONED = {
    "queue_event_id": "a1e5c1d3-9e14-4df1-bd2c-4ab5cbf55f91",
    "event_type": "queue_abandoned",
    "track_id": 101,
    "store_id": "ST1076",
    "camera_id": "PURPLLE_MUM_1076_CAM6",
    "zone_id": "PURPLLE_MUM_1076_Z_BILLING_01",
    "zone_name": "Billing Counter Queue",
    "zone_type": "BILLING",
    "is_revenue_zone": "Yes",
    "queue_join_ts": "2026-03-08T18:12:58.240000",
    "queue_served_ts": None,
    "queue_exit_ts": "2026-03-08T18:14:02.880000",
    "wait_seconds": 65,
    "queue_position_at_join": 4,
    "abandoned": True,
    "zone_hotspot_x": 598.1,
    "zone_hotspot_y": 176.8,
    "gender": "F",
    "age": 28,
    "age_bucket": "25-34",
}

SPEC_CANONICAL_ENTRY = {
    "event_id": "uuid-v4-test-entry-001",
    "store_id": "STORE_BLR_002",
    "camera_id": "CAM_ENTRY_01",
    "visitor_id": "VIS_c8a2f1",
    "event_type": "ENTRY",
    "timestamp": "2026-03-03T14:22:10Z",
    "zone_id": None,
    "dwell_ms": 0,
    "is_staff": False,
    "confidence": 0.91,
    "metadata": {
        "queue_depth": None,
        "sku_zone": None,
        "session_seq": 1,
    },
}

SPEC_CANONICAL_ZONE_DWELL = {
    "event_id": "uuid-v4-test-dwell-001",
    "store_id": "ST1008",
    "camera_id": "CAM1_zone",
    "visitor_id": "VIS_c8a2f1",
    "event_type": "ZONE_DWELL",
    "timestamp": "2026-03-03T14:22:10Z",
    "zone_id": "SKINCARE",
    "dwell_ms": 8400,
    "is_staff": False,
    "confidence": 0.87,
    "metadata": {
        "queue_depth": None,
        "sku_zone": "MOISTURISER",
        "session_seq": 5,
    },
}
