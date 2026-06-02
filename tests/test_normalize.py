"""
# PROMPT: "Design a normalizer for a retail event ingestion system that handles
# four observed input dialects: spec canonical (UPPER event types, event_id,
# nested metadata), sample entry/exit (id_token, store_code, event_timestamp),
# sample zone (track_id, store_id, event_time), and a collapsed queue lifecycle
# record (queue_event_id, queue_join_ts, queue_exit_ts, abandoned flag) that
# must expand to two canonical events. The normalizer must be a pure function
# accepting an optional store_resolver callable so tests need no filesystem."
#
# CHANGES MADE:
# - Added _to_bool() for is_revenue_zone ("Yes" string → bool) after seeing the
#   sample data used strings instead of booleans.
# - Added _clamp_confidence() instead of rejecting OOB values — spec says
#   don't suppress low-conf events; rejecting on OOB would be inconsistent.
# - Made store_resolver Optional with late import default, so tests that pass
#   a resolver never touch config files.
# - queue_depth = pos_at_join - 1 (not pos_at_join) after re-reading the spec:
#   "BILLING_QUEUE_JOIN when queue_depth > 0", meaning position 1 = depth 0.
# - Unknown top-level keys folded into metadata (not silently dropped) after
#   noticing the held-out event set could carry extra fields.
"""

from __future__ import annotations

from datetime import timezone

import pytest

from app.models import EventType
from app.normalize import normalize
from tests.conftest import (
    SAMPLE_ENTRY,
    SAMPLE_EXIT,
    SAMPLE_GROUP_ENTRY,
    SAMPLE_QUEUE_ABANDONED,
    SAMPLE_QUEUE_COMPLETED,
    SAMPLE_ZONE_ENTERED,
    SAMPLE_ZONE_EXITED,
    SPEC_CANONICAL_ENTRY,
    SPEC_CANONICAL_ZONE_DWELL,
    store_resolver,
)

R = store_resolver  # shorthand


# ── entry/exit dialect ────────────────────────────────────────────────────────

class TestEntryExitDialect:
    def test_entry_event_type(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.event_type == EventType.ENTRY

    def test_exit_event_type(self):
        [ev] = normalize(SAMPLE_EXIT, R)
        assert ev.event_type == EventType.EXIT

    def test_visitor_id_is_id_token(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.visitor_id == "ID_60001"

    def test_store_code_resolved(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.store_id == "ST1076"

    def test_timestamp_is_utc(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.timestamp.tzinfo is not None
        assert ev.timestamp.tzinfo == timezone.utc

    def test_timestamp_value(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.timestamp.year == 2026
        assert ev.timestamp.month == 3
        assert ev.timestamp.day == 8
        assert ev.timestamp.hour == 18
        assert ev.timestamp.minute == 10

    def test_demographics_in_metadata(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.metadata.gender == "F"
        assert ev.metadata.age == 28
        assert ev.metadata.age_bucket == "25-34"
        assert ev.metadata.is_face_hidden is False

    def test_group_fields_in_metadata(self):
        [ev] = normalize(SAMPLE_GROUP_ENTRY, R)
        assert ev.metadata.group_id == "G_10"
        assert ev.metadata.group_size == 2

    def test_no_group_id_is_none(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.metadata.group_id is None
        assert ev.metadata.group_size is None

    def test_event_id_synthesized(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.event_id.startswith("SYN-")

    def test_event_id_deterministic(self):
        [ev1] = normalize(SAMPLE_ENTRY, R)
        [ev2] = normalize(SAMPLE_ENTRY, R)
        assert ev1.event_id == ev2.event_id

    def test_event_id_preserved_if_provided(self):
        raw = {**SAMPLE_ENTRY, "event_id": "my-custom-id"}
        [ev] = normalize(raw, R)
        assert ev.event_id == "my-custom-id"

    def test_is_staff_false(self):
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.is_staff is False

    def test_returns_single_event(self):
        result = normalize(SAMPLE_ENTRY, R)
        assert len(result) == 1


# ── zone dialect ──────────────────────────────────────────────────────────────

class TestZoneDialect:
    def test_zone_entered_event_type(self):
        [ev] = normalize(SAMPLE_ZONE_ENTERED, R)
        assert ev.event_type == EventType.ZONE_ENTER

    def test_zone_exited_event_type(self):
        [ev] = normalize(SAMPLE_ZONE_EXITED, R)
        assert ev.event_type == EventType.ZONE_EXIT

    def test_visitor_id_prefixed_with_trk(self):
        [ev] = normalize(SAMPLE_ZONE_ENTERED, R)
        assert ev.visitor_id == "TRK-101"

    def test_zone_id_passed_through(self):
        [ev] = normalize(SAMPLE_ZONE_ENTERED, R)
        assert ev.zone_id == "PURPLLE_MUM_1076_Z01"

    def test_store_id_resolved(self):
        [ev] = normalize(SAMPLE_ZONE_ENTERED, R)
        assert ev.store_id == "ST1076"

    def test_zone_metadata_fields(self):
        [ev] = normalize(SAMPLE_ZONE_ENTERED, R)
        assert ev.metadata.zone_name == "Left Shelf"
        assert ev.metadata.zone_type == "SHELF"
        assert ev.metadata.is_revenue_zone is True  # "Yes" coerced to True
        assert ev.metadata.zone_hotspot_x == pytest.approx(412.6)
        assert ev.metadata.zone_hotspot_y == pytest.approx(238.4)

    def test_is_revenue_zone_string_coercion(self):
        raw = {**SAMPLE_ZONE_ENTERED, "is_revenue_zone": "Yes"}
        [ev] = normalize(raw, R)
        assert ev.metadata.is_revenue_zone is True

    def test_uses_event_time_field(self):
        [ev] = normalize(SAMPLE_ZONE_ENTERED, R)
        assert ev.timestamp.hour == 18
        assert ev.timestamp.minute == 10

    def test_returns_single_event(self):
        assert len(normalize(SAMPLE_ZONE_ENTERED, R)) == 1


# ── queue lifecycle dialect ───────────────────────────────────────────────────

class TestQueueLifecycleDialect:
    def test_completed_expands_to_two_events(self):
        events = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert len(events) == 2

    def test_abandoned_expands_to_two_events(self):
        events = normalize(SAMPLE_QUEUE_ABANDONED, R)
        assert len(events) == 2

    def test_completed_first_event_is_join(self):
        join, _ = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert join.event_type == EventType.BILLING_QUEUE_JOIN

    def test_completed_second_event_is_complete(self):
        _, close = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert close.event_type == EventType.BILLING_QUEUE_COMPLETE

    def test_abandoned_second_event_is_abandon(self):
        _, close = normalize(SAMPLE_QUEUE_ABANDONED, R)
        assert close.event_type == EventType.BILLING_QUEUE_ABANDON

    def test_join_timestamp_is_queue_join_ts(self):
        join, _ = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert join.timestamp.minute == 13
        assert join.timestamp.second == 5

    def test_close_timestamp_is_queue_exit_ts(self):
        _, close = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert close.timestamp.minute == 15
        assert close.timestamp.second == 31

    def test_queue_depth_is_position_minus_one(self):
        join, _ = normalize(SAMPLE_QUEUE_COMPLETED, R)
        # queue_position_at_join=2 → queue_depth=1
        assert join.metadata.queue_depth == 1

    def test_abandoned_queue_depth(self):
        join, _ = normalize(SAMPLE_QUEUE_ABANDONED, R)
        # queue_position_at_join=4 → queue_depth=3
        assert join.metadata.queue_depth == 3

    def test_close_event_dwell_ms_from_wait_seconds(self):
        _, close = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert close.dwell_ms == 8 * 1000

    def test_close_event_wait_seconds_in_metadata(self):
        _, close = normalize(SAMPLE_QUEUE_ABANDONED, R)
        assert close.metadata.wait_seconds == 65

    def test_two_events_have_different_event_ids(self):
        join, close = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert join.event_id != close.event_id

    def test_expansion_is_deterministic(self):
        events1 = normalize(SAMPLE_QUEUE_COMPLETED, R)
        events2 = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert events1[0].event_id == events2[0].event_id
        assert events1[1].event_id == events2[1].event_id

    def test_visitor_id_is_trk_prefixed(self):
        join, close = normalize(SAMPLE_QUEUE_COMPLETED, R)
        assert join.visitor_id == "TRK-102"
        assert close.visitor_id == "TRK-102"


# ── spec canonical dialect ────────────────────────────────────────────────────

class TestSpecCanonicalDialect:
    def test_entry_event_type(self):
        [ev] = normalize(SPEC_CANONICAL_ENTRY, R)
        assert ev.event_type == EventType.ENTRY

    def test_event_id_preserved(self):
        [ev] = normalize(SPEC_CANONICAL_ENTRY, R)
        assert ev.event_id == "uuid-v4-test-entry-001"

    def test_store_alias_resolved(self):
        # STORE_BLR_002 is the acceptance-gate store ID
        [ev] = normalize(SPEC_CANONICAL_ENTRY, R)
        assert ev.store_id == "ST1008"

    def test_visitor_id_preserved(self):
        [ev] = normalize(SPEC_CANONICAL_ENTRY, R)
        assert ev.visitor_id == "VIS_c8a2f1"

    def test_confidence_preserved(self):
        [ev] = normalize(SPEC_CANONICAL_ENTRY, R)
        assert ev.confidence == pytest.approx(0.91)

    def test_zone_dwell_metadata(self):
        [ev] = normalize(SPEC_CANONICAL_ZONE_DWELL, R)
        assert ev.dwell_ms == 8400
        assert ev.metadata.sku_zone == "MOISTURISER"
        assert ev.metadata.session_seq == 5

    def test_z_suffix_timestamp_parsed(self):
        [ev] = normalize(SPEC_CANONICAL_ENTRY, R)
        assert ev.timestamp.tzinfo == timezone.utc

    def test_returns_single_event(self):
        assert len(normalize(SPEC_CANONICAL_ENTRY, R)) == 1


# ── cross-dialect / edge cases ────────────────────────────────────────────────

class TestEdgeCases:
    def test_missing_event_type_raises(self):
        with pytest.raises(ValueError, match="event_type"):
            normalize({"store_id": "ST1008", "timestamp": "2026-01-01T00:00:00Z"}, R)

    def test_unknown_event_type_raises(self):
        raw = {**SAMPLE_ENTRY, "event_type": "teleport"}
        with pytest.raises(ValueError, match="unrecognized event_type"):
            normalize(raw, R)

    def test_missing_timestamp_raises(self):
        raw = {k: v for k, v in SAMPLE_ENTRY.items() if k != "event_timestamp"}
        with pytest.raises(ValueError, match="timestamp"):
            normalize(raw, R)

    def test_tz_naive_timestamp_becomes_utc(self):
        # sample timestamps are naive (no Z, no +00:00)
        [ev] = normalize(SAMPLE_ENTRY, R)
        assert ev.timestamp.tzinfo == timezone.utc

    def test_unknown_top_level_fields_preserved_in_metadata(self):
        raw = {**SPEC_CANONICAL_ENTRY, "custom_field": "custom_value"}
        [ev] = normalize(raw, R)
        # model_extra stores the overflow
        assert ev.metadata.model_extra.get("custom_field") == "custom_value"

    def test_unknown_store_id_passes_through(self):
        raw = {**SAMPLE_ENTRY, "store_code": "STORE_UNKNOWN_XYZ"}
        [ev] = normalize(raw, R)
        assert ev.store_id == "STORE_UNKNOWN_XYZ"

    def test_confidence_above_one_clamped(self):
        raw = {**SPEC_CANONICAL_ENTRY, "confidence": 1.5}
        [ev] = normalize(raw, R)
        assert ev.confidence == pytest.approx(1.0)

    def test_confidence_below_zero_clamped(self):
        raw = {**SPEC_CANONICAL_ENTRY, "confidence": -0.1}
        [ev] = normalize(raw, R)
        assert ev.confidence == pytest.approx(0.0)

    def test_low_confidence_event_not_suppressed(self):
        # The spec explicitly requires low-conf events to be emitted, not dropped.
        raw = {**SPEC_CANONICAL_ENTRY, "confidence": 0.05}
        result = normalize(raw, R)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.05)

    def test_is_staff_true_passes_through(self):
        raw = {**SAMPLE_ENTRY, "is_staff": True}
        [ev] = normalize(raw, R)
        assert ev.is_staff is True

    def test_age_pred_cast_to_int(self):
        raw = {**SAMPLE_ENTRY, "age_pred": 28.7}
        [ev] = normalize(raw, R)
        assert ev.metadata.age == 28
        assert isinstance(ev.metadata.age, int)

    def test_entry_exit_different_event_ids(self):
        # Same person, same camera — entry vs exit should produce different SYN- IDs
        [entry_ev] = normalize(SAMPLE_ENTRY, R)
        [exit_ev] = normalize(SAMPLE_EXIT, R)
        assert entry_ev.event_id != exit_ev.event_id
