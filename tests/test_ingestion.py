"""
# PROMPT: "Write integration tests for an event ingestion service that:
# normalizes raw events from multiple dialects, deduplicates by event_id,
# stores via a Repository Protocol, and returns a partial-success response.
# Key requirements: idempotent double-ingest, mixed-validity batches, batch
# size enforcement, collapsed queue record expansion counted correctly."
#
# CHANGES MADE:
# - Added test_collapsed_queue_counted_as_two_accepted: the spec says a
#   collapsed queue record expands to 2 canonical events, so accepted should
#   reflect 2 stored, not 1 raw input.
# - Added test_duplicate_within_batch: same raw event twice in one batch.
#   The first goes through; the second sees the event_id exists → duplicate.
# - HTTP smoke test added (not in AI prompt) to verify the FastAPI route wires
#   the service correctly; catches wiring bugs that unit tests miss.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.ingestion import ingest_batch, _MAX_BATCH_SIZE
from app.models import EventType
from tests.conftest import (
    SAMPLE_ENTRY,
    SAMPLE_EXIT,
    SAMPLE_GROUP_ENTRY,
    SAMPLE_QUEUE_COMPLETED,
    SAMPLE_QUEUE_ABANDONED,
    SAMPLE_ZONE_ENTERED,
    SPEC_CANONICAL_ENTRY,
    InMemoryRepository,
    store_resolver,
)

# Patch normalize to use the test resolver (no filesystem access)
import app.normalize as _normalize_module


@pytest.fixture(autouse=True)
def patch_resolver(monkeypatch):
    """Inject store_resolver into the normalize module's default import path."""
    _resolver = store_resolver  # capture before lambda shadows the name
    monkeypatch.setattr(
        _normalize_module,
        "normalize",
        lambda raw, _r=_resolver: _normalize_module.normalize(raw, _r),
    )


# ── helper ────────────────────────────────────────────────────────────────────

def _ingest(raw_events: list[dict[str, Any]], repo=None):
    if repo is None:
        repo = InMemoryRepository()
    result = ingest_batch(raw_events, repo)
    return result, repo


# ── basic ingestion ───────────────────────────────────────────────────────────

class TestBasicIngestion:
    def test_single_valid_event_accepted(self):
        result, repo = _ingest([SAMPLE_ENTRY])
        assert result.accepted == 1
        assert result.duplicates == 0
        assert result.errors == []
        assert result.total_received == 1
        assert repo.count() == 1

    def test_multiple_valid_events_accepted(self):
        batch = [SAMPLE_ENTRY, SAMPLE_EXIT, SAMPLE_ZONE_ENTERED]
        result, repo = _ingest(batch)
        assert result.accepted == 3
        assert result.total_received == 3
        assert repo.count() == 3

    def test_empty_batch_is_valid(self):
        result, _ = _ingest([])
        assert result.accepted == 0
        assert result.duplicates == 0
        assert result.errors == []
        assert result.total_received == 0

    def test_event_stored_with_correct_type(self):
        _, repo = _ingest([SAMPLE_ENTRY])
        stored = repo.all_events()
        assert stored[0].event_type == EventType.ENTRY

    def test_store_id_resolved_on_stored_event(self):
        _, repo = _ingest([SAMPLE_ENTRY])
        assert repo.all_events()[0].store_id == "ST1076"


# ── idempotency ───────────────────────────────────────────────────────────────

class TestIdempotency:
    def test_double_ingest_same_batch(self):
        repo = InMemoryRepository()
        batch = [SAMPLE_ENTRY, SAMPLE_EXIT]

        r1 = ingest_batch(batch, repo)
        r2 = ingest_batch(batch, repo)

        assert r1.accepted == 2
        assert r1.duplicates == 0
        assert r2.accepted == 0
        assert r2.duplicates == 2
        # State unchanged after second ingest
        assert repo.count() == 2

    def test_double_ingest_single_event(self):
        repo = InMemoryRepository()
        r1 = ingest_batch([SPEC_CANONICAL_ENTRY], repo)
        r2 = ingest_batch([SPEC_CANONICAL_ENTRY], repo)

        assert r1.accepted == 1
        assert r2.duplicates == 1
        assert repo.count() == 1

    def test_idempotency_preserves_stored_content(self):
        repo = InMemoryRepository()
        ingest_batch([SPEC_CANONICAL_ENTRY], repo)
        before = repo.all_events()[0].model_dump()
        ingest_batch([SPEC_CANONICAL_ENTRY], repo)
        after = repo.all_events()[0].model_dump()
        assert before == after


# ── partial success ───────────────────────────────────────────────────────────

class TestPartialSuccess:
    def test_malformed_event_produces_error_entry(self):
        bad = {"store_id": "ST1008"}  # missing event_type
        result, _ = _ingest([bad])
        assert len(result.errors) == 1
        assert result.errors[0].index == 0
        assert "event_type" in result.errors[0].error

    def test_valid_events_stored_despite_earlier_bad_event(self):
        bad = {"store_id": "ST1008"}
        result, repo = _ingest([bad, SAMPLE_ENTRY, SAMPLE_EXIT])
        assert result.accepted == 2
        assert len(result.errors) == 1
        assert repo.count() == 2

    def test_valid_events_stored_despite_later_bad_event(self):
        bad = {"event_type": "teleport", "store_id": "ST1008", "timestamp": "2026-01-01T00:00:00Z"}
        result, repo = _ingest([SAMPLE_ENTRY, bad, SAMPLE_EXIT])
        assert result.accepted == 2
        assert result.errors[0].index == 1
        assert repo.count() == 2

    def test_error_index_matches_raw_batch_position(self):
        batch = [
            SAMPLE_ENTRY,              # index 0 — valid
            {"broken": True},          # index 1 — invalid
            SAMPLE_EXIT,               # index 2 — valid
            {"also_broken": True},     # index 3 — invalid
        ]
        result, _ = _ingest(batch)
        error_indexes = [e.index for e in result.errors]
        assert error_indexes == [1, 3]

    def test_all_bad_batch_accepted_zero(self):
        batch = [{"broken": True}, {"also_broken": True}]
        result, repo = _ingest(batch)
        assert result.accepted == 0
        assert len(result.errors) == 2
        assert repo.count() == 0


# ── queue expansion ───────────────────────────────────────────────────────────

class TestQueueExpansion:
    def test_completed_queue_counts_as_two_accepted(self):
        result, repo = _ingest([SAMPLE_QUEUE_COMPLETED])
        assert result.accepted == 2
        assert result.total_received == 1
        assert repo.count() == 2

    def test_abandoned_queue_counts_as_two_accepted(self):
        result, repo = _ingest([SAMPLE_QUEUE_ABANDONED])
        assert result.accepted == 2
        assert repo.count() == 2

    def test_queue_expansion_idempotent(self):
        repo = InMemoryRepository()
        r1 = ingest_batch([SAMPLE_QUEUE_COMPLETED], repo)
        r2 = ingest_batch([SAMPLE_QUEUE_COMPLETED], repo)
        assert r1.accepted == 2
        assert r2.duplicates == 2
        assert repo.count() == 2

    def test_expanded_events_have_correct_types(self):
        _, repo = _ingest([SAMPLE_QUEUE_COMPLETED])
        types = {ev.event_type for ev in repo.all_events()}
        assert EventType.BILLING_QUEUE_JOIN in types
        assert EventType.BILLING_QUEUE_COMPLETE in types


# ── batch size limit ──────────────────────────────────────────────────────────

class TestBatchSizeLimit:
    def test_exactly_max_batch_accepted(self):
        batch = [SAMPLE_ENTRY.copy() for _ in range(_MAX_BATCH_SIZE)]
        # All have same natural keys → deduplicated after first, but no error raised
        result, _ = _ingest(batch)
        assert result.errors == []

    def test_over_limit_raises_value_error(self):
        batch = [SAMPLE_ENTRY] * (_MAX_BATCH_SIZE + 1)
        with pytest.raises(ValueError, match="limit"):
            ingest_batch(batch, InMemoryRepository())


# ── duplicate within batch ────────────────────────────────────────────────────

class TestDuplicateWithinBatch:
    def test_same_raw_event_twice_in_one_batch(self):
        # Both normalize to the same SYN- event_id
        result, repo = _ingest([SAMPLE_ENTRY, SAMPLE_ENTRY])
        assert result.accepted == 1
        assert result.duplicates == 1
        assert repo.count() == 1

    def test_different_events_same_batch_both_stored(self):
        result, _ = _ingest([SAMPLE_ENTRY, SAMPLE_EXIT])
        assert result.accepted == 2
        assert result.duplicates == 0


# ── HTTP smoke test ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHTTPRoute:
    @pytest_asyncio.fixture
    async def client(self):
        from app.main import app, _InMemoryRepository
        # ASGITransport does not fire lifespan events, so we seed app.state
        # directly. The smoke test verifies route wiring, not lifespan logic.
        app.state.repo = _InMemoryRepository()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c

    async def test_ingest_returns_200(self, client):
        payload = {"events": [SAMPLE_ENTRY]}
        resp = await client.post("/events/ingest", json=payload)
        assert resp.status_code == 200

    async def test_ingest_response_schema(self, client):
        payload = {"events": [SAMPLE_ENTRY, SAMPLE_EXIT]}
        resp = await client.post("/events/ingest", json=payload)
        body = resp.json()
        assert "accepted" in body
        assert "duplicates" in body
        assert "errors" in body
        assert "total_received" in body

    async def test_ingest_idempotent_via_http(self, client):
        payload = {"events": [SPEC_CANONICAL_ENTRY]}
        r1 = await client.post("/events/ingest", json=payload)
        r2 = await client.post("/events/ingest", json=payload)
        assert r1.json()["accepted"] == 1
        # Second call: same event_id → duplicate (in-memory repo persists within session)
        assert r2.json()["duplicates"] == 1

    async def test_empty_batch_via_http(self, client):
        resp = await client.post("/events/ingest", json={"events": []})
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 0

    async def test_health_endpoint(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
