"""
Canonical event model — single source of truth shared by the API and the pipeline.
Both app/normalize.py (consumer) and pipeline/emit.py (producer) import from here,
so the API can never ingest something the pipeline couldn't have emitted.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    # Collapsed queue_completed records expand to this internal type.
    # Separate from ABANDON so metrics can distinguish served vs walked-away.
    BILLING_QUEUE_COMPLETE = "BILLING_QUEUE_COMPLETE"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    """
    Typed fields for known metadata. extra="allow" preserves any unknown keys
    so new pipeline fields never cause an ingestion failure.
    """
    model_config = {"extra": "allow"}

    # Spec canonical
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None

    # Group handling (sample entry/exit dialect)
    group_id: Optional[str] = None
    group_size: Optional[int] = None

    # Demographics (both dialects)
    gender: Optional[str] = None
    age: Optional[int] = None
    age_bucket: Optional[str] = None
    is_face_hidden: Optional[bool] = None

    # Queue lifecycle (sample queue dialect)
    queue_position_at_join: Optional[int] = None
    wait_seconds: Optional[int] = None

    # Zone enrichment (sample zone dialect)
    zone_name: Optional[str] = None
    zone_type: Optional[str] = None
    is_revenue_zone: Optional[bool] = None
    zone_hotspot_x: Optional[float] = None
    zone_hotspot_y: Optional[float] = None


class CanonicalEvent(BaseModel):
    """
    Internal canonical event. Every adapter must produce this.
    Persisted verbatim; all analytics derive from this table.
    """
    event_id: str                        # globally unique; synthesized if absent in input
    store_id: str                        # canonical ID after alias resolution
    camera_id: str
    visitor_id: str                      # per-session Re-ID token
    event_type: EventType
    timestamp: datetime                  # UTC always
    zone_id: Optional[str] = None        # None for ENTRY/EXIT
    dwell_ms: int = 0                    # 0 for instantaneous events
    is_staff: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: EventMetadata = Field(default_factory=EventMetadata)


# ── API request / response models ────────────────────────────────────────────

class IngestError(BaseModel):
    index: int    # position of the raw event in the submitted batch
    error: str


class IngestResponse(BaseModel):
    accepted: int       # canonical events written to storage (may be > total_received for queue expansion)
    duplicates: int     # events skipped because event_id already existed
    errors: list[IngestError]
    total_received: int # raw event dicts in the submitted batch
