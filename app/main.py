"""
FastAPI entrypoint — the only file that is framework-aware.

Stage 1 ships with an in-memory repository so the route works end-to-end
without a database. Stage 2 replaces _InMemoryRepository with SQLiteRepository
inside the lifespan; nothing else in this file changes.

Endpoints implemented so far:
  POST /events/ingest   ← Stage 1
  GET  /health          ← stub (accurate implementation in Stage 4)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.ingestion import ingest_batch, _MAX_BATCH_SIZE
from app.models import CanonicalEvent, IngestResponse
from app.repository import EventRepository


# ── temporary in-memory repository (replaced by SQLite in Stage 2) ────────────

class _InMemoryRepository:
    def __init__(self) -> None:
        self._store: dict[str, CanonicalEvent] = {}

    def exists(self, event_id: str) -> bool:
        return event_id in self._store

    def save(self, event: CanonicalEvent) -> None:
        self._store[event.event_id] = event


# ── app factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Stage 2: swap _InMemoryRepository() for SQLiteRepository(settings.db_path)
    app.state.repo = _InMemoryRepository()
    yield


app = FastAPI(
    title="Store Intelligence API",
    version="0.1.0",
    description="Apex Retail offline store analytics — Purplle Tech Challenge",
    lifespan=lifespan,
)


def _get_repo(request: Request) -> EventRepository:
    return request.app.state.repo


# ── request schema (only used here, so defined here not in models.py) ─────────

class IngestRequest(BaseModel):
    events: list[dict[str, Any]]


# ── routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/events/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a batch of store events (up to 500)",
)
async def ingest(
    body: IngestRequest,
    repo: EventRepository = Depends(_get_repo),
) -> IngestResponse:
    try:
        return ingest_batch(body.events, repo)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(exc)},
        )


@app.get(
    "/health",
    summary="Service liveness + feed staleness check (stub — full impl in Stage 4)",
)
async def health() -> dict[str, Any]:
    # Stage 4 adds: last_event_per_store, STALE_FEED warning, db connectivity check
    return {"status": "ok", "version": app.version}
