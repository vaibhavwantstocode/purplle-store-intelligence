"""
ByteTrack wrapper for per-camera person tracking.

One StoreTracker instance per camera. Track IDs are scoped to the instance —
callers must namespace them as f"{camera_id}::{track_id}" when writing
canonical events to avoid cross-camera collisions.

ByteTrack two-stage matching summary:
  Stage A — high-confidence detections (conf >= high_thresh) matched against
             active + lost tracks via IoU + Kalman prediction.
  Stage B — detections in (low_thresh, high_thresh) matched against unmatched
             active tracks from stage A.
  New      — unmatched high-conf detections create tentative tracks.

sv.ByteTrack is deprecated in supervision 0.28 (→ removed in 0.30).  The
wrapper isolates the dependency to this file; swapping the underlying tracker
only requires changes here, not in callers.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import supervision as sv

from pipeline.types import Detection, Track, TrackState


class StoreTracker:
    """
    Per-camera ByteTrack adapter.

    Args:
        camera_id:              Identifies the camera; stored for caller use,
                                not used internally.
        fps:                    Frames per second of the source clip.  Used to
                                convert max_time_lost_seconds → frames.
        max_time_lost_seconds:  How long (wall-clock) a track survives without
                                a matching detection before it is removed.
        min_hits:               Consecutive frames a new detection must appear
                                before it is promoted to TRACKED state.
                                Note: due to ByteTrack internals, a brand-new
                                track first appears in update() output after
                                min_hits + 1 calls (except on the very first
                                call to update() where min_hits=1 activates
                                immediately).
        high_thresh:            Confidence threshold above which a detection
                                participates in first-stage matching.
        low_thresh:             Hard minimum confidence.  Detections below this
                                are discarded before tracking begins.
    """

    def __init__(
        self,
        camera_id: str,
        fps: float,
        max_time_lost_seconds: float = 1.0,
        min_hits: int = 3,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
    ) -> None:
        self.camera_id = camera_id
        self._fps = fps
        self._low_thresh = low_thresh

        # ByteTrack internally computes max_time_lost = int(frame_rate / 30 * buffer).
        # To produce exactly (max_time_lost_seconds * fps) frames of tolerance we need
        # buffer = max_time_lost_seconds * 30  (the 30fps-normalised equivalent).
        lost_buffer = max(1, math.ceil(max_time_lost_seconds * 30))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=high_thresh,
                lost_track_buffer=lost_buffer,
                minimum_matching_threshold=0.8,  # IoU threshold for matching
                frame_rate=int(fps),
                minimum_consecutive_frames=min_hits,
            )

        # Bookkeeping not exposed by supervision internals
        self._frame_count: dict[int, int] = {}  # track_id → times returned

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Process one frame's detections and return all TRACKED objects.

        Only TRACKED (matched-this-frame) objects are returned.  LOST tracks
        live inside ByteTrack's internal buffer; they will re-appear with the
        same track_id if re-detected within max_time_lost_seconds.

        Args:
            detections: Person bounding boxes from the detector for this frame.

        Returns:
            Tracks whose state is TRACKED.  Empty list if no active tracks.
        """
        filtered = [d for d in detections if d.confidence >= self._low_thresh]
        sv_dets = self._to_sv_detections(filtered)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            sv_result = self._tracker.update_with_detections(sv_dets)

        return self._to_tracks(sv_result)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_sv_detections(detections: list[Detection]) -> sv.Detections:
        if not detections:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty(0, dtype=np.float32),
                class_id=np.empty(0, dtype=int),
            )
        xyxy = np.array([list(d.bbox) for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def _to_tracks(self, sv_result: sv.Detections) -> list[Track]:
        if sv_result.tracker_id is None or len(sv_result) == 0:
            return []

        tracks: list[Track] = []
        for i in range(len(sv_result)):
            tid = int(sv_result.tracker_id[i])
            self._frame_count[tid] = self._frame_count.get(tid, 0) + 1
            x1, y1, x2, y2 = [float(v) for v in sv_result.xyxy[i]]
            conf = (
                float(sv_result.confidence[i])
                if sv_result.confidence is not None
                else 1.0
            )
            tracks.append(Track(
                track_id=tid,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                state=TrackState.TRACKED,
                frame_count=self._frame_count[tid],
                frames_since_update=0,
            ))
        return tracks
