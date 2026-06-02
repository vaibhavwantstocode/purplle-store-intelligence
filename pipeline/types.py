from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"   # not yet confirmed (below min_hits threshold)
    TRACKED = "TRACKED"       # actively matched this frame
    LOST = "LOST"             # unmatched but within max_time_lost window
    REMOVED = "REMOVED"       # exceeded window — garbage collected


@dataclass(frozen=True)
class Detection:
    """Single person bounding box from one video frame (YOLO output)."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    class_id: int = 0  # 0 = person


@dataclass(frozen=True)
class Track:
    """Tracked object with a stable ID across frames."""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    state: TrackState
    frame_count: int           # total frames this track has appeared in output
    frames_since_update: int   # frames since last matched to a detection (0 = matched this frame)

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.TRACKED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST
