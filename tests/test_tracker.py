"""
Tests for the Stage 2 tracking layer (pipeline/tracker.py).

All tests use synthetic Detection sequences — no video files, no YOLO,
no filesystem access.  A fresh StoreTracker is constructed per test.

ByteTrack API note (empirically verified against supervision 0.28):
  A new track appears in update() output after min_hits + 1 consecutive calls
  — EXCEPT on the very first call to a fresh tracker where min_hits=1, which
  activates immediately.  Tests are written to match this behaviour exactly.
"""

from __future__ import annotations

import pytest

from pipeline.tracker import StoreTracker
from pipeline.types import Detection, Track, TrackState


# ── helpers ───────────────────────────────────────────────────────────────────

def make_tracker(
    camera_id: str = "cam1",
    fps: float = 30.0,
    min_hits: int = 1,
    high_thresh: float = 0.6,
    low_thresh: float = 0.1,
    max_time_lost_seconds: float = 1.0,
) -> StoreTracker:
    return StoreTracker(
        camera_id=camera_id,
        fps=fps,
        min_hits=min_hits,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
        max_time_lost_seconds=max_time_lost_seconds,
    )


def det(
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 300.0,
    conf: float = 0.9,
) -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf)


def _run(tracker: StoreTracker, detection: Detection, n: int) -> list[list[Track]]:
    """Feed the same detection for n frames; returns per-frame track lists."""
    return [tracker.update([detection]) for _ in range(n)]


def _run_empty(tracker: StoreTracker, n: int) -> None:
    """Feed n empty frames."""
    for _ in range(n):
        tracker.update([])


# ── track creation ────────────────────────────────────────────────────────────

class TestTrackCreation:
    def test_single_detection_creates_track(self):
        # Fresh tracker, min_hits=1: first call is frame_id=1 — immediate activation
        tracker = make_tracker(min_hits=1)
        tracks = tracker.update([det()])
        assert len(tracks) == 1

    def test_track_has_positive_integer_id(self):
        tracker = make_tracker(min_hits=1)
        [track] = tracker.update([det()])
        assert isinstance(track.track_id, int)
        assert track.track_id > 0

    def test_track_bbox_close_to_detection(self):
        # Kalman filter on first frame should return the observation with minimal drift
        tracker = make_tracker(min_hits=1)
        d = det(x1=50.0, y1=60.0, x2=150.0, y2=280.0)
        [track] = tracker.update([d])
        assert abs(track.bbox[0] - 50.0) < 5
        assert abs(track.bbox[1] - 60.0) < 5
        assert abs(track.bbox[2] - 150.0) < 5
        assert abs(track.bbox[3] - 280.0) < 5

    def test_empty_frame_returns_no_tracks(self):
        tracker = make_tracker(min_hits=1)
        assert tracker.update([]) == []

    def test_track_state_is_tracked(self):
        tracker = make_tracker(min_hits=1)
        [track] = tracker.update([det()])
        assert track.state == TrackState.TRACKED
        assert track.is_confirmed

    def test_is_lost_false_for_matched_track(self):
        tracker = make_tracker(min_hits=1)
        [track] = tracker.update([det()])
        assert not track.is_lost

    def test_two_non_overlapping_detections_create_two_tracks(self):
        tracker = make_tracker(min_hits=1)
        d1 = det(x1=0,   y1=0, x2=100, y2=200)
        d2 = det(x1=500, y1=0, x2=600, y2=200)
        tracks = tracker.update([d1, d2])
        assert len(tracks) == 2

    def test_track_confidence_matches_detection(self):
        tracker = make_tracker(min_hits=1)
        [track] = tracker.update([det(conf=0.85)])
        assert abs(track.confidence - 0.85) < 0.02


# ── track persistence ─────────────────────────────────────────────────────────

class TestTrackPersistence:
    def test_same_object_same_id_across_frames(self):
        tracker = make_tracker(min_hits=1)
        ids: list[int] = []
        for i in range(10):
            tracks = tracker.update([det(x1=100 + i, y1=100, x2=200 + i, y2=300)])
            assert len(tracks) == 1
            ids.append(tracks[0].track_id)
        assert len(set(ids)) == 1

    def test_frame_count_increments_each_appearance(self):
        tracker = make_tracker(min_hits=1)
        for n in range(1, 6):
            tracks = tracker.update([det()])
            assert tracks[0].frame_count == n

    def test_frames_since_update_is_zero_when_matched(self):
        tracker = make_tracker(min_hits=1)
        for _ in range(5):
            [track] = tracker.update([det()])
        assert track.frames_since_update == 0

    def test_two_parallel_objects_keep_separate_ids(self):
        tracker = make_tracker(min_hits=1)
        d1 = det(x1=0,   y1=0, x2=100, y2=200)
        d2 = det(x1=500, y1=0, x2=600, y2=200)
        id_pairs: list[tuple[int, int]] = []
        for _ in range(8):
            tracks = tracker.update([d1, d2])
            assert len(tracks) == 2
            id_pairs.append(tuple(sorted(t.track_id for t in tracks)))
        assert len(set(id_pairs)) == 1

    def test_ids_do_not_swap_during_parallel_motion(self):
        """Two objects moving left→right in parallel should never exchange IDs."""
        tracker = make_tracker(min_hits=1)
        id_by_position: list[tuple[int, int]] = []
        for i in range(10):
            d1 = det(x1=i * 2,       y1=0, x2=i * 2 + 100, y2=200)
            d2 = det(x1=i * 2 + 500, y1=0, x2=i * 2 + 600, y2=200)
            tracks = tracker.update([d1, d2])
            assert len(tracks) == 2
            by_x = sorted(tracks, key=lambda t: t.bbox[0])
            id_by_position.append((by_x[0].track_id, by_x[1].track_id))
        left_ids  = {pair[0] for pair in id_by_position}
        right_ids = {pair[1] for pair in id_by_position}
        assert len(left_ids) == 1
        assert len(right_ids) == 1


# ── confidence thresholding ───────────────────────────────────────────────────

class TestConfidenceThresholding:
    def test_below_low_thresh_creates_no_track(self):
        tracker = make_tracker(min_hits=1, low_thresh=0.3)
        tracks = tracker.update([det(conf=0.2)])
        assert tracks == []

    def test_strictly_below_low_thresh_discarded(self):
        tracker = make_tracker(min_hits=1, low_thresh=0.5)
        tracks = tracker.update([det(conf=0.49)])
        assert tracks == []

    def test_at_low_thresh_accepted(self):
        # conf == low_thresh passes the >= filter
        tracker = make_tracker(min_hits=1, low_thresh=0.3, high_thresh=0.6)
        # Low-conf detection alone won't create a new track (second-stage only
        # matches existing tracks).  Seed a track first, then feed low-conf.
        tracker.update([det(conf=0.9)])
        tracks = tracker.update([det(conf=0.3)])
        assert len(tracks) == 1

    def test_min_hits_gates_tentative_tracks(self):
        # With min_hits=3: track appears on call 4 (verified empirically).
        # On call 1 (frame_id=1) the track is created but gets no external ID
        # until tracklet_len reaches min_hits via consecutive update() calls.
        tracker = make_tracker(min_hits=3, high_thresh=0.5)
        assert tracker.update([det(conf=0.8)]) == []  # call 1
        assert tracker.update([det(conf=0.8)]) == []  # call 2
        assert tracker.update([det(conf=0.8)]) == []  # call 3
        tracks = tracker.update([det(conf=0.8)])       # call 4: confirmed
        assert len(tracks) == 1

    def test_min_hits_1_activates_on_first_call(self):
        # Special case: fresh tracker, first-ever call, min_hits=1.
        tracker = make_tracker(min_hits=1, high_thresh=0.5)
        tracks = tracker.update([det(conf=0.9)])
        assert len(tracks) == 1
        assert tracks[0].state == TrackState.TRACKED

    def test_low_conf_second_stage_matches_existing_track(self):
        # Create a confirmed track, then feed a low-conf detection in its
        # position — ByteTrack second stage should match it.
        tracker = make_tracker(min_hits=1, high_thresh=0.6, low_thresh=0.1)
        original = tracker.update([det(conf=0.9)])
        original_id = original[0].track_id
        low_tracks = tracker.update([det(conf=0.15)])
        assert len(low_tracks) == 1
        assert low_tracks[0].track_id == original_id


# ── lost track handling ───────────────────────────────────────────────────────

class TestLostTrackHandling:
    def test_track_recovers_same_id_within_window(self):
        """Object disappears for a short time then reappears → same track_id."""
        tracker = make_tracker(min_hits=1, fps=30.0, max_time_lost_seconds=2.0)

        tracks = _run(tracker, det(), 5)
        original_id = tracks[-1][0].track_id

        _run_empty(tracker, 10)  # 10 frames ≪ 60-frame window

        recovered = tracker.update([det()])
        assert len(recovered) == 1
        assert recovered[0].track_id == original_id

    def test_track_gone_after_window_exceeded(self):
        """Object exceeds the lost window → ByteTrack removes the track."""
        fps = 30.0
        max_s = 0.5  # 15 frames
        tracker = make_tracker(min_hits=1, fps=fps, max_time_lost_seconds=max_s)

        _run(tracker, det(), 5)

        exceed_by = 5
        lost_frames = int(max_s * fps) + exceed_by  # 20 frames > 15
        _run_empty(tracker, lost_frames)

        # Reappear: track is expired; a brand-new TENTATIVE track is created.
        # With min_hits=1 on frame_id > 1, it needs one extra call to confirm.
        first_back = tracker.update([det()])
        second_back = tracker.update([det()])

        if first_back:
            # If it somehow activated on the first call, its frame_count is 1
            assert first_back[0].frame_count == 1
        else:
            # More commonly: confirmed on second call
            assert len(second_back) == 1
            assert second_back[0].frame_count == 1

    def test_new_id_assigned_after_expiry(self):
        """After expiry the reappearing object gets a higher track_id."""
        fps = 10.0
        tracker = make_tracker(min_hits=1, fps=fps, max_time_lost_seconds=0.5)

        _run(tracker, det(), 3)  # original id = 1

        _run_empty(tracker, 10)  # well past 5-frame window

        # Run until we get a confirmed track
        new_id = None
        for _ in range(5):
            tracks = tracker.update([det()])
            if tracks:
                new_id = tracks[0].track_id
                break

        assert new_id is not None
        assert new_id > 1  # ByteTrack never reuses IDs

    def test_longer_disappearance_still_recovers_if_within_window(self):
        # ByteTrack buffer formula: max_time_lost = int(fps / 30 * buffer).
        # With max_time_lost_seconds=3.0, buffer=90 → max_time_lost = 30 frames at fps=10.
        # We disappear for 20 frames (2.0 s) which is within the 30-frame window.
        tracker = make_tracker(min_hits=1, fps=10.0, max_time_lost_seconds=3.0)

        _run(tracker, det(), 5)
        original_id = tracker.update([det()])[0].track_id

        _run_empty(tracker, 20)  # 20 frames at 10fps = 2.0 s < 3.0 s window

        recovered = tracker.update([det()])
        assert len(recovered) == 1
        assert recovered[0].track_id == original_id


# ── multiple objects ──────────────────────────────────────────────────────────

class TestMultipleObjects:
    def test_distinct_ids_for_non_overlapping_detections(self):
        tracker = make_tracker(min_hits=1)
        d1 = det(x1=0,   y1=0, x2=100, y2=200)
        d2 = det(x1=500, y1=0, x2=600, y2=200)
        tracks = tracker.update([d1, d2])
        ids = [t.track_id for t in tracks]
        assert ids[0] != ids[1]

    def test_one_object_leaves_other_persists_as_tracked(self):
        tracker = make_tracker(min_hits=1)
        d1 = det(x1=0,   y1=0, x2=100, y2=200)
        d2 = det(x1=500, y1=0, x2=600, y2=200)

        for _ in range(5):
            tracker.update([d1, d2])

        # d1 leaves; only d2 remains
        for _ in range(5):
            tracks = tracker.update([d2])

        assert len(tracks) == 1
        assert tracks[0].state == TrackState.TRACKED

    def test_third_object_gets_incremented_id(self):
        # d1 and d2 activate on frame 1 (frame_id=1 special case).
        # d3 first appears on frame 2 (frame_id=2): it is TENTATIVE and not
        # returned.  It confirms on frame 3 when tracklet_len reaches min_hits=1.
        tracker = make_tracker(min_hits=1)
        d1 = det(x1=0,   y1=0, x2=100, y2=200)
        d2 = det(x1=500, y1=0, x2=600, y2=200)
        d3 = det(x1=250, y1=0, x2=350, y2=200)

        two_tracks = tracker.update([d1, d2])
        tracker.update([d1, d2, d3])       # d3 is tentative here
        three_tracks = tracker.update([d1, d2, d3])  # d3 confirmed

        ids_two   = {t.track_id for t in two_tracks}
        ids_three = {t.track_id for t in three_tracks}
        assert len(ids_three) == 3
        assert ids_two.issubset(ids_three)


# ── camera namespacing ────────────────────────────────────────────────────────

class TestCameraNamespacing:
    def test_two_trackers_independently_assign_id_1(self):
        """Each camera's tracker has its own ID counter starting at 1."""
        t1 = make_tracker(camera_id="cam1", min_hits=1)
        t2 = make_tracker(camera_id="cam2", min_hits=1)

        [tr1] = t1.update([det()])
        [tr2] = t2.update([det()])

        assert tr1.track_id == 1
        assert tr2.track_id == 1  # independent counters

    def test_camera_id_stored_on_tracker(self):
        tracker = make_tracker(camera_id="CAM_ENTRY_01")
        assert tracker.camera_id == "CAM_ENTRY_01"

    def test_namespaced_ids_are_globally_unique(self):
        t1 = make_tracker(camera_id="cam1", min_hits=1)
        t2 = make_tracker(camera_id="cam2", min_hits=1)

        [tr1] = t1.update([det()])
        [tr2] = t2.update([det()])

        ns1 = f"{t1.camera_id}::{tr1.track_id}"
        ns2 = f"{t2.camera_id}::{tr2.track_id}"
        assert ns1 != ns2
        assert ns1 == "cam1::1"
        assert ns2 == "cam2::1"

    def test_trackers_are_fully_independent(self):
        """Events on one camera's tracker do not affect the other."""
        t1 = make_tracker(camera_id="cam1", min_hits=1)
        t2 = make_tracker(camera_id="cam2", min_hits=1)

        # Run many frames on t1
        for _ in range(20):
            t1.update([det()])

        # t2 still starts fresh
        [tr] = t2.update([det()])
        assert tr.track_id == 1
        assert tr.frame_count == 1
