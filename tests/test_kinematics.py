"""
Unit tests for KinematicEstimator (src/physics_engine/kinematics.py).

Tests use synthetic detection boxes — no GPU, camera, or video file needed.
"""

import numpy as np
import pytest

from src.physics_engine.kinematics import KinematicEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box(track_id: int, x: float, y: float, w: float = 50, h: float = 40) -> list:
    """Return a fake tracked box [x1, y1, x2, y2, track_id, conf, cls]."""
    return [x, y, x + w, y + h, track_id, 0.9, 0]


def _feed_constant_motion(
    est: KinematicEstimator,
    track_id: int,
    n_frames: int,
    vx_px: float = 10.0,
    vy_px: float = 0.0,
) -> None:
    """
    Feed `n_frames` of a single track moving at constant pixel velocity.
    Bottom-centre position used as the 2-D point.
    """
    for f in range(n_frames):
        x = float(f * vx_px)
        y = float(f * vy_px)
        boxes = np.array([[_make_box(track_id, x, y)]], dtype=object)
        # KinematicEstimator.update expects list of dicts {track_id: [x, y, vx, vy, ax, ay]}
        # but internally it works from the raw (x, y) position history.
        # We use the public update() interface.
        est.update({track_id: [x, y, 0, 0, 0, 0]})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKinematicEstimatorInit:
    def test_creates_estimator(self):
        est = KinematicEstimator(fps=30.0)
        assert est is not None

    def test_warm_tracks_empty_at_start(self):
        est = KinematicEstimator(fps=30.0)
        assert len(est.warm_tracks) == 0


class TestKinematicEstimatorUpdate:
    def test_update_returns_dict(self):
        est = KinematicEstimator(fps=30.0)
        result = est.update({1: (0.0, 0.0)})
        # update() returns state_vectors dict
        assert isinstance(result, dict)

    def test_track_becomes_warm_after_window(self):
        est = KinematicEstimator(fps=30.0)
        # Feed enough frames to fill the Savitzky-Golay window (default=15)
        for f in range(20):
            est.update({1: (float(f), 0.0)})
        assert 1 in est.warm_tracks

    def test_new_track_not_warm_yet(self):
        est = KinematicEstimator(fps=30.0)
        est.update({1: (0.0, 0.0)})
        assert 1 not in est.warm_tracks

    def test_multiple_tracks_independent(self):
        est = KinematicEstimator(fps=30.0)
        for f in range(20):
            est.update({
                1: (float(f),       0.0),
                2: (float(f) * 2.0, 0.0),
            })
        assert 1 in est.warm_tracks
        assert 2 in est.warm_tracks

    def test_velocity_direction_correct(self):
        """A track moving right (increasing x) should have positive vel_x."""
        est = KinematicEstimator(fps=30.0)
        for f in range(25):
            est.update({1: (float(f * 5), 0.0)})
        result = est.update({1: (float(25 * 5), 0.0)})
        assert 1 in result, "Track 1 should be in result"
        vx = result[1][2]
        assert vx > 0, f"Expected positive vel_x, got {vx}"

    def test_stale_track_pruned(self):
        """A track absent for more than MAX_MISSED_FRAMES should be removed."""
        est = KinematicEstimator(fps=30.0)
        # Feed track 1 for 20 frames
        for f in range(20):
            est.update({1: (float(f), 0.0)})
        # Now stop feeding track 1; feed only track 2 for many frames
        for f in range(20):
            est.update({2: (float(f), 0.0)})
        # Track 1 should have been pruned (missed > MAX_MISSED_FRAMES=5)
        assert 1 not in est.warm_tracks

    def test_empty_update_returns_empty(self):
        est = KinematicEstimator(fps=30.0)
        result = est.update({})
        assert result == {}

    def test_state_vector_shape(self):
        """Warm track output must have 6 elements: [x, y, vx, vy, ax, ay]."""
        est = KinematicEstimator(fps=30.0)
        for f in range(25):
            est.update({1: (float(f), 0.0)})
        result = est.update({1: (25.0, 0.0)})
        assert 1 in result, "Track 1 should be in result after warm-up"
        assert len(result[1]) == 6

    def test_gap_interpolation_keeps_track_alive(self):
        """Track reappearing within MAX_MISSED_FRAMES should stay warm."""
        from src.physics_engine.kinematics import MAX_MISSED_FRAMES
        est = KinematicEstimator(fps=30.0)
        for f in range(20):
            est.update({1: (float(f), 0.0)})
        # Miss exactly MAX_MISSED_FRAMES frames (track 2 only)
        for _ in range(MAX_MISSED_FRAMES):
            est.update({2: (0.0, 0.0)})
        # Reappear — track should still be warm and trajectory intact
        result = est.update({1: (25.0, 0.0)})
        assert 1 in result, "Track should survive a short gap"
        assert 1 in est.warm_tracks

    def test_collision_suspected_alert_fires(self):
        """Two vehicles < 2 m apart should trigger COLLISION_SUSPECTED."""
        from src.symbolic_engine.alert_engine import AlertEngine, COLLISION_THRESHOLD_M
        fired = []
        engine = AlertEngine(on_alert=fired.append, cooldown_secs=0.0)
        state = {
            1: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            2: [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        }
        real_coords = {1: (0.0, 0.0), 2: (COLLISION_THRESHOLD_M - 0.5, 0.0)}
        engine.check(state, real_coords, timestamp=1.0, frame_id=30)
        assert any(a.alert_type == "COLLISION_SUSPECTED" for a in fired)
