"""
Unit tests for AlertEngine (src/symbolic_engine/alert_engine.py).

Tests verify that each alert type fires at the correct thresholds and that
the cooldown mechanism prevents duplicate alerts.
No GPU or video file required.
"""

import pytest
from typing import List

from src.symbolic_engine.alert_engine import (
    AlertEngine,
    TrafficAlert,
    SPEEDING_THRESHOLD_MS,
    HARD_BRAKING_THRESHOLD,
    AGGRESSIVE_ACCEL_THRESHOLD,
    PROXIMITY_WARNING_M,
    COLLISION_THRESHOLD_M,
)


# ---------------------------------------------------------------------------
# Helper — collect alerts fired into a list
# ---------------------------------------------------------------------------

def _make_engine(cooldown: float = 0.0) -> tuple[AlertEngine, List[TrafficAlert]]:
    """Return (engine, captured_alerts). cooldown=0 so all alerts fire in tests."""
    fired: List[TrafficAlert] = []
    engine = AlertEngine(on_alert=fired.append, cooldown_secs=cooldown)
    return engine, fired


# ---------------------------------------------------------------------------
# Tests — speeding
# ---------------------------------------------------------------------------

class TestSpeedingAlert:
    def test_fires_above_threshold(self):
        engine, fired = _make_engine()
        state = {1: [0.0, 0.0, SPEEDING_THRESHOLD_MS + 2.0, 0.0, 0.0, 0.0]}
        engine.check(state, {1: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        types = [a.alert_type for a in fired]
        assert "SPEEDING" in types

    def test_no_alert_below_threshold(self):
        engine, fired = _make_engine()
        state = {1: [0.0, 0.0, SPEEDING_THRESHOLD_MS - 1.0, 0.0, 0.0, 0.0]}
        engine.check(state, {1: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        assert not any(a.alert_type == "SPEEDING" for a in fired)

    def test_alert_contains_track_id(self):
        engine, fired = _make_engine()
        state = {7: [0.0, 0.0, SPEEDING_THRESHOLD_MS + 5.0, 0.0, 0.0, 0.0]}
        engine.check(state, {7: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        speeding = [a for a in fired if a.alert_type == "SPEEDING"]
        assert speeding[0].track_id == 7

    def test_severity_is_critical(self):
        engine, fired = _make_engine()
        state = {1: [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]}
        engine.check(state, {1: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        speeding = [a for a in fired if a.alert_type == "SPEEDING"]
        assert speeding[0].severity == "critical"


# ---------------------------------------------------------------------------
# Tests — hard braking
# ---------------------------------------------------------------------------

class TestHardBrakingAlert:
    def test_fires_on_strong_deceleration(self):
        engine, fired = _make_engine()
        # Moving in Y direction, strong negative Y acceleration = braking
        state = {2: [0.0, 0.0, 0.0, 10.0, 0.0, HARD_BRAKING_THRESHOLD - 1.0]}
        engine.check(state, {2: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        assert any(a.alert_type == "HARD_BRAKING" for a in fired)

    def test_no_alert_gentle_braking(self):
        engine, fired = _make_engine()
        state = {2: [0.0, 0.0, 0.0, 5.0, 0.0, -1.0]}
        engine.check(state, {2: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        assert not any(a.alert_type == "HARD_BRAKING" for a in fired)


# ---------------------------------------------------------------------------
# Tests — aggressive acceleration
# ---------------------------------------------------------------------------

class TestAggressiveAccelAlert:
    def test_fires_on_high_accel_magnitude(self):
        engine, fired = _make_engine()
        state = {3: [0.0, 0.0, 0.0, 0.0, AGGRESSIVE_ACCEL_THRESHOLD + 1.0, 0.0]}
        engine.check(state, {3: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        assert any(a.alert_type == "AGGRESSIVE_ACCEL" for a in fired)

    def test_no_alert_moderate_accel(self):
        engine, fired = _make_engine()
        state = {3: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]}
        engine.check(state, {3: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        assert not any(a.alert_type == "AGGRESSIVE_ACCEL" for a in fired)


# ---------------------------------------------------------------------------
# Tests — proximity warning
# ---------------------------------------------------------------------------

class TestProximityAlert:
    def test_fires_when_vehicles_are_close_and_closing(self):
        engine, fired = _make_engine()
        # Two vehicles 1.5 m apart, moving toward each other
        state = {
            1: [0.0, 0.0,  3.0, 0.0, 0.0, 0.0],
            2: [1.5, 0.0, -3.0, 0.0, 0.0, 0.0],
        }
        real_coords = {1: (0.0, 0.0), 2: (1.5, 0.0)}
        engine.check(state, real_coords, timestamp=1.0, frame_id=30)
        proximity_types = [a.alert_type for a in fired]
        # Either PROXIMITY_WARNING or COLLISION_SUSPECTED
        assert any(t in proximity_types for t in ("PROXIMITY_WARNING", "COLLISION_SUSPECTED"))

    def test_no_alert_when_vehicles_far_apart(self):
        engine, fired = _make_engine()
        state = {
            1: [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            2: [50.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        }
        real_coords = {1: (0.0, 0.0), 2: (50.0, 0.0)}
        engine.check(state, real_coords, timestamp=1.0, frame_id=30)
        assert not any(a.alert_type in ("PROXIMITY_WARNING", "COLLISION_SUSPECTED") for a in fired)

    def test_single_vehicle_no_proximity_alert(self):
        engine, fired = _make_engine()
        state = {1: [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]}
        engine.check(state, {1: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        assert not any(a.alert_type in ("PROXIMITY_WARNING", "COLLISION_SUSPECTED") for a in fired)


# ---------------------------------------------------------------------------
# Tests — cooldown mechanism
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_cooldown_suppresses_duplicate_alert(self):
        engine, fired = _make_engine(cooldown=5.0)
        state = {1: [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]}
        # First check at t=1.0 — should fire
        engine.check(state, {1: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        # Second check at t=2.0 — should be suppressed (cooldown=5s)
        engine.check(state, {1: (0.0, 0.0)}, timestamp=2.0, frame_id=60)
        speeding = [a for a in fired if a.alert_type == "SPEEDING"]
        assert len(speeding) == 1, "Duplicate alert should have been suppressed"

    def test_cooldown_expires_and_refires(self):
        engine, fired = _make_engine(cooldown=2.0)
        state = {1: [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]}
        engine.check(state, {1: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        # After cooldown expires
        engine.check(state, {1: (0.0, 0.0)}, timestamp=4.0, frame_id=120)
        speeding = [a for a in fired if a.alert_type == "SPEEDING"]
        assert len(speeding) == 2, "Alert should re-fire after cooldown expires"


# ---------------------------------------------------------------------------
# Tests — alert serialisation
# ---------------------------------------------------------------------------

class TestAlertSerialisation:
    def test_to_dict_is_json_serialisable(self):
        import json
        engine, fired = _make_engine()
        state = {1: [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]}
        engine.check(state, {1: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        assert fired, "Expected at least one alert"
        payload = fired[0].to_dict()
        json.dumps(payload)  # Must not raise

    def test_to_dict_has_required_keys(self):
        engine, fired = _make_engine()
        state = {1: [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]}
        engine.check(state, {1: (0.0, 0.0)}, timestamp=1.0, frame_id=30)
        assert fired
        d = fired[0].to_dict()
        for key in ("alert_type", "severity", "track_id", "timestamp", "message", "evidence"):
            assert key in d, f"Missing key in alert dict: {key}"
