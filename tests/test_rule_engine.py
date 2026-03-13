"""
Unit tests for TrafficRuleEngine (src/symbolic_engine/rule_engine.py).

All tests use in-memory synthetic DataFrames — no database access needed.
"""

import pandas as pd
import pytest

from src.symbolic_engine.rule_engine import (
    TrafficRuleEngine,
    RuleViolation,
    SPEEDING_THRESHOLD_MS,
    HARD_BRAKING_THRESHOLD,
    AGGRESSIVE_ACCEL_THRESHOLD,
)
from tests.conftest import make_trajectory_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate(df: pd.DataFrame, track_id: int = 1):
    engine = TrafficRuleEngine()
    return engine.evaluate(df, track_id)


# ---------------------------------------------------------------------------
# Tests — basic structure
# ---------------------------------------------------------------------------

class TestRuleEngineInit:
    def test_can_instantiate(self):
        assert TrafficRuleEngine() is not None

    def test_evaluate_returns_list(self):
        df = make_trajectory_df(speed_ms=5.0)
        result = _evaluate(df)
        assert isinstance(result, list)

    def test_violation_is_dataclass(self):
        df = make_trajectory_df(speed_ms=20.0)
        violations = _evaluate(df)
        if violations:
            v = violations[0]
            assert hasattr(v, "rule_id")
            assert hasattr(v, "severity")
            assert hasattr(v, "evidence")
            assert hasattr(v, "track_id")


# ---------------------------------------------------------------------------
# Tests — no violation cases
# ---------------------------------------------------------------------------

class TestNoViolation:
    def test_slow_constant_speed_no_violation(self):
        df = make_trajectory_df(speed_ms=5.0, accel_ms2=0.0)
        assert _evaluate(df) == []

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=[
            "timestamp", "pos_x", "pos_y",
            "vel_x", "vel_y", "accel_x", "accel_y",
        ])
        assert _evaluate(df) == []

    def test_stopped_vehicle_no_violation(self):
        df = make_trajectory_df(speed_ms=0.0, accel_ms2=0.0)
        assert _evaluate(df) == []


# ---------------------------------------------------------------------------
# Tests — speeding
# ---------------------------------------------------------------------------

class TestSpeedingRule:
    def test_speeding_detected_above_threshold(self):
        df = make_trajectory_df(speed_ms=SPEEDING_THRESHOLD_MS + 2.0)
        violations = _evaluate(df)
        rule_ids = [v.rule_id for v in violations]
        assert "SPEEDING" in rule_ids

    def test_no_speeding_just_below_threshold(self):
        df = make_trajectory_df(speed_ms=SPEEDING_THRESHOLD_MS - 0.5)
        violations = _evaluate(df)
        rule_ids = [v.rule_id for v in violations]
        assert "SPEEDING" not in rule_ids

    def test_speeding_evidence_contains_speed(self):
        df = make_trajectory_df(speed_ms=20.0)
        violations = _evaluate(df)
        speeding = [v for v in violations if v.rule_id == "SPEEDING"]
        assert speeding, "Expected SPEEDING violation"
        assert "max_speed_ms" in speeding[0].evidence

    def test_speeding_severity_is_violation(self):
        df = make_trajectory_df(speed_ms=20.0)
        violations = _evaluate(df)
        speeding = [v for v in violations if v.rule_id == "SPEEDING"]
        assert speeding[0].severity == "violation"


# ---------------------------------------------------------------------------
# Tests — hard braking
# ---------------------------------------------------------------------------

class TestHardBrakingRule:
    def test_hard_braking_detected(self):
        df = make_trajectory_df(speed_ms=12.0, accel_ms2=-5.5)
        violations = _evaluate(df)
        rule_ids = [v.rule_id for v in violations]
        assert "HARD_BRAKING" in rule_ids

    def test_gentle_braking_no_violation(self):
        df = make_trajectory_df(speed_ms=10.0, accel_ms2=-1.0)
        violations = _evaluate(df)
        rule_ids = [v.rule_id for v in violations]
        assert "HARD_BRAKING" not in rule_ids

    def test_hard_braking_evidence_contains_accel(self):
        df = make_trajectory_df(speed_ms=12.0, accel_ms2=-5.5)
        violations = _evaluate(df)
        braking = [v for v in violations if v.rule_id == "HARD_BRAKING"]
        assert braking, "Expected HARD_BRAKING violation"
        assert "min_accel_y" in braking[0].evidence or "min_decel_ms2" in braking[0].evidence or braking[0].evidence


# ---------------------------------------------------------------------------
# Tests — aggressive acceleration
# ---------------------------------------------------------------------------

class TestAggressiveAccelRule:
    def test_aggressive_accel_detected(self):
        df = make_trajectory_df(speed_ms=0.0, accel_ms2=AGGRESSIVE_ACCEL_THRESHOLD + 0.5)
        violations = _evaluate(df)
        rule_ids = [v.rule_id for v in violations]
        assert "AGGRESSIVE_ACCELERATION" in rule_ids

    def test_normal_accel_no_violation(self):
        df = make_trajectory_df(speed_ms=3.0, accel_ms2=1.0)
        violations = _evaluate(df)
        rule_ids = [v.rule_id for v in violations]
        assert "AGGRESSIVE_ACCELERATION" not in rule_ids


# ---------------------------------------------------------------------------
# Tests — to_dict serialisability
# ---------------------------------------------------------------------------

class TestViolationSerialisation:
    def test_to_dict_is_json_serialisable(self):
        import json
        df = make_trajectory_df(speed_ms=20.0)
        violations = _evaluate(df)
        assert violations, "Expected at least one violation for serialisation test"
        payload = violations[0].to_dict()
        # Must not raise
        json.dumps(payload)

    def test_to_dict_has_required_keys(self):
        df = make_trajectory_df(speed_ms=20.0)
        violations = _evaluate(df)
        d = violations[0].to_dict()
        for key in ("rule_id", "description", "track_id", "severity", "evidence"):
            assert key in d, f"Missing key: {key}"
