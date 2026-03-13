"""
Symbolic Traffic Rule Engine
============================

This is the **symbolic reasoning layer** of the Neuro-Symbolic pipeline.

Role in the Architecture
------------------------
Neural components (YOLO, ByteTrack, Savitzky-Golay, VLM) produce continuous,
uncertain observations.  This module converts those observations into discrete,
auditable symbolic verdicts by applying explicit, parameterised traffic safety
rules.

Each rule is a deterministic function over a vehicle's kinematic time-series
(DuckDB). The output is a list of ``RuleViolation`` dataclass instances — one
per triggered rule — each carrying the rule identifier, severity level, and
the exact evidence values that caused it to fire.

This separation is fundamental to Neuro-Symbolic AI:
  - Neural side  → perceives the world (noisy, probabilistic)
  - Symbolic side → judges the world (deterministic, explainable)

Rules
-----
SPEEDING                — sustained speed above the urban threshold (50 km/h)
HARD_BRAKING            — deceleration below −4.0 m/s²
AGGRESSIVE_ACCELERATION — acceleration magnitude above 3.5 m/s²

Thresholds are module-level constants and can be adjusted to match local
speed limits or road types without touching any rule logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Rule thresholds — all SI units (m/s, m/s²)
# ---------------------------------------------------------------------------

SPEEDING_THRESHOLD_MS: float = 13.89        # 50 km/h in m/s
HARD_BRAKING_THRESHOLD: float = -4.0        # m/s²  (negative = deceleration)
AGGRESSIVE_ACCEL_THRESHOLD: float = 3.5     # m/s²  (magnitude)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class RuleViolation:
    """
    A single fired traffic safety rule with full evidence for auditability.

    Attributes:
        rule_id:             Machine-readable rule name (e.g. 'SPEEDING').
        description:         Human-readable summary of what was detected.
        track_id:            Integer track identifier of the offending vehicle.
        first_occurrence_s:  Timestamp (seconds) when the rule first fired.
        severity:            'warning' for hazardous behaviour; 'violation' for
                             a clear breach of a traffic standard.
        evidence:            Raw numeric values that caused the rule to fire.
                             Provides a complete audit trail.
    """

    rule_id: str
    description: str
    track_id: int
    first_occurrence_s: float
    severity: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON output."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TrafficRuleEngine:
    """
    Applies deterministic symbolic traffic rules to a vehicle's kinematic
    time-series and returns a list of RuleViolation instances.

    This is the symbolic side of the Neuro-Symbolic pipeline: rules are
    explicit, interpretable, and completely independent of any neural model.
    Adding a new rule means adding a new ``_check_*`` method and calling it
    inside ``evaluate()``.
    """

    def evaluate(self, df: pd.DataFrame, track_id: int) -> List[RuleViolation]:
        """
        Evaluate all rules against the trajectory DataFrame for one vehicle.

        Args:
            df:       DataFrame from ``DuckDBClient.get_trajectory_window()``.
                      Required columns: timestamp, vel_x, vel_y, accel_x, accel_y.
            track_id: The vehicle's integer track identifier.

        Returns:
            List of RuleViolation instances (empty list if no rules fire).
        """
        if df.empty:
            return []

        # Pre-compute derived columns used by multiple rules.
        df = df.copy()
        df["speed"] = (df["vel_x"] ** 2 + df["vel_y"] ** 2) ** 0.5
        df["accel_mag"] = (df["accel_x"] ** 2 + df["accel_y"] ** 2) ** 0.5
        # Fix 6: direction-aware signed acceleration — same formula as AlertEngine.
        # Uses dot(velocity, acceleration) to determine braking vs accelerating,
        # so the rule fires correctly for vehicles braking at any heading angle.
        _dot = df["vel_x"] * df["accel_x"] + df["vel_y"] * df["accel_y"]
        df["signed_accel"] = df["accel_mag"].where(_dot >= 0, -df["accel_mag"])

        violations: List[RuleViolation] = []
        violations += self._check_speeding(df, track_id)
        violations += self._check_hard_braking(df, track_id)
        violations += self._check_aggressive_acceleration(df, track_id)
        return violations

    # ------------------------------------------------------------------
    # Individual rule checkers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_speeding(df: pd.DataFrame, track_id: int) -> List[RuleViolation]:
        """SPEEDING — any frame where speed exceeds the urban limit."""
        over = df[df["speed"] > SPEEDING_THRESHOLD_MS]
        if over.empty:
            return []

        max_speed = float(df["speed"].max())
        duration_s = round(
            float(len(over)) / max(float(len(df)), 1)
            * (float(df["timestamp"].max()) - float(df["timestamp"].min())),
            2,
        )
        return [RuleViolation(
            rule_id="SPEEDING",
            description=(
                f"Vehicle {track_id} exceeded {SPEEDING_THRESHOLD_MS * 3.6:.0f} km/h — "
                f"peak {max_speed * 3.6:.1f} km/h"
            ),
            track_id=track_id,
            first_occurrence_s=float(over["timestamp"].iloc[0]),
            severity="violation",
            evidence={
                "max_speed_ms": round(max_speed, 2),
                "max_speed_kmh": round(max_speed * 3.6, 1),
                "threshold_kmh": round(SPEEDING_THRESHOLD_MS * 3.6, 0),
                "duration_over_limit_s": duration_s,
            },
        )]

    @staticmethod
    def _check_hard_braking(df: pd.DataFrame, track_id: int) -> List[RuleViolation]:
        """
        HARD_BRAKING — signed deceleration below the threshold.
        Fix 6: uses direction-aware signed_accel (pre-computed in evaluate()) so
        the rule fires for vehicles braking at any heading, not just Y-axis motion.
        """
        braking = df[df["signed_accel"] < HARD_BRAKING_THRESHOLD]
        if braking.empty:
            return []

        min_accel = float(df["signed_accel"].min())
        return [RuleViolation(
            rule_id="HARD_BRAKING",
            description=(
                f"Vehicle {track_id} hard braking — "
                f"peak deceleration {abs(min_accel):.2f} m/s²"
            ),
            track_id=track_id,
            first_occurrence_s=float(braking["timestamp"].iloc[0]),
            severity="warning",
            evidence={
                "peak_deceleration_ms2": round(abs(min_accel), 2),
                "threshold_ms2": abs(HARD_BRAKING_THRESHOLD),
                "frames_in_hard_braking": int(len(braking)),
            },
        )]

    @staticmethod
    def _check_aggressive_acceleration(
        df: pd.DataFrame, track_id: int
    ) -> List[RuleViolation]:
        """AGGRESSIVE_ACCELERATION — acceleration magnitude exceeds threshold."""
        aggressive = df[df["accel_mag"] > AGGRESSIVE_ACCEL_THRESHOLD]
        if aggressive.empty:
            return []

        max_accel = float(df["accel_mag"].max())
        return [RuleViolation(
            rule_id="AGGRESSIVE_ACCELERATION",
            description=(
                f"Vehicle {track_id} aggressive acceleration — "
                f"peak {max_accel:.2f} m/s²"
            ),
            track_id=track_id,
            first_occurrence_s=float(aggressive["timestamp"].iloc[0]),
            severity="warning",
            evidence={
                "peak_accel_ms2": round(max_accel, 2),
                "threshold_ms2": AGGRESSIVE_ACCEL_THRESHOLD,
                "frames_flagged": int(len(aggressive)),
            },
        )]
