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
SPEEDING                — sustained speed above the posted limit (default 50 km/h)
HARD_BRAKING            — deceleration below −4.0 m/s²
AGGRESSIVE_ACCELERATION — acceleration magnitude above 3.5 m/s²
WRONG_WAY               — vehicle heading > 90° from expected flow direction
STATIONARY_IN_LANE      — vehicle stopped (< 0.5 m/s) for > 10 s in a live lane

Thresholds are module-level constants and can be adjusted to match local
speed limits or road types without touching any rule logic.
Per-zone speed limit and flow direction are passed as optional parameters
to ``evaluate()`` and override the module-level defaults automatically.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Rule thresholds — all SI units (m/s, m/s²)
# ---------------------------------------------------------------------------

SPEEDING_THRESHOLD_MS: float = 13.89        # 50 km/h in m/s  (overridden per-zone)
HARD_BRAKING_THRESHOLD: float = -4.0        # m/s²  (negative = deceleration)
AGGRESSIVE_ACCEL_THRESHOLD: float = 3.5     # m/s²  (magnitude)
WRONG_WAY_ANGLE_DEG: float = 90.0          # heading must differ > this to fire
WRONG_WAY_MIN_SPEED_MS: float = 1.0        # ignore stationary/very-slow vehicles
WRONG_WAY_MIN_DURATION_S: float = 1.0      # require sustained wrong-way before firing
STATIONARY_MIN_DURATION_S: float = 10.0    # seconds stopped to trigger the rule
STATIONARY_SPEED_THRESHOLD_MS: float = 0.5 # m/s below which vehicle is "stopped"


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
        severity_score:      Continuous 0.0–1.0 score combining magnitude and
                             duration. 1.0 = severe sustained violation.
                             Enables ranking violations by seriousness.
        evidence:            Raw numeric values that caused the rule to fire.
                             Provides a complete audit trail.
        assumptions:         Defaults or fallbacks used during evaluation,
                             e.g. speed limit sourced from global default.
    """

    rule_id: str
    description: str
    track_id: int
    first_occurrence_s: float
    severity: str
    severity_score: float = 0.0
    evidence: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)

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

    def evaluate(
        self,
        df: pd.DataFrame,
        track_id: int,
        speed_limit_ms: Optional[float] = None,
        flow_direction_deg: Optional[float] = None,
    ) -> List[RuleViolation]:
        """
        Evaluate all rules against the trajectory DataFrame for one vehicle.

        Args:
            df:                DataFrame from ``DuckDBClient.get_trajectory_window()``.
                               Required columns: timestamp, vel_x, vel_y, accel_x, accel_y.
            track_id:          The vehicle's integer track identifier.
            speed_limit_ms:    Per-zone posted speed limit in m/s.  When provided,
                               overrides the module-level ``SPEEDING_THRESHOLD_MS``.
                               Pass ``zone_config.speed_limit_kmh / 3.6``.
            flow_direction_deg: Expected travel direction (degrees, 0=East 90=North CCW).
                               When provided, enables the WRONG_WAY rule.

        Returns:
            List of RuleViolation instances (empty list if no rules fire).
        """
        if df.empty:
            return []

        # Pre-compute derived columns used by multiple rules.
        df = df.copy()
        df["speed"] = (df["vel_x"] ** 2 + df["vel_y"] ** 2) ** 0.5
        df["accel_mag"] = (df["accel_x"] ** 2 + df["accel_y"] ** 2) ** 0.5
        # Direction-aware signed acceleration: dot(velocity, acceleration) determines
        # braking vs accelerating, so the rule fires correctly at any heading angle.
        _dot = df["vel_x"] * df["accel_x"] + df["vel_y"] * df["accel_y"]
        df["signed_accel"] = df["accel_mag"].where(_dot >= 0, -df["accel_mag"])

        # Build assumption log — every default or fallback used in this evaluation.
        assumptions: List[str] = []
        if speed_limit_ms is None:
            assumptions.append(
                f"speed_limit_kmh defaulted to {SPEEDING_THRESHOLD_MS * 3.6:.0f} km/h "
                "(zone_config.json not found or speed_limit_kmh not set)"
            )
        if flow_direction_deg is None:
            assumptions.append(
                "WRONG_WAY rule disabled: flow_direction_deg not set in zone_config.json"
            )

        violations: List[RuleViolation] = []
        violations += self._check_speeding(df, track_id, speed_limit_ms, assumptions)
        violations += self._check_hard_braking(df, track_id, assumptions)
        violations += self._check_aggressive_acceleration(df, track_id, assumptions)
        violations += self._check_stationary_in_lane(df, track_id, assumptions)
        if flow_direction_deg is not None:
            violations += self._check_wrong_way(df, track_id, flow_direction_deg, assumptions)
        return violations

    # ------------------------------------------------------------------
    # Individual rule checkers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_speeding(
        df: pd.DataFrame,
        track_id: int,
        speed_limit_ms: Optional[float] = None,
        assumptions: Optional[list] = None,
    ) -> List[RuleViolation]:
        """SPEEDING — any frame where speed exceeds the posted limit."""
        threshold = speed_limit_ms if speed_limit_ms is not None else SPEEDING_THRESHOLD_MS
        over = df[df["speed"] > threshold]
        if over.empty:
            return []

        max_speed    = float(df["speed"].max())
        total_secs   = float(df["timestamp"].max()) - float(df["timestamp"].min())
        duration_s   = round(float(len(over)) / max(float(len(df)), 1) * total_secs, 2)
        pct_over     = round(100.0 * len(over) / max(len(df), 1), 1)
        excess_ms    = max_speed - threshold

        # severity_score: 0→1 based on excess speed (normalised to limit) × time fraction
        severity_score = round(
            min(1.0, excess_ms / max(threshold, 0.1)) * min(1.0, duration_s / 10.0),
            3,
        )

        return [RuleViolation(
            rule_id="SPEEDING",
            description=(
                f"Vehicle {track_id} exceeded {threshold * 3.6:.0f} km/h speed limit — "
                f"peak {max_speed * 3.6:.1f} km/h for {duration_s:.1f}s "
                f"({pct_over:.0f}% of window)"
            ),
            track_id=track_id,
            first_occurrence_s=float(over["timestamp"].iloc[0]),
            severity="violation",
            severity_score=severity_score,
            evidence={
                "max_speed_ms": round(max_speed, 2),
                "max_speed_kmh": round(max_speed * 3.6, 1),
                "speed_limit_kmh": round(threshold * 3.6, 0),
                "duration_over_limit_s": duration_s,
                "pct_time_over_limit": pct_over,
                "excess_speed_kmh": round(excess_ms * 3.6, 1),
            },
            assumptions=list(assumptions or []),
        )]

    @staticmethod
    def _check_hard_braking(
        df: pd.DataFrame,
        track_id: int,
        assumptions: Optional[list] = None,
    ) -> List[RuleViolation]:
        """
        HARD_BRAKING — signed deceleration below the threshold.
        Uses direction-aware signed_accel (pre-computed in evaluate()) so
        the rule fires for vehicles braking at any heading, not just Y-axis motion.
        """
        braking = df[df["signed_accel"] < HARD_BRAKING_THRESHOLD]
        if braking.empty:
            return []

        min_accel = float(df["signed_accel"].min())

        # Longest consecutive braking run (frames) — distinguishes one spike vs sustained
        consecutive = 1
        worst_consecutive = 1
        prev_idx = None
        for idx in braking.index:
            if prev_idx is not None and idx == prev_idx + 1:
                consecutive += 1
                worst_consecutive = max(worst_consecutive, consecutive)
            else:
                consecutive = 1
            prev_idx = idx

        # severity_score: magnitude above threshold × worst_consecutive (normalised to 10 frames)
        excess = abs(min_accel) - abs(HARD_BRAKING_THRESHOLD)
        severity_score = round(
            min(1.0, excess / 4.0) * min(1.0, worst_consecutive / 10.0),
            3,
        )

        return [RuleViolation(
            rule_id="HARD_BRAKING",
            description=(
                f"Vehicle {track_id} hard braking — "
                f"peak deceleration {abs(min_accel):.2f} m/s² "
                f"(worst run: {worst_consecutive} frames)"
            ),
            track_id=track_id,
            first_occurrence_s=float(braking["timestamp"].iloc[0]),
            severity="warning",
            severity_score=severity_score,
            evidence={
                "peak_deceleration_ms2": round(abs(min_accel), 2),
                "threshold_ms2": abs(HARD_BRAKING_THRESHOLD),
                "frames_in_hard_braking": int(len(braking)),
                "worst_consecutive_frames": worst_consecutive,
            },
            assumptions=list(assumptions or []),
        )]

    @staticmethod
    def _check_aggressive_acceleration(
        df: pd.DataFrame,
        track_id: int,
        assumptions: Optional[list] = None,
    ) -> List[RuleViolation]:
        """AGGRESSIVE_ACCELERATION — acceleration magnitude exceeds threshold."""
        aggressive = df[df["accel_mag"] > AGGRESSIVE_ACCEL_THRESHOLD]
        if aggressive.empty:
            return []

        max_accel  = float(df["accel_mag"].max())
        pct_flagged = round(100.0 * len(aggressive) / max(len(df), 1), 1)
        excess     = max_accel - AGGRESSIVE_ACCEL_THRESHOLD
        severity_score = round(min(1.0, excess / 3.5) * min(1.0, pct_flagged / 20.0), 3)

        return [RuleViolation(
            rule_id="AGGRESSIVE_ACCELERATION",
            description=(
                f"Vehicle {track_id} aggressive acceleration — "
                f"peak {max_accel:.2f} m/s² ({pct_flagged:.0f}% of window)"
            ),
            track_id=track_id,
            first_occurrence_s=float(aggressive["timestamp"].iloc[0]),
            severity="warning",
            severity_score=severity_score,
            evidence={
                "peak_accel_ms2": round(max_accel, 2),
                "threshold_ms2": AGGRESSIVE_ACCEL_THRESHOLD,
                "frames_flagged": int(len(aggressive)),
                "pct_frames_flagged": pct_flagged,
            },
            assumptions=list(assumptions or []),
        )]

    @staticmethod
    def _check_wrong_way(
        df: pd.DataFrame,
        track_id: int,
        flow_direction_deg: float,
        assumptions: Optional[list] = None,
    ) -> List[RuleViolation]:
        """
        WRONG_WAY — vehicle heading differs from expected flow direction by > 90°.

        Only applied to moving vehicles (speed > WRONG_WAY_MIN_SPEED_MS) to avoid
        false positives from noisy headings of near-stationary vehicles.
        Requires the anomaly to persist for at least WRONG_WAY_MIN_DURATION_S
        seconds to filter out transient noise (lane changes, U-turns completing).
        """
        moving = df[df["speed"] > WRONG_WAY_MIN_SPEED_MS].copy()
        if moving.empty:
            return []

        moving["heading_deg"] = moving.apply(
            lambda r: math.degrees(math.atan2(r["vel_y"], r["vel_x"])) % 360,
            axis=1,
        )

        def _angular_diff(heading: float) -> float:
            d = abs(heading - flow_direction_deg) % 360
            return min(d, 360.0 - d)

        moving["ang_diff"] = moving["heading_deg"].apply(_angular_diff)
        wrong = moving[moving["ang_diff"] > WRONG_WAY_ANGLE_DEG]
        if wrong.empty:
            return []

        duration = float(wrong["timestamp"].max()) - float(wrong["timestamp"].min())
        if duration < WRONG_WAY_MIN_DURATION_S and len(wrong) < 5:
            return []

        severity_score = round(min(1.0, duration / 5.0), 3)

        return [RuleViolation(
            rule_id="WRONG_WAY",
            description=(
                f"Vehicle {track_id} travelling against expected flow direction "
                f"(mean heading {wrong['heading_deg'].mean():.0f}° vs expected {flow_direction_deg:.0f}°) "
                f"for {duration:.1f}s"
            ),
            track_id=track_id,
            first_occurrence_s=float(wrong["timestamp"].iloc[0]),
            severity="violation",
            severity_score=severity_score,
            evidence={
                "expected_flow_direction_deg": flow_direction_deg,
                "mean_heading_deg": round(float(wrong["heading_deg"].mean()), 1),
                "mean_angular_difference_deg": round(float(wrong["ang_diff"].mean()), 1),
                "duration_s": round(duration, 2),
                "frames_flagged": int(len(wrong)),
            },
            assumptions=list(assumptions or []),
        )]

    @staticmethod
    def _check_stationary_in_lane(
        df: pd.DataFrame,
        track_id: int,
        assumptions: Optional[list] = None,
    ) -> List[RuleViolation]:
        """
        STATIONARY_IN_LANE — vehicle stopped in an active lane for > 10 s.

        Groups consecutive stopped frames into segments (gap < 2 s between rows
        is considered the same stop event) and flags any segment lasting longer
        than STATIONARY_MIN_DURATION_S.  This catches broken-down vehicles,
        double-parking incidents, and stalled vehicles — all primary causes
        of secondary collisions.
        """
        stopped = df[df["speed"] < STATIONARY_SPEED_THRESHOLD_MS]
        if stopped.empty:
            return []

        times = stopped["timestamp"].values
        if len(times) < 2:
            return []

        # Build contiguous stop segments: consecutive rows within 2 s of each other.
        segments: List[tuple] = []
        seg_start = seg_end = times[0]
        for i in range(1, len(times)):
            if times[i] - times[i - 1] < 2.0:
                seg_end = times[i]
            else:
                segments.append((seg_start, seg_end))
                seg_start = seg_end = times[i]
        segments.append((seg_start, seg_end))

        long_stops = [(s, e) for s, e in segments if e - s >= STATIONARY_MIN_DURATION_S]
        if not long_stops:
            return []

        first_start      = long_stops[0][0]
        max_duration     = max(e - s for s, e in long_stops)
        total_stopped_s  = round(sum(e - s for s, e in long_stops), 2)
        severity_score   = round(min(1.0, max_duration / 30.0), 3)

        return [RuleViolation(
            rule_id="STATIONARY_IN_LANE",
            description=(
                f"Vehicle {track_id} stationary in active lane — "
                f"longest stop {max_duration:.1f}s across {len(long_stops)} event(s)"
            ),
            track_id=track_id,
            first_occurrence_s=float(first_start),
            severity="warning",
            severity_score=severity_score,
            evidence={
                "max_stop_duration_s": round(max_duration, 2),
                "total_stopped_time_s": total_stopped_s,
                "threshold_s": STATIONARY_MIN_DURATION_S,
                "stop_events": len(long_stops),
                "speed_threshold_ms": STATIONARY_SPEED_THRESHOLD_MS,
            },
            assumptions=list(assumptions or []),
        )]
