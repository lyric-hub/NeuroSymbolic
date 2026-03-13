"""
Alert Engine — Real-time traffic event detection.

Runs in the micro-loop (every frame) alongside the physics engine.
Fires callbacks immediately when kinematic thresholds are crossed,
without waiting for the VLM macro-loop.

Alert types:
  - SPEEDING               : speed > 50 km/h (13.89 m/s)
  - HARD_BRAKING           : signed deceleration < -4.0 m/s²
  - AGGRESSIVE_ACCEL       : acceleration magnitude > 3.5 m/s²
  - PROXIMITY_WARNING      : two vehicles < 5 m apart and closing
  - COLLISION_SUSPECTED    : two vehicles < 2 m apart

Thresholds match TrafficRuleEngine constants for consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Callable, Dict, List, Tuple, Union

# Thresholds — kept in sync with rule_engine.py
SPEEDING_THRESHOLD_MS       = 13.89   # m/s  (50 km/h)
HARD_BRAKING_THRESHOLD      = -4.0    # m/s²
AGGRESSIVE_ACCEL_THRESHOLD  =  3.5    # m/s²
PROXIMITY_WARNING_M         =  5.0    # metres between two vehicles
COLLISION_THRESHOLD_M       =  2.0    # metres — probable contact

log = logging.getLogger(__name__)


@dataclass
class TrafficAlert:
    """
    A single real-time alert produced by AlertEngine.

    Attributes:
        alert_type:   One of SPEEDING / HARD_BRAKING / AGGRESSIVE_ACCEL /
                      PROXIMITY_WARNING / COLLISION_SUSPECTED.
        severity:     "critical" or "warning".
        track_id:     Primary vehicle ID (-1 for pair-only alerts).
        involved_ids: All vehicle IDs implicated (1 or 2 vehicles).
        timestamp:    Video timestamp in seconds when the alert fired.
        frame_id:     Frame index when the alert fired.
        message:      Human-readable description.
        evidence:     Exact numeric values that triggered the rule.
    """

    alert_type:   str
    severity:     str
    track_id:     int
    involved_ids: List[int]
    timestamp:    float
    frame_id:     int
    message:      str
    evidence:     Dict

    def to_dict(self) -> dict:
        """Serialises the alert to a plain dict (JSON-safe)."""
        return asdict(self)


class AlertEngine:
    """
    Real-time alert detector that runs in the micro-loop every frame.

    All checks are deterministic and threshold-based — no neural models
    are involved.  Each alert type has a per-entity cooldown so a
    persisting condition (e.g. a vehicle speeding for 10 seconds) does
    not spam the callback.

    Usage::

        def my_handler(alert: TrafficAlert) -> None:
            print(alert.message)

        engine = AlertEngine(on_alert=my_handler, cooldown_secs=3.0)

        # Inside the micro-loop:
        engine.check(state_vectors, real_coords, timestamp, frame_id)

    Args:
        on_alert:      Callable invoked synchronously for each fired alert.
        cooldown_secs: Minimum gap (seconds) before the same alert type
                       re-fires for the same vehicle or vehicle pair.
                       Default: 3.0 s.
    """

    def __init__(
        self,
        on_alert: Callable[[TrafficAlert], None],
        cooldown_secs: float = 3.0,
    ) -> None:
        self._on_alert  = on_alert
        self._cooldown  = cooldown_secs
        # Maps (entity_key, alert_type) → last fired timestamp.
        # entity_key is int (single vehicle) or tuple (vehicle pair).
        self._last_fired: Dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        state_vectors: Dict[int, List[float]],
        real_coords:   Dict[int, Tuple[float, float]],
        timestamp:     float,
        frame_id:      int,
    ) -> None:
        """
        Runs all alert checks for one frame.

        Call this in the micro-loop immediately after
        ``KinematicEstimator.update()``.

        Args:
            state_vectors: ``{track_id: [x, y, vx, vy, ax, ay]}`` from
                           KinematicEstimator.
            real_coords:   ``{track_id: (real_x, real_y)}`` in metres from
                           CoordinateTransformer.
            timestamp:     Current video timestamp in seconds.
            frame_id:      Current frame index.
        """
        for track_id, sv in state_vectors.items():
            _, _, vx, vy, ax, ay = sv
            self._check_speeding(track_id, vx, vy, timestamp, frame_id)
            self._check_hard_braking(track_id, vx, vy, ax, ay, timestamp, frame_id)
            self._check_aggressive_accel(track_id, ax, ay, timestamp, frame_id)

        if len(real_coords) >= 2:
            self._check_proximity(real_coords, state_vectors, timestamp, frame_id)

    # ------------------------------------------------------------------
    # Per-vehicle checks
    # ------------------------------------------------------------------

    def _check_speeding(
        self,
        track_id: int,
        vx: float, vy: float,
        timestamp: float,
        frame_id: int,
    ) -> None:
        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed <= SPEEDING_THRESHOLD_MS:
            return
        if not self._should_fire(track_id, "SPEEDING", timestamp):
            return
        self._fire(TrafficAlert(
            alert_type="SPEEDING",
            severity="critical",
            track_id=track_id,
            involved_ids=[track_id],
            timestamp=timestamp,
            frame_id=frame_id,
            message=(
                f"Vehicle {track_id} is SPEEDING at "
                f"{speed:.1f} m/s ({speed * 3.6:.1f} km/h) — "
                f"limit is {SPEEDING_THRESHOLD_MS * 3.6:.0f} km/h."
            ),
            evidence={
                "speed_ms":       round(speed, 2),
                "speed_kmh":      round(speed * 3.6, 1),
                "threshold_kmh":  round(SPEEDING_THRESHOLD_MS * 3.6, 1),
            },
        ))

    def _check_hard_braking(
        self,
        track_id: int,
        vx: float, vy: float,
        ax: float, ay: float,
        timestamp: float,
        frame_id: int,
    ) -> None:
        accel_mag    = (ax ** 2 + ay ** 2) ** 0.5
        dot          = vx * ax + vy * ay
        signed_accel = -accel_mag if dot < 0 else accel_mag
        if signed_accel >= HARD_BRAKING_THRESHOLD:
            return
        if not self._should_fire(track_id, "HARD_BRAKING", timestamp):
            return
        self._fire(TrafficAlert(
            alert_type="HARD_BRAKING",
            severity="warning",
            track_id=track_id,
            involved_ids=[track_id],
            timestamp=timestamp,
            frame_id=frame_id,
            message=(
                f"Vehicle {track_id} HARD BRAKING at "
                f"{signed_accel:.1f} m/s² "
                f"(threshold: {HARD_BRAKING_THRESHOLD} m/s²)."
            ),
            evidence={
                "signed_accel_ms2": round(signed_accel, 2),
                "threshold_ms2":    HARD_BRAKING_THRESHOLD,
            },
        ))

    def _check_aggressive_accel(
        self,
        track_id: int,
        ax: float, ay: float,
        timestamp: float,
        frame_id: int,
    ) -> None:
        accel_mag = (ax ** 2 + ay ** 2) ** 0.5
        if accel_mag <= AGGRESSIVE_ACCEL_THRESHOLD:
            return
        if not self._should_fire(track_id, "AGGRESSIVE_ACCEL", timestamp):
            return
        self._fire(TrafficAlert(
            alert_type="AGGRESSIVE_ACCEL",
            severity="warning",
            track_id=track_id,
            involved_ids=[track_id],
            timestamp=timestamp,
            frame_id=frame_id,
            message=(
                f"Vehicle {track_id} AGGRESSIVE ACCELERATION at "
                f"{accel_mag:.1f} m/s² "
                f"(threshold: {AGGRESSIVE_ACCEL_THRESHOLD} m/s²)."
            ),
            evidence={
                "accel_mag_ms2": round(accel_mag, 2),
                "threshold_ms2": AGGRESSIVE_ACCEL_THRESHOLD,
            },
        ))

    # ------------------------------------------------------------------
    # Cross-vehicle proximity check
    # ------------------------------------------------------------------

    def _check_proximity(
        self,
        real_coords:   Dict[int, Tuple[float, float]],
        state_vectors: Dict[int, List[float]],
        timestamp:     float,
        frame_id:      int,
    ) -> None:
        ids = list(real_coords.keys())
        for id_a, id_b in combinations(ids, 2):
            xa, ya = real_coords[id_a]
            xb, yb = real_coords[id_b]
            dx   = xb - xa
            dy   = yb - ya
            dist = (dx ** 2 + dy ** 2) ** 0.5

            pair_key = (min(id_a, id_b), max(id_a, id_b))

            if dist < COLLISION_THRESHOLD_M:
                if self._should_fire(pair_key, "COLLISION_SUSPECTED", timestamp):
                    self._fire(TrafficAlert(
                        alert_type="COLLISION_SUSPECTED",
                        severity="critical",
                        track_id=id_a,
                        involved_ids=[id_a, id_b],
                        timestamp=timestamp,
                        frame_id=frame_id,
                        message=(
                            f"COLLISION SUSPECTED: Vehicle {id_a} and "
                            f"Vehicle {id_b} are only {dist:.2f} m apart."
                        ),
                        evidence={
                            "distance_m":            round(dist, 2),
                            "collision_threshold_m": COLLISION_THRESHOLD_M,
                            "vehicle_a": id_a,
                            "vehicle_b": id_b,
                        },
                    ))
                # Skip the proximity warning for the same pair this frame.
                continue

            if dist < PROXIMITY_WARNING_M:
                closing = self._is_closing(id_a, id_b, dx, dy, state_vectors)
                if closing and self._should_fire(pair_key, "PROXIMITY_WARNING", timestamp):
                    self._fire(TrafficAlert(
                        alert_type="PROXIMITY_WARNING",
                        severity="warning",
                        track_id=id_a,
                        involved_ids=[id_a, id_b],
                        timestamp=timestamp,
                        frame_id=frame_id,
                        message=(
                            f"PROXIMITY WARNING: Vehicle {id_a} and "
                            f"Vehicle {id_b} are {dist:.1f} m apart and closing."
                        ),
                        evidence={
                            "distance_m":          round(dist, 2),
                            "warning_threshold_m": PROXIMITY_WARNING_M,
                            "vehicle_a": id_a,
                            "vehicle_b": id_b,
                        },
                    ))

    @staticmethod
    def _is_closing(
        id_a: int,
        id_b: int,
        dx: float,
        dy: float,
        state_vectors: Dict[int, List[float]],
    ) -> bool:
        """
        Returns True when vehicles a and b are moving toward each other.

        Uses the sign of the dot product between the relative position
        vector (b − a) and the relative velocity vector (vb − va).
        A negative dot product means b is approaching a.
        If velocity data is unavailable, conservatively returns True.
        """
        sv_a = state_vectors.get(id_a)
        sv_b = state_vectors.get(id_b)
        if sv_a is None or sv_b is None:
            return True  # No data — assume closing (conservative)
        rvx = sv_b[2] - sv_a[2]
        rvy = sv_b[3] - sv_a[3]
        return (dx * rvx + dy * rvy) < 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_fire(
        self,
        entity_key: Union[int, tuple],
        alert_type: str,
        timestamp: float,
    ) -> bool:
        """
        Returns True and records the timestamp when the cooldown has elapsed.

        Args:
            entity_key: track_id (int) for single-vehicle alerts, or
                        (min_id, max_id) tuple for pair alerts.
            alert_type: Alert type string.
            timestamp:  Current video timestamp.
        """
        cooldown_key = (entity_key, alert_type)
        last = self._last_fired.get(cooldown_key, -self._cooldown - 1.0)
        if timestamp - last >= self._cooldown:
            self._last_fired[cooldown_key] = timestamp
            return True
        return False

    def _fire(self, alert: TrafficAlert) -> None:
        """Logs the alert and invokes the user-supplied callback."""
        log.info("[ALERT] %s | %s", alert.alert_type, alert.message)
        try:
            self._on_alert(alert)
        except Exception:
            log.exception("on_alert callback raised an exception.")
