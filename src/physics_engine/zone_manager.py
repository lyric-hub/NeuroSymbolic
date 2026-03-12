"""
Zone-based Vehicle Entry/Exit and OD (Origin-Destination) Tracker.

Concepts
--------
Gate
    A named directed line segment drawn across the road (in pixel coordinates).
    A crossing is detected when a vehicle's bottom-centre trajectory crosses the
    segment between two consecutive frames.  Direction (enter/exit) is inferred
    from which side of the gate normal the vehicle moved to.

Zone
    An optional closed polygon (pixel coordinates) used for occupancy counting
    and as a fallback membership check.  When a vehicle appears inside the
    polygon on its first visible frame it is registered as already inside.

OD study
    Each vehicle's first gate crossing is its "origin gate"; its last gate
    crossing is its "destination gate".  The dwell time between them and the
    full (origin, destination) pair are recorded per track_id so flow matrices
    can be queried after processing.

Missing-crossing recovery
-------------------------
Three scenarios cause a gate crossing to be silently missed:

  1. Vehicle appears already inside the polygon on frame 1 (camera started
     late, vehicle was parked inside, or entered before detection began).
  2. The detector drops the track for 1–2 frames exactly at the gate, so the
     movement vector "jumps" over the gate line entirely.
  3. ByteTrack drops and re-assigns a new track_id at the boundary; the new
     id appears inside with no crossing history.

In all three cases the zone manager detects the polygon-membership transition
(outside → inside, or inside → outside) and assigns the **nearest gate** as an
``"estimated"`` crossing.  Confirmed crossings (actual line-segment
intersection detected) are tagged ``"confirmed"``.  The confidence field is
stored in DuckDB and exposed through the agent tool so analysts can distinguish
deterministic from inferred OD pairs.

Gate coordinates and homography
--------------------------------
Gates are defined in **pixel coordinates** on the video frame.  Crossing
detection is pixel-space geometry (line-segment intersection) and is correct
regardless of camera perspective.  The **real-world** position of each
crossing (``real_x``, ``real_y``) is sourced from ``CoordinateTransformer``
output (the homography), not from the gate definition itself — so no separate
gate calibration is required.  If the camera is moved, both the calibration
and the zone must be redrawn.

Configuration
-------------
Zones and gates are defined in ``zone_config.json`` at the project root.
Use the web UI at ``/zone-ui`` to draw and save a zone interactively.

Example ``zone_config.json``::

    {
      "zone_id": "main_intersection",
      "polygon": [[120, 180], [820, 180], [820, 620], [120, 620]],
      "gates": [
        {"name": "North", "p1": [120, 180], "p2": [820, 180]},
        {"name": "South", "p1": [120, 620], "p2": [820, 620]},
        {"name": "East",  "p1": [820, 180], "p2": [820, 620]},
        {"name": "West",  "p1": [120, 180], "p2": [120, 620]}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    """A named virtual line segment across the road (pixel coordinates)."""
    name: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]


@dataclass
class ZoneConfig:
    """Serialisable zone definition loaded from / saved to zone_config.json."""
    zone_id: str
    polygon: List[Tuple[float, float]]  # closed polygon in pixel coordinates
    gates: List[Gate]

    @classmethod
    def from_json(cls, path: str = "zone_config.json") -> "ZoneConfig":
        """Loads a ZoneConfig from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        gates = [
            Gate(g["name"], tuple(g["p1"]), tuple(g["p2"]))
            for g in data.get("gates", [])
        ]
        polygon = [tuple(pt) for pt in data.get("polygon", [])]
        return cls(zone_id=data["zone_id"], polygon=polygon, gates=gates)

    def to_json(self, path: str = "zone_config.json") -> None:
        """Serialises the zone configuration to a JSON file."""
        data = {
            "zone_id": self.zone_id,
            "polygon": [list(pt) for pt in self.polygon],
            "gates": [
                {"name": g.name, "p1": list(g.p1), "p2": list(g.p2)}
                for g in self.gates
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class CrossingEvent:
    """
    A single gate-crossing event emitted by ZoneManager.update().

    Attributes:
        confidence: ``"confirmed"`` when an actual line-segment intersection
            was detected; ``"estimated"`` when the crossing was inferred from a
            polygon-membership transition (missed detector, late appearance, or
            track re-assignment at the boundary).
    """
    track_id: int
    zone_id: str
    gate_name: str
    direction: str          # 'enter' or 'exit'
    confidence: str         # 'confirmed' or 'estimated'
    timestamp: float
    frame_id: int
    pixel_x: float          # pixel coordinate of the crossing point
    pixel_y: float
    real_x: float           # real-world coordinate at that frame
    real_y: float


@dataclass
class _VehicleZoneState:
    """Internal per-vehicle tracking state."""
    inside: bool
    entry_gate: Optional[str] = None
    entry_time: Optional[float] = None
    entry_gate_confidence: str = "confirmed"
    exit_gate: Optional[str] = None
    exit_time: Optional[float] = None
    exit_gate_confidence: str = "confirmed"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _vec2(a: Tuple, b: Tuple) -> np.ndarray:
    return np.array([b[0] - a[0], b[1] - a[1]], dtype=float)


def _segments_intersect(
    p1: Tuple, p2: Tuple, p3: Tuple, p4: Tuple
) -> Optional[Tuple[float, float]]:
    """
    Returns the intersection point of segment p1→p2 with segment p3→p4,
    or None if they do not intersect.
    """
    d1 = _vec2(p1, p2)
    d2 = _vec2(p3, p4)
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return None  # parallel or collinear
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (p1[0] + t * d1[0], p1[1] + t * d1[1])
    return None


def _crossing_direction(
    prev: Tuple, curr: Tuple, g_p1: Tuple, g_p2: Tuple
) -> str:
    """
    Returns 'enter' or 'exit' based on which side of the gate the vehicle
    moved to.  The gate's inward normal points 90° counter-clockwise from
    the p1→p2 direction vector.
    """
    gate_vec = _vec2(g_p1, g_p2)
    normal = np.array([-gate_vec[1], gate_vec[0]])
    movement = _vec2(prev, curr)
    return "enter" if np.dot(movement, normal) >= 0 else "exit"


def _point_in_polygon(point: Tuple, polygon: List[Tuple]) -> bool:
    """Ray-casting algorithm for point-in-polygon membership test."""
    px, py = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def _nearest_gate_name(pos: Tuple, gates: List[Gate]) -> Optional[str]:
    """
    Returns the name of the gate whose midpoint is closest (Euclidean) to
    ``pos``.  Used when a gate line intersection was not detected (missed
    crossing recovery).
    """
    if not gates:
        return None
    best_name, best_d2 = None, float("inf")
    for gate in gates:
        mx = (gate.p1[0] + gate.p2[0]) / 2.0
        my = (gate.p1[1] + gate.p2[1]) / 2.0
        d2 = (pos[0] - mx) ** 2 + (pos[1] - my) ** 2
        if d2 < best_d2:
            best_d2, best_name = d2, gate.name
    return best_name


# ---------------------------------------------------------------------------
# ZoneManager
# ---------------------------------------------------------------------------

class ZoneManager:
    """
    Processes tracked bounding boxes frame-by-frame to detect gate crossings
    and polygon entry/exit events.  Maintains per-vehicle OD state.

    Missing-crossing recovery
    -------------------------
    After checking each gate's line-segment intersection, the manager compares
    the vehicle's polygon membership before and after the frame.  If membership
    changed (outside→inside or inside→outside) without a corresponding gate
    crossing being detected, it emits an ``"estimated"`` CrossingEvent
    attributed to the nearest gate midpoint.

    This covers:
    - Vehicles present inside the zone on the first processed frame.
    - Fast vehicles whose movement vector skips over the gate line.
    - Track re-assignments at the zone boundary.

    Usage::

        config = ZoneConfig.from_json("zone_config.json")
        zm = ZoneManager(config)

        # Inside the micro-loop:
        events = zm.update(tracked_boxes, real_coords, timestamp, frame_id)
        for ev in events:
            duckdb_client.insert_crossing_event(ev)

        # After processing, query completed OD pairs:
        od_pairs = zm.get_od_summary()
    """

    def __init__(self, config: ZoneConfig) -> None:
        self.config = config
        self._prev_positions: Dict[int, Tuple[float, float]] = {}
        self._vehicle_state: Dict[int, _VehicleZoneState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracked_boxes: np.ndarray,
        real_coords: Dict[int, Tuple[float, float]],
        timestamp: float,
        frame_id: int,
    ) -> List[CrossingEvent]:
        """
        Processes one frame and returns any gate-crossing events.

        Args:
            tracked_boxes: (N, 8) array from VehicleTracker.update().
                           Columns: [x1, y1, x2, y2, track_id, conf, cls, ind].
            real_coords:   {track_id: (real_x, real_y)} from CoordinateTransformer.
            timestamp:     Current video timestamp in seconds.
            frame_id:      Current frame index.

        Returns:
            List of CrossingEvent — one per gate crossed per vehicle this frame.
            Includes both ``"confirmed"`` (line-segment intersection) and
            ``"estimated"`` (missed-crossing recovery) events.
        """
        if tracked_boxes.shape[0] == 0:
            return []

        events: List[CrossingEvent] = []

        # Build pixel bottom-centres: bottom-centre = ((x1+x2)/2, y2)
        curr_positions: Dict[int, Tuple[float, float]] = {}
        for row in tracked_boxes:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            tid = int(row[4])
            curr_positions[tid] = ((x1 + x2) / 2.0, y2)

        for tid, curr_pos in curr_positions.items():
            real_pos = real_coords.get(tid, (0.0, 0.0))

            # ── First appearance ──────────────────────────────────────────
            if tid not in self._vehicle_state:
                inside = (
                    _point_in_polygon(curr_pos, self.config.polygon)
                    if self.config.polygon else False
                )
                state = _VehicleZoneState(inside=inside)
                self._vehicle_state[tid] = state

                if inside and self.config.gates:
                    # Vehicle appeared already inside — no crossing history.
                    # Attribute entry to nearest gate with "estimated" confidence.
                    gate_name = _nearest_gate_name(curr_pos, self.config.gates)
                    self._record_od(tid, gate_name, "enter", timestamp, "estimated")
                    events.append(self._make_event(
                        tid, gate_name, "enter", "estimated",
                        timestamp, frame_id, curr_pos, real_pos,
                    ))
                # Skip gate-crossing check — no previous position yet
                self._prev_positions[tid] = curr_pos
                continue

            state = self._vehicle_state[tid]
            was_inside = state.inside

            # ── Confirmed gate-crossing check ─────────────────────────────
            gate_crossed_this_frame = False
            if tid in self._prev_positions:
                prev_pos = self._prev_positions[tid]
                for gate in self.config.gates:
                    pt = _segments_intersect(prev_pos, curr_pos, gate.p1, gate.p2)
                    if pt is None:
                        continue
                    direction = _crossing_direction(prev_pos, curr_pos, gate.p1, gate.p2)
                    events.append(self._make_event(
                        tid, gate.name, direction, "confirmed",
                        timestamp, frame_id, pt, real_pos,
                    ))
                    self._record_od(tid, gate.name, direction, timestamp, "confirmed")
                    gate_crossed_this_frame = True

            # ── Update polygon membership ─────────────────────────────────
            now_inside = (
                _point_in_polygon(curr_pos, self.config.polygon)
                if self.config.polygon else was_inside
            )
            state.inside = now_inside

            # ── Missed-crossing recovery ──────────────────────────────────
            # Membership changed but no gate line was intersected this frame.
            # This catches: fast vehicles, frame gaps, track re-assignments.
            if not gate_crossed_this_frame and self.config.gates:
                if not was_inside and now_inside:
                    # Entered zone without a detected gate crossing.
                    gate_name = _nearest_gate_name(curr_pos, self.config.gates)
                    if self._vehicle_state[tid].entry_gate is None:
                        self._record_od(tid, gate_name, "enter", timestamp, "estimated")
                        events.append(self._make_event(
                            tid, gate_name, "enter", "estimated",
                            timestamp, frame_id, curr_pos, real_pos,
                        ))

                elif was_inside and not now_inside:
                    # Exited zone without a detected gate crossing.
                    gate_name = _nearest_gate_name(curr_pos, self.config.gates)
                    self._record_od(tid, gate_name, "exit", timestamp, "estimated")
                    events.append(self._make_event(
                        tid, gate_name, "exit", "estimated",
                        timestamp, frame_id, curr_pos, real_pos,
                    ))

            self._prev_positions[tid] = curr_pos

        return events

    def current_occupancy(self) -> int:
        """Returns the number of vehicles currently inside the zone polygon."""
        return sum(1 for s in self._vehicle_state.values() if s.inside)

    def get_od_summary(self) -> List[Dict]:
        """
        Returns completed OD pairs — vehicles that have both entered and exited.

        Each row: track_id, origin_gate, destination_gate, entry_time,
        exit_time, dwell_time_seconds, entry_confidence, exit_confidence.
        """
        rows = []
        for tid, state in self._vehicle_state.items():
            if state.entry_gate and state.exit_gate:
                dwell = (
                    round(state.exit_time - state.entry_time, 2)
                    if state.exit_time is not None and state.entry_time is not None
                    else None
                )
                rows.append({
                    "track_id": tid,
                    "origin_gate": state.entry_gate,
                    "destination_gate": state.exit_gate,
                    "entry_time": state.entry_time,
                    "exit_time": state.exit_time,
                    "dwell_time_seconds": dwell,
                    "entry_confidence": state.entry_gate_confidence,
                    "exit_confidence": state.exit_gate_confidence,
                })
        return rows

    def get_gate_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Returns cumulative enter/exit counts per gate (confirmed + estimated).
        """
        counts: Dict[str, Dict[str, int]] = {
            g.name: {"enter": 0, "exit": 0} for g in self.config.gates
        }
        for state in self._vehicle_state.values():
            if state.entry_gate and state.entry_gate in counts:
                counts[state.entry_gate]["enter"] += 1
            if state.exit_gate and state.exit_gate in counts:
                counts[state.exit_gate]["exit"] += 1
        return counts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_event(
        self,
        tid: int,
        gate_name: str,
        direction: str,
        confidence: str,
        timestamp: float,
        frame_id: int,
        pixel_pos: Tuple,
        real_pos: Tuple,
    ) -> CrossingEvent:
        return CrossingEvent(
            track_id=tid,
            zone_id=self.config.zone_id,
            gate_name=gate_name,
            direction=direction,
            confidence=confidence,
            timestamp=timestamp,
            frame_id=frame_id,
            pixel_x=pixel_pos[0],
            pixel_y=pixel_pos[1],
            real_x=real_pos[0],
            real_y=real_pos[1],
        )

    def _record_od(
        self,
        tid: int,
        gate_name: str,
        direction: str,
        timestamp: float,
        confidence: str,
    ) -> None:
        state = self._vehicle_state[tid]
        if direction == "enter" and state.entry_gate is None:
            state.entry_gate = gate_name
            state.entry_time = timestamp
            state.entry_gate_confidence = confidence
            state.inside = True
        elif direction == "exit":
            # Always update so the last crossing is the destination gate
            state.exit_gate = gate_name
            state.exit_time = timestamp
            state.exit_gate_confidence = confidence
            state.inside = False
