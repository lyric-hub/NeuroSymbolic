"""
Kinematic State Vector Extractor.

Estimates per-track position, velocity, and acceleration from world-coordinate
histories using longer trailing windows rather than near-instantaneous
frame-to-frame derivatives.

Design
------
- Speed is computed over a trailing displacement window (default 0.25 s).
- Acceleration is computed from the change in windowed speed over a longer
  trailing window (default 0.5 s).
- Short detector dropouts are backfilled with synthetic rows so the stored
  trajectory stays continuous, but those rows are explicitly marked low-trust.
- Quality flags are returned for every real and synthetic row so downstream
  rule logic and VLM summaries can ignore unstable motion estimates.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


# Backward-compatible constant — retained so existing imports/tests don't break.
MAX_MISSED_FRAMES: int = 5

_MIN_HISTORY_SAMPLES: int = 2


def _kmh_to_ms(value_kmh: float) -> float:
    return float(value_kmh) / 3.6


class KinematicEstimator:
    """Estimates position, velocity, and acceleration for each tracked vehicle."""

    def __init__(
        self,
        fps: float = 30.0,
        window_length: int = 7,
        polyorder: int = 3,
        max_missed_frames: int = 30,
        frame_step: int = 1,
        speed_window_s: float = 0.25,
        accel_window_s: float = 0.5,
        min_trusted_samples: int = 5,
        max_speed_kmh: float = 140.0,
        max_pos_accel_kmhps: float = 25.0,
        max_brake_kmhps: float = 35.0,
    ) -> None:
        """
        Initialises the kinematic estimator.

        Args:
            fps:                 Effective sampling rate in Hz (video_fps / frame_step).
            window_length:       Minimum retained history depth. Increased
                                 internally when needed for the trailing windows.
            polyorder:           Unused — kept for API compatibility.
            max_missed_frames:   Consecutive processed frames a track may be
                                 absent before its trajectory history is purged.
            frame_step:          Video frames between consecutive observations.
            speed_window_s:      Trailing time window used for reported speed.
            accel_window_s:      Trailing time window used for reported acceleration.
            min_trusted_samples: Minimum count of real measurements before a
                                 track is considered warm enough for rules.
            max_speed_kmh:       Hard plausibility gate for speed.
            max_pos_accel_kmhps: Hard plausibility gate for positive acceleration.
            max_brake_kmhps:     Hard plausibility gate for braking magnitude.
        """
        self.fps = float(fps)
        self.dt = 1.0 / max(self.fps, 1e-6)
        self.max_missed_frames = max_missed_frames
        self.frame_step = frame_step
        self.speed_window_s = float(speed_window_s)
        self.accel_window_s = float(accel_window_s)
        self.min_trusted_samples = max(2, int(min_trusted_samples))
        self.max_speed_ms = _kmh_to_ms(max_speed_kmh)
        self.max_pos_accel_ms2 = _kmh_to_ms(max_pos_accel_kmhps)
        self.max_brake_ms2 = _kmh_to_ms(max_brake_kmhps)

        history_secs = max(
            1.5,
            self.speed_window_s + self.accel_window_s + 2.0 * self.dt,
        )
        self.window_length = max(
            _MIN_HISTORY_SAMPLES,
            window_length,
            int(round(history_secs * self.fps)) + 2,
        )
        self.polyorder = polyorder  # retained for API compatibility only

        self.trajectories: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.window_length)
        )
        self._meta: Dict[int, Deque[dict]] = defaultdict(
            lambda: deque(maxlen=self.window_length)
        )
        self._missed: Dict[int, int] = {}
        self.warm_tracks: Set[int] = set()
        self._last_timestamp: Dict[int, float] = {}
        self._last_frame_id: Dict[int, int] = {}
        self._last_trusted_motion: Dict[int, dict] = {}
        self.last_state_quality: Dict[int, dict] = {}

    def _default_measurement_info(self) -> dict:
        return {
            "measurement_used": True,
            "predicted_only": False,
            "association_iou": None,
            "strong_association": True,
            "inside_calibration_core": True,
        }

    def _clean_measurement_info(self, info: Optional[dict], *, interpolated: bool) -> dict:
        base = self._default_measurement_info()
        if info:
            base.update(info)
        base["measurement_used"] = bool(base.get("measurement_used", True))
        base["predicted_only"] = bool(base.get("predicted_only", False))
        base["strong_association"] = bool(base.get("strong_association", True))
        base["inside_calibration_core"] = bool(
            base.get("inside_calibration_core", True)
        )
        assoc_iou = base.get("association_iou")
        base["association_iou"] = (
            float(assoc_iou) if assoc_iou is not None else None
        )
        base["interpolated"] = bool(interpolated)
        return base

    def _append_sample(
        self,
        track_id: int,
        x: float,
        y: float,
        timestamp: float,
        frame_id: int,
        info: dict,
    ) -> None:
        self.trajectories[track_id].append((x, y))
        self._meta[track_id].append({
            "timestamp": float(timestamp),
            "frame_id": int(frame_id),
            **info,
        })

    def _find_window_start(
        self,
        meta: Deque[dict],
        end_idx: int,
        window_s: float,
    ) -> Optional[int]:
        if end_idx <= 0:
            return None
        end_ts = float(meta[end_idx]["timestamp"])
        for idx in range(end_idx - 1, -1, -1):
            if end_ts - float(meta[idx]["timestamp"]) >= window_s:
                return idx
        return 0 if end_idx > 0 else None

    def _compute_velocity(
        self,
        history: Deque[Tuple[float, float]],
        meta: Deque[dict],
        end_idx: int,
    ) -> Tuple[float, float, bool, float, int, bool]:
        if end_idx <= 0:
            return 0.0, 0.0, False, 0.0, end_idx, False

        start_idx = self._find_window_start(meta, end_idx, self.speed_window_s)
        if start_idx is None:
            return 0.0, 0.0, False, 0.0, end_idx, False

        end_ts = float(meta[end_idx]["timestamp"])
        start_ts = float(meta[start_idx]["timestamp"])
        dt = end_ts - start_ts
        if dt <= 1e-6:
            return 0.0, 0.0, False, 0.0, start_idx, False

        # Collect inter-frame velocity vectors for consecutive pairs in the
        # window, then return the component-wise median.  A single bad kp31
        # position produces exactly one positive and one negative spike; the
        # median of the remaining correct samples is unaffected.
        vx_samples: List[float] = []
        vy_samples: List[float] = []
        for i in range(start_idx + 1, end_idx + 1):
            x0, y0 = history[i - 1]
            x1, y1 = history[i]
            ts0 = float(meta[i - 1]["timestamp"])
            ts1 = float(meta[i]["timestamp"])
            frame_dt = ts1 - ts0
            if frame_dt > 1e-6:
                vx_samples.append((x1 - x0) / frame_dt)
                vy_samples.append((y1 - y0) / frame_dt)

        if not vx_samples:
            return 0.0, 0.0, False, dt, start_idx, False

        n = len(vx_samples)
        vx_samples.sort()
        vy_samples.sort()
        mid = n // 2
        if n % 2 == 1:
            vx = vx_samples[mid]
            vy = vy_samples[mid]
        else:
            vx = (vx_samples[mid - 1] + vx_samples[mid]) / 2.0
            vy = (vy_samples[mid - 1] + vy_samples[mid]) / 2.0

        ready = dt >= (0.95 * self.speed_window_s)
        clean = all(
            m["measurement_used"]
            and not m["predicted_only"]
            and not m["interpolated"]
            for m in list(meta)[start_idx : end_idx + 1]
        )
        return vx, vy, ready, dt, start_idx, clean

    def _compute_acceleration(
        self,
        history: Deque[Tuple[float, float]],
        meta: Deque[dict],
        end_idx: int,
        vx: float,
        vy: float,
    ) -> Tuple[float, float, bool, float, float]:
        if end_idx <= 1:
            return 0.0, 0.0, False, 0.0, 0.0

        anchor_idx = self._find_window_start(meta, end_idx, self.accel_window_s)
        if anchor_idx is None or anchor_idx <= 0:
            return 0.0, 0.0, False, 0.0, 0.0

        prev_vx, prev_vy, prev_ready, _, _, _ = self._compute_velocity(
            history, meta, anchor_idx
        )
        end_ts = float(meta[end_idx]["timestamp"])
        anchor_ts = float(meta[anchor_idx]["timestamp"])
        dt = end_ts - anchor_ts
        if dt <= 1e-6 or not prev_ready:
            return 0.0, 0.0, False, 0.0, 0.0

        curr_speed = math.hypot(vx, vy)
        prev_speed = math.hypot(prev_vx, prev_vy)
        signed_accel = (curr_speed - prev_speed) / dt

        if curr_speed > 1e-6:
            ux = vx / curr_speed
            uy = vy / curr_speed
        elif prev_speed > 1e-6:
            ux = prev_vx / prev_speed
            uy = prev_vy / prev_speed
        else:
            ux = 0.0
            uy = 0.0

        ax = signed_accel * ux
        ay = signed_accel * uy
        ready = dt >= (0.95 * self.accel_window_s)
        return ax, ay, ready, dt, signed_accel

    def _accepted_measurement_count(self, meta: Deque[dict]) -> int:
        return sum(
            1
            for m in meta
            if m["measurement_used"]
            and not m["predicted_only"]
            and not m["interpolated"]
        )

    def _build_state(
        self,
        track_id: int,
        current_info: dict,
    ) -> Tuple[List[float], dict]:
        history = self.trajectories[track_id]
        meta = self._meta[track_id]
        end_idx = len(history) - 1
        x, y = history[end_idx]

        vx, vy, speed_ready, _, _, speed_window_clean = self._compute_velocity(
            history, meta, end_idx
        )
        speed_ms = math.hypot(vx, vy)

        ax, ay, accel_ready, _, signed_accel = self._compute_acceleration(
            history, meta, end_idx, vx, vy
        )

        track_warm = self._accepted_measurement_count(meta) >= self.min_trusted_samples
        current_real = (
            current_info["measurement_used"]
            and not current_info["predicted_only"]
            and not current_info["interpolated"]
        )

        outlier_rejected = False
        if speed_ms > self.max_speed_ms:
            outlier_rejected = True
        if accel_ready and (
            signed_accel > self.max_pos_accel_ms2
            or signed_accel < -self.max_brake_ms2
        ):
            outlier_rejected = True

        if outlier_rejected:
            fallback = self._last_trusted_motion.get(track_id)
            if fallback is not None:
                vx = fallback["vel_x"]
                vy = fallback["vel_y"]
                ax = fallback["accel_x"]
                ay = fallback["accel_y"]
                speed_ms = fallback["speed_ms"]
                signed_accel = fallback["signed_accel_ms2"]
            else:
                vx = vy = ax = ay = 0.0
                speed_ms = 0.0
                signed_accel = 0.0

        trusted_for_rules = all((
            track_warm,
            speed_ready,
            current_real,
            current_info["strong_association"],
            current_info["inside_calibration_core"],
            speed_window_clean,
            not outlier_rejected,
        ))

        if track_warm:
            self.warm_tracks.add(track_id)
        else:
            self.warm_tracks.discard(track_id)

        if trusted_for_rules:
            self._last_trusted_motion[track_id] = {
                "vel_x": vx,
                "vel_y": vy,
                "accel_x": ax,
                "accel_y": ay,
                "speed_ms": speed_ms,
                "signed_accel_ms2": signed_accel,
            }

        quality = {
            "measurement_used": current_info["measurement_used"],
            "predicted_only": current_info["predicted_only"],
            "interpolated": current_info["interpolated"],
            "outlier_rejected": outlier_rejected,
            "track_warm": track_warm,
            "trusted_for_rules": trusted_for_rules,
            "speed_ms": speed_ms,
            "speed_kmh": speed_ms * 3.6,
            "signed_accel_ms2": signed_accel,
            "association_iou": current_info["association_iou"],
            "inside_calibration_core": current_info["inside_calibration_core"],
            "strong_association": current_info["strong_association"],
        }
        return [x, y, vx, vy, ax, ay], quality

    def update(
        self,
        current_coords: Dict[int, Tuple[float, float]],
        timestamp: float,
        frame_id: int,
        measurement_info: Optional[Dict[int, dict]] = None,
    ) -> Tuple[Dict[int, List[float]], List[dict]]:
        """
        Updates kinematic state for the current frame.

        Args:
            current_coords:    Mapping of {track_id: (real_x, real_y)}.
            timestamp:         Current video timestamp in seconds.
            frame_id:          Current video frame index.
            measurement_info:  Optional per-track quality metadata from the
                               coordinate transformer.

        Returns:
            state_vectors  — {track_id: [x, y, v_x, v_y, a_x, a_y]}
            synthetic_rows — list of interpolated gap-fill dicts for DuckDB.
        """
        active_ids = set(current_coords.keys())
        synthetic_rows: List[dict] = []
        state_quality: Dict[int, dict] = {}
        measurement_info = measurement_info or {}

        for tid in list(self._missed.keys()):
            if tid not in active_ids:
                self._missed[tid] += 1

        stale_ids = [
            tid for tid, count in self._missed.items()
            if count > self.max_missed_frames
        ]
        for tid in stale_ids:
            del self._missed[tid]
            self.trajectories.pop(tid, None)
            self._meta.pop(tid, None)
            self.warm_tracks.discard(tid)
            self._last_timestamp.pop(tid, None)
            self._last_frame_id.pop(tid, None)
            self._last_trusted_motion.pop(tid, None)

        state_vectors: Dict[int, List[float]] = {}

        for track_id, (x, y) in current_coords.items():
            gap = self._missed.get(track_id, 0)
            last_ts = self._last_timestamp.get(track_id)
            last_fid = self._last_frame_id.get(track_id)

            if gap > 0 and len(self.trajectories[track_id]) > 0 and last_ts is not None:
                last_x, last_y = self.trajectories[track_id][-1]
                actual_gap = max(
                    0,
                    (
                        (frame_id - last_fid) // self.frame_step - 1
                        if last_fid is not None else gap
                    ),
                )
                for i in range(actual_gap):
                    alpha = (i + 1) / (actual_gap + 1)
                    syn_x = last_x + alpha * (x - last_x)
                    syn_y = last_y + alpha * (y - last_y)
                    syn_timestamp = last_ts + (i + 1) * self.dt
                    syn_frame_id = (
                        last_fid + (i + 1) * self.frame_step
                        if last_fid is not None else frame_id
                    )
                    syn_info = self._clean_measurement_info(
                        {
                            "measurement_used": False,
                            "predicted_only": False,
                            "association_iou": None,
                            "strong_association": False,
                            "inside_calibration_core": False,
                        },
                        interpolated=True,
                    )
                    self._append_sample(
                        track_id, syn_x, syn_y, syn_timestamp, syn_frame_id, syn_info
                    )
                    syn_state, syn_quality = self._build_state(track_id, syn_info)
                    synthetic_rows.append({
                        "timestamp": syn_timestamp,
                        "frame_id": syn_frame_id,
                        "track_id": track_id,
                        "pos_x": syn_state[0],
                        "pos_y": syn_state[1],
                        "vel_x": syn_state[2],
                        "vel_y": syn_state[3],
                        "accel_x": syn_state[4],
                        "accel_y": syn_state[5],
                        **syn_quality,
                    })

            self._missed[track_id] = 0
            current_info = self._clean_measurement_info(
                measurement_info.get(track_id),
                interpolated=False,
            )
            self._append_sample(track_id, x, y, timestamp, frame_id, current_info)
            state_vec, quality = self._build_state(track_id, current_info)
            state_vectors[track_id] = state_vec
            state_quality[track_id] = quality
            self._last_timestamp[track_id] = timestamp
            self._last_frame_id[track_id] = frame_id

        self.last_state_quality = state_quality
        return state_vectors, synthetic_rows
