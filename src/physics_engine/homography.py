"""
Homography Coordinate Transformer.

Loads the pre-computed homography matrix from calibration.yaml (produced by
the web-based calibration tool at /calibrate-ui) and maps detected vehicle
positions to real-world ground-plane coordinates during the micro-loop.

Ground-plane projection strategy:
  kp31 (rear wheel bottom) from YOLO-Pose is the sole reference point when
  pose data is available.  kp31 lies on z=0 (the road surface), which is the
  only plane where planar homography H is geometrically valid.

Dual-stage filtering:
  Stage 1 — Kalman filter in pixel space (per-track, constant-velocity model).
    Smooths heatmap localization noise before the perspective transform so H
    receives a cleaner input.  For frames where kp31 is undetected, the
    Kalman prediction bridges the gap rather than forcing an interpolation in
    world space.  Predictions are trusted for at most _MAX_KP31_PREDICT_FRAMES
    consecutive frames; beyond that the state resets and KinematicEstimator's
    SG gap-fill takes over.

  Stage 2 — Savitzky-Golay filter in world space (KinematicEstimator).
    Smooths residual world-coordinate jitter and provides clean velocity and
    acceleration derivatives.

  When no pose model is supplied (pose_detections=None), bbox bottom-center
  is used as a consistent fallback for every track in that run.
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rear wheel bottom keypoint index in the 33-keypoint YOLO-Pose model.
# This is the only keypoint physically on the road surface (z=0), making
# planar homography projection geometrically valid.
_GROUND_KP_IDX: int = 31
_KP_CONF_THRESH: float = 0.5

# Minimum IoU between a ByteTrack bbox and a pose detection bbox to accept
# the pose detection as belonging to that track.  Values below this mean the
# two boxes are essentially non-overlapping — no match is recorded and the
# track is treated as a missed detection for that frame.
# 0.25 requires meaningful overlap before accepting a kp31; 0.10 was too
# permissive in dense traffic where adjacent vehicles share ~10% IoU.
_MIN_IOU_MATCH: float = 0.25
_TRUSTED_IOU_MATCH: float = 0.3

# Kalman filter parameters for kp31 pixel-space smoothing.
# Q_pos: process noise on position (pixels²) — small because the prediction
#        x' = x + vx is accurate for smooth vehicle motion.
# Q_vel: process noise on velocity (pixels²/frame²) — larger to accommodate
#        vehicle acceleration and deceleration between frames.
# R:     measurement noise (pixels²) — based on typical heatmap keypoint
#        localization error of ~2-3 pixels RMS (σ² ≈ 5 px²).
_KF_PROCESS_NOISE_POS: float = 1e-2
_KF_PROCESS_NOISE_VEL: float = 1.0
_KF_MEASURE_NOISE: float = 5.0

# Maximum consecutive frames where kp31 is absent and the Kalman filter
# predicts without a measurement.  Beyond this the filter state resets and
# the track falls through to KinematicEstimator's linear interpolation.
# 8 frames at 30 fps ≈ 0.27 s — covers most short occlusions cleanly.
_MAX_KP31_PREDICT_FRAMES: int = 8

# Level 2 reference-point correction (bbox rear-bottom → vehicle centre).
# For this camera geometry, y2 maps to the vehicle's REAR ground contact.
# Shifting by half the vehicle length in the travel direction moves the
# reference to the vehicle's midpoint, cutting positional error from ~2 m
# to near zero without any additional model or retraining.
_VEHICLE_HALF_LENGTH_M: float = 2.25   # half of 4.5 m average vehicle length
_LEVEL2_MIN_SPEED_MS: float   = 0.5    # m/s — below this, skip correction


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _compute_iou(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    """Return the Intersection-over-Union of two axis-aligned rectangles."""
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CoordinateTransformer:
    """
    Stateful coordinate transformer: loads the homography matrix and maps
    detected vehicle positions to real-world ground-plane coordinates.

    Maintains a per-track Kalman filter for kp31 pixel-space smoothing
    (Stage 1 of the dual-stage filtering pipeline).
    """

    def __init__(self, calibration_file: str = "calibration.yaml") -> None:
        """
        Loads the homography matrix from the calibration YAML file.

        Args:
            calibration_file: Path to the calibration YAML produced by the
                web calibration tool. Defaults to 'calibration.yaml'.

        Raises:
            FileNotFoundError: If the calibration file does not exist.
                Run the web calibration tool at /calibrate-ui first.
        """
        try:
            with open(calibration_file, "r") as f:
                calib_data = yaml.safe_load(f)
            self.H = np.array(calib_data["homography"], dtype=np.float32)
            ref = calib_data.get("reference_point", {})
            self.reference_name: str = ref.get("name", "Origin")
            # All named landmarks — used for nearest-point position descriptions.
            self.named_points: list = calib_data.get("named_points", [])
            self.world_points: list = calib_data.get("world_points", [])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Calibration file '{calibration_file}' not found. "
                "Run the web calibration tool at /calibrate-ui first."
            )

        # Valid world-coordinate bounds derived from the calibration control
        # points.  Planar homography is only accurate within the convex hull
        # of those points; extrapolation beyond it grows nonlinearly and
        # produces incorrect world positions and therefore incorrect velocities.
        # A 2 m buffer on every side accommodates vehicles that are partially
        # crossing the zone boundary.
        _WORLD_BUFFER_M: float = 2.0
        if self.named_points:
            _xs = [p["world_x"] for p in self.named_points]
            _ys = [p["world_y"] for p in self.named_points]
            self._valid_world_x: Tuple[float, float] = (
                min(_xs) - _WORLD_BUFFER_M,
                max(_xs) + _WORLD_BUFFER_M,
            )
            self._valid_world_y: Tuple[float, float] = (
                min(_ys) - _WORLD_BUFFER_M,
                max(_ys) + _WORLD_BUFFER_M,
            )
        else:
            self._valid_world_x = (-float("inf"), float("inf"))
            self._valid_world_y = (-float("inf"), float("inf"))

        self._world_core_polygon: Optional[np.ndarray] = None
        core_points = self.world_points or [
            [p["world_x"], p["world_y"]] for p in self.named_points
        ]
        if len(core_points) >= 3:
            core_arr = np.array(core_points, dtype=np.float32).reshape(-1, 1, 2)
            self._world_core_polygon = cv2.convexHull(core_arr)

        # Per-track Kalman filter state for kp31 pixel-space smoothing.
        self._kf_filters: Dict[int, cv2.KalmanFilter] = {}
        # Consecutive prediction-only frames per track (no kp31 measurement).
        self._kf_predict_count: Dict[int, int] = {}
        self.last_measurement_quality: Dict[int, dict] = {}
        self.last_kf_predicted_ids: set = set()

        # Level 2: inverse homography (world → pixel) and per-track velocity
        # cache used to shift bbox rear-bottom to vehicle centre each frame.
        self.H_inv: np.ndarray = np.linalg.inv(self.H).astype(np.float32)
        self._level2_velocities: Dict[int, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Kalman filter lifecycle
    # ------------------------------------------------------------------

    def _init_kf(self, x: float, y: float) -> cv2.KalmanFilter:
        """
        Initialise a constant-velocity Kalman filter for kp31 at pixel (x, y).

        State vector: [kp_x, kp_y, vel_x, vel_y]
        Measurement:  [kp_x, kp_y]
        """
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.diag([
            _KF_PROCESS_NOISE_POS,
            _KF_PROCESS_NOISE_POS,
            _KF_PROCESS_NOISE_VEL,
            _KF_PROCESS_NOISE_VEL,
        ]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * _KF_MEASURE_NOISE
        # High initial position uncertainty forces K_pos ≈ 0.95 on the first
        # correction so the filter tracks the raw measurement immediately.
        # With P_pos=1 (identity), K_pos = 1/(1+5) ≈ 0.17 — the corrected
        # pixel lags far behind the actual kp31 for ~10 frames, which the
        # downstream SG differentiator interprets as a velocity spike when
        # the filter finally catches up.  P_pos=500 → K_pos ≈ 0.99 on
        # frame 2, so there is no lag and no spike.
        kf.errorCovPost = np.diag([500.0, 500.0, 10.0, 10.0]).astype(np.float32)
        kf.statePost = np.array(
            [[x], [y], [0.0], [0.0]], dtype=np.float32
        )
        return kf

    def _update_kp31(
        self,
        track_id: int,
        raw_kp31: Optional[Tuple[float, float]],
    ) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        """
        Run one Kalman step for the given track.

        When raw_kp31 is detected (not None):
          - Initialise the filter on first detection.
          - On subsequent frames: predict → correct → return smoothed pixel.

        When raw_kp31 is absent (None):
          - Return None if no prior state exists (new track, or reset).
          - Otherwise: predict only → return predicted pixel.
          - After _MAX_KP31_PREDICT_FRAMES consecutive absent frames, reset
            the filter state so the track falls back to KinematicEstimator's
            linear interpolation rather than accumulating Kalman drift.

        Args:
            track_id:  Integer track identifier.
            raw_kp31:  Raw kp31 pixel from _find_kp31_pixel, or None.

        Returns:
            Smoothed or predicted (kp_x, kp_y) in pixels, or None.
        """
        if raw_kp31 is not None:
            x, y = raw_kp31
            if track_id not in self._kf_filters:
                # First detection: seed the filter and return raw position.
                # No prior velocity estimate exists yet, so correction would
                # be identical to the raw measurement — skip the KF step.
                self._kf_filters[track_id] = self._init_kf(x, y)
                self._kf_predict_count[track_id] = 0
                return x, y
            kf = self._kf_filters[track_id]
            self._kf_predict_count[track_id] = 0
            kf.predict()
            corrected = kf.correct(
                np.array([[x], [y]], dtype=np.float32)
            )
            # correct() returns shape (4, 1); index [row, 0] to get scalar.
            return float(corrected[0, 0]), float(corrected[1, 0])
        else:
            if track_id not in self._kf_filters:
                return None
            count = self._kf_predict_count.get(track_id, 0) + 1
            if count > _MAX_KP31_PREDICT_FRAMES:
                # Drift risk — reset so KinematicEstimator interpolates.
                del self._kf_filters[track_id]
                self._kf_predict_count.pop(track_id, None)
                return None
            self._kf_predict_count[track_id] = count
            predicted = self._kf_filters[track_id].predict()
            # predict() returns shape (4, 1); index [row, 0] to get scalar.
            return float(predicted[0, 0]), float(predicted[1, 0])

    # ------------------------------------------------------------------
    # kp31 detection
    # ------------------------------------------------------------------

    def _find_kp31_pixel(
        self,
        track_x1: float,
        track_y1: float,
        track_x2: float,
        track_y2: float,
        pose_boxes: np.ndarray,
        keypoints: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """
        Find the raw kp31 pixel for a tracked bbox via IoU matching.

        IoU-based: computes Intersection-over-Union between the ByteTrack bbox
        and every raw pose detection bbox, selecting the highest-IoU match.
        A minimum IoU of _MIN_IOU_MATCH is required to accept any match, which
        rejects detections that belong to adjacent vehicles in dense scenes.

        Args:
            track_x1/y1/x2/y2: ByteTrack bbox corners.
            pose_boxes        : (M, >=4) raw YOLO-Pose bbox array.
            keypoints         : (M, 33, 3) keypoint array aligned to pose_boxes.

        Returns:
            Tuple of:
              - Raw (kp31_x, kp31_y) in pixels, or None when no confident kp31
                is found.
              - The best pose/track IoU when a match was found, else None.
        """
        best_kps: Optional[np.ndarray] = None
        best_iou: float = _MIN_IOU_MATCH  # acceptance floor

        for det_box, kps in zip(pose_boxes, keypoints):
            px1, py1, px2, py2 = det_box[:4]
            iou = _compute_iou(
                track_x1, track_y1, track_x2, track_y2,
                px1, py1, px2, py2,
            )
            if iou > best_iou:
                best_iou = iou
                best_kps = kps

        if best_kps is None:
            return None, None

        kp31 = best_kps[_GROUND_KP_IDX]
        if float(kp31[2]) >= _KP_CONF_THRESH and kp31[0] > 0 and kp31[1] > 0:
            return (float(kp31[0]), float(kp31[1])), best_iou
        return None, best_iou

    def _inside_world_core(self, x: float, y: float) -> bool:
        if self._world_core_polygon is None:
            return True
        return cv2.pointPolygonTest(
            self._world_core_polygon, (float(x), float(y)), False
        ) >= 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_velocity_cache(
        self,
        state_vectors: Dict[int, List[float]],
    ) -> None:
        """Cache per-track velocities for Level 2 bbox-to-centre correction.

        Call immediately after KinematicEstimator.update() each frame.
        The stored velocities are consumed on the NEXT frame's projection
        call to shift the bbox rear-bottom reference to the vehicle centre.

        Args:
            state_vectors: {track_id: [x, y, vx, vy, ax, ay]} from KE.
        """
        active: set = set(state_vectors.keys())
        for tid, sv in state_vectors.items():
            vx, vy = float(sv[2]), float(sv[3])
            if math.hypot(vx, vy) >= _LEVEL2_MIN_SPEED_MS:
                self._level2_velocities[tid] = (vx, vy)
            else:
                self._level2_velocities.pop(tid, None)
        for tid in list(self._level2_velocities):
            if tid not in active:
                del self._level2_velocities[tid]

    def get_real_world_coords(
        self,
        tracked_boxes: np.ndarray,
        pose_detections: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[Dict[int, Tuple[float, float]], "set[int]"]:
        """
        Map each tracked vehicle to a real-world ground-plane coordinate.

        When ``pose_detections`` is provided:
          - kp31 is detected via IoU matching (_find_kp31_pixel).
          - The raw kp31 passes through _update_kp31 which runs the per-track
            Kalman filter (predict + correct when detected; predict-only when
            absent for up to _MAX_KP31_PREDICT_FRAMES frames).
          - Tracks for which no valid pixel is available (new track with absent
            kp31, or Kalman reset) are omitted so KinematicEstimator gap-fills.

        When ``pose_detections`` is None:
          - Bbox bottom-centre is used for every track (degraded mode).
          - Kalman state is not used or updated.

        Stale Kalman states (tracks no longer in tracked_boxes) are pruned
        at the start of each call to bound memory usage.

        Args:
            tracked_boxes   : Shape (N, >=5).
                              Row format: [x1, y1, x2, y2, track_id, ...]
            pose_detections : Optional (pose_boxes, keypoints) tuple where
                              pose_boxes is (M, >=4) raw YOLO-Pose bbox output
                              and keypoints is (M, 33, 3) aligned to pose_boxes.
                              Pass None to run without a pose model.

        Returns:
            Tuple of:
              real_world_positions — {track_id: (world_x, world_y)} metres.
              kf_predicted_ids — set of track_ids whose position came from a
                Kalman prediction rather than a real keypoint measurement.
            Per-track quality flags are exposed on ``self.last_measurement_quality``.
        """
        if len(tracked_boxes) == 0:
            self.last_measurement_quality = {}
            self.last_kf_predicted_ids = set()
            return {}, set()

        # Prune Kalman state for tracks ByteTrack has dropped.
        active_ids = {int(b[4]) for b in tracked_boxes}
        stale_ids = [tid for tid in self._kf_filters if tid not in active_ids]
        for tid in stale_ids:
            del self._kf_filters[tid]
            self._kf_predict_count.pop(tid, None)

        # pose_mode = True whenever the pose model is active, regardless of
        # whether YOLO found any vehicles in this specific frame.
        # Separating this from pose_boxes/keypoints being non-empty is the
        # fix for the empty-frame fallback bug: when YOLO returns 0 detections
        # but ByteTrack still has predicted tracks, we must run _update_kp31
        # in prediction-only mode rather than silently switching to bbox
        # bottom-centre (which would corrupt the reference-point consistency).
        pose_mode: bool = pose_detections is not None
        pose_boxes: Optional[np.ndarray] = None
        keypoints:  Optional[np.ndarray] = None
        if pose_mode and len(pose_detections[0]) > 0:
            pose_boxes, keypoints = pose_detections

        real_world_positions: Dict[int, Tuple[float, float]] = {}
        kf_predicted_ids: set = set()
        measurement_quality: Dict[int, dict] = {}

        for box_data in tracked_boxes:
            x1, y1, x2, y2, track_id = box_data[:5]
            track_id = int(track_id)

            if pose_mode:
                # Pose mode: find raw kp31 if YOLO produced detections this
                # frame; otherwise pass None so _update_kp31 uses its Kalman
                # prediction.  Never fall back to bbox bottom-centre — mixing
                # two reference points corrupts the trajectory signal.
                raw_kp31, best_iou = (
                    self._find_kp31_pixel(x1, y1, x2, y2, pose_boxes, keypoints)
                    if pose_boxes is not None
                    else (None, None)
                )
                result = self._update_kp31(track_id, raw_kp31)
                if result is None:
                    continue
                pixel_x, pixel_y = result
                # Track which positions came from Kalman prediction vs real
                # detection so the caller can mark them correctly in DuckDB.
                if raw_kp31 is None:
                    kf_predicted_ids.add(track_id)
                measurement_used = raw_kp31 is not None
                predicted_only = raw_kp31 is None
                association_iou = best_iou if raw_kp31 is not None else None
                strong_association = (
                    association_iou is not None and association_iou >= _TRUSTED_IOU_MATCH
                )
            else:
                # Level 2 degraded mode: project bbox rear-bottom-centre
                # through H, then shift half a vehicle length forward in the
                # travel direction to approximate the vehicle's ground-contact
                # midpoint.
                #
                # For this camera, y2 (bbox bottom) maps to the vehicle's REAR
                # footprint at z=0, not the centre.  The Level 2 correction
                # eliminates the ~2 m systematic positional offset seen with
                # plain bbox bottom-centre projection.
                pixel_x = (x1 + x2) / 2.0
                pixel_y = float(y2)
                measurement_used = True
                predicted_only = False
                association_iou = None
                strong_association = True

            pts   = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pts, self.H)
            wx    = float(world[0, 0, 0])
            wy    = float(world[0, 0, 1])

            # Level 2 centre correction (degraded mode only).
            if not pose_mode:
                vel = self._level2_velocities.get(track_id)
                if vel is not None:
                    speed = math.hypot(vel[0], vel[1])
                    wx += _VEHICLE_HALF_LENGTH_M * vel[0] / speed
                    wy += _VEHICLE_HALF_LENGTH_M * vel[1] / speed
                else:
                    # No velocity yet — default to +X (road direction for
                    # this calibration; rear-to-front is always +world_X).
                    wx += _VEHICLE_HALF_LENGTH_M

            inside_core = self._inside_world_core(wx, wy)

            # Reject positions outside the calibrated region.  The KF remains
            # active in pixel space so it is warm when the vehicle crosses in.
            if not (self._valid_world_x[0] <= wx <= self._valid_world_x[1]
                    and self._valid_world_y[0] <= wy <= self._valid_world_y[1]):
                kf_predicted_ids.discard(track_id)
                continue

            real_world_positions[track_id] = (wx, wy)
            measurement_quality[track_id] = {
                "measurement_used": measurement_used,
                "predicted_only": predicted_only,
                "association_iou": association_iou,
                "strong_association": strong_association,
                "inside_calibration_core": inside_core,
            }

        self.last_measurement_quality = measurement_quality
        self.last_kf_predicted_ids = kf_predicted_ids
        return real_world_positions, kf_predicted_ids

    def nearest_landmark(self, x: float, y: float) -> Tuple[str, float]:
        """
        Returns the name and distance (metres) of the closest named calibration
        point to the given real-world coordinate.

        Falls back to ("Origin", distance) when no named points are loaded
        (e.g. old calibration files without the named_points key).

        Args:
            x: Real-world X position (metres).
            y: Real-world Y position (metres).

        Returns:
            Tuple of (landmark_name, distance_metres).
        """
        if not self.named_points:
            dist = float(np.sqrt(x ** 2 + y ** 2))
            return self.reference_name, round(dist, 2)

        best_name, best_dist = self.reference_name, float("inf")
        for pt in self.named_points:
            dx = x - pt["world_x"]
            dy = y - pt["world_y"]
            dist = float(np.sqrt(dx ** 2 + dy ** 2))
            if dist < best_dist:
                best_dist = dist
                best_name = pt["name"]
        return best_name, round(best_dist, 2)
