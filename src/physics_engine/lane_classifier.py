"""Lane classifier — determines which road lane a vehicle occupies.

Setup (run once)
----------------
Call ``detect_and_save_lanes(frame, H, output_path)`` on a single video
frame.  It warps the frame to a bird's-eye view using the homography,
detects white lane markings via Hough lines, and saves the resulting
world-Y lane boundaries to a JSON file.

Runtime
-------
Load ``LaneClassifier.load(path)`` and call ``classify(world_y)`` for
each tracked vehicle to get its lane index (0-based, left to right in
world Y space).

World Y coordinates
-------------------
The homography maps pixel → world where Y runs across the road width
(0 m at one kerb, ~10.4 m at the other kerb for session6_center).
Lane boundaries are stored as a sorted list of Y thresholds.

  boundaries = [5.2]        → lane 0: Y < 5.2,  lane 1: Y ≥ 5.2
  boundaries = [3.5, 6.9]   → lane 0: Y < 3.5,  lane 1: 3.5 ≤ Y < 6.9,
                                lane 2: Y ≥ 6.9
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Pixels per metre in the bird's-eye warp used for lane detection.
_PIXELS_PER_METRE: int = 50


def _warp_to_bev(
    frame: np.ndarray,
    H: np.ndarray,
    x_range: tuple[float, float] = (0.0, 30.0),
    y_range: tuple[float, float] = (0.0, 11.0),
) -> tuple[np.ndarray, float]:
    """Warp a camera frame to a bird's-eye view using the homography.

    The output image has PIXELS_PER_METRE pixels per metre in world space,
    with world X running horizontally and world Y running vertically.

    Args:
        frame  : BGR camera frame.
        H      : (3, 3) pixel-to-world homography.
        x_range: (x_min, x_max) world metres to include horizontally.
        y_range: (y_min, y_max) world metres to include vertically.

    Returns:
        bev_frame : Warped bird's-eye view image.
        ppm       : Pixels per metre used (for converting pixel Y back to world Y).
    """
    ppm    = float(_PIXELS_PER_METRE)
    w_out  = int((x_range[1] - x_range[0]) * ppm)
    h_out  = int((y_range[1] - y_range[0]) * ppm)

    # Compose: (pixel → world in metres) then (world → scaled output pixel).
    # Output pixel = scale @ translate @ H @ input_pixel
    offset = np.array([
        [1, 0, -x_range[0]],
        [0, 1, -y_range[0]],
        [0, 0,  1          ],
    ], dtype=np.float64)
    scale  = np.diag([ppm, ppm, 1.0])
    M      = scale @ offset @ H

    bev = cv2.warpPerspective(frame, M, (w_out, h_out))
    return bev, ppm


def _detect_lane_y_boundaries(
    bev: np.ndarray,
    ppm: float,
    y_range: tuple[float, float],
    peak_width_max_m: float = 0.6,
    edge_margin_m: float = 3.5,
    min_peak_strength: float = 10_000.0,
    merge_gap_m: float = 1.0,
) -> List[float]:
    """Detect lane-dividing line Y positions from a bird's-eye view.

    Uses a row-brightness profile rather than Hough lines.  This avoids
    false positives from wide road-edge barriers (which appear as broad
    bright bands in the BEV) and correctly finds narrow dashed lane
    markings (which appear as sharp, narrow peaks in the row profile).

    Algorithm:
      1. Threshold the BEV to isolate bright pixels.
      2. Sum bright pixels row-by-row → 1-D profile along world Y axis.
      3. Smooth the profile to separate structural peaks from noise.
      4. Find peaks whose *width* is narrow (lane paint) vs wide (barrier).
      5. Cluster neighbouring peak rows and return their world-Y means.

    Args:
        bev             : Bird's-eye view image (from _warp_to_bev).
        ppm             : Pixels per metre.
        y_range         : (y_min, y_max) world Y extent of bev rows.
        peak_width_max_m: Maximum width (metres) of a valid lane marking peak.
                          Barriers are wider; dashed lines are narrower.
        edge_margin_m      : Ignore the first/last N metres of world Y — these
                             are road barriers and kerbs, not lane dividers.
                             3.5 m excludes the bright barrier on the left.
        min_peak_strength  : Minimum row-sum of bright pixels at a peak centre.
                             Filters out faint noise; real lane paint > 10 000.
        merge_gap_m        : Peaks closer than this are merged (same stripe).

    Returns:
        Sorted list of world-Y lane boundary positions in metres.
    """
    gray  = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Row-wise sum of bright pixels (1-D profile along world Y).
    profile = bright.sum(axis=1).astype(np.float32)

    # Adaptive threshold: a row is "bright" if it exceeds the background.
    # Use a wide Gaussian blur as the background estimate.
    bg_ksize = int(ppm * 2) | 1          # ~2 m kernel, must be odd
    background = cv2.GaussianBlur(
        profile.reshape(-1, 1), (1, bg_ksize), 0
    ).ravel()
    above_bg = (profile - background) > 0

    # Find connected runs of above-background rows.
    boundaries: List[float] = []
    in_peak   = False
    peak_start = 0
    max_width_px = int(peak_width_max_m * ppm)
    edge_px_lo   = int(edge_margin_m * ppm)
    edge_px_hi   = bev.shape[0] - edge_px_lo

    for row, is_bright in enumerate(above_bg):
        if is_bright and not in_peak:
            peak_start = row
            in_peak    = True
        elif not is_bright and in_peak:
            peak_end   = row
            width_px   = peak_end - peak_start
            centre_px  = (peak_start + peak_end) / 2.0
            in_peak    = False

            # Accept narrow peaks (lane paint) that are strong enough and
            # away from road edges (barriers register as wide bright zones).
            peak_max = float(profile[peak_start:row].max())
            if (width_px <= max_width_px
                    and edge_px_lo < centre_px < edge_px_hi
                    and peak_max >= min_peak_strength):
                world_y = centre_px / ppm + y_range[0]
                boundaries.append(round(world_y, 2))

    # Merge boundaries closer than merge_gap_m (two edges of same stripe).
    if not boundaries:
        return []

    merged: List[float] = [boundaries[0]]
    for y in boundaries[1:]:
        if y - merged[-1] < merge_gap_m:
            merged[-1] = round((merged[-1] + y) / 2.0, 2)
        else:
            merged.append(y)

    return sorted(merged)


def detect_and_save_lanes(
    frame: np.ndarray,
    H: np.ndarray,
    output_path: str = "lane_config.json",
    x_range: tuple[float, float] = (0.0, 30.0),
    y_range: tuple[float, float] = (0.0, 11.0),
) -> List[float]:
    """Detect lane boundaries from a single frame and save to JSON.

    Run this once as a setup step.  The result is loaded by
    ``LaneClassifier.load()`` at pipeline startup.

    Args:
        frame      : A representative BGR video frame with visible lane markings.
        H          : (3, 3) pixel-to-world homography from calibration.yaml.
        output_path: Where to save the lane_config.json.
        x_range    : World X extent for the bird's-eye warp.
        y_range    : World Y extent for the bird's-eye warp.

    Returns:
        Detected world-Y lane boundaries (also saved to output_path).
    """
    bev, ppm       = _warp_to_bev(frame, H, x_range, y_range)
    boundaries     = _detect_lane_y_boundaries(bev, ppm, y_range)

    config = {
        "lane_boundaries_world_y": boundaries,
        "y_range": list(y_range),
        "n_lanes": len(boundaries) + 1,
    }
    Path(output_path).write_text(json.dumps(config, indent=2))
    return boundaries


class LaneClassifier:
    """Classifies a vehicle world position to a lane index (0-based).

    Lane 0 is the lowest world-Y lane (one side of the road).
    Indices increase with world Y.

    Example:
        boundaries = [5.2]
        classify(3.0) → 0  (lane 0)
        classify(7.5) → 1  (lane 1)
    """

    def __init__(self, boundaries: List[float]) -> None:
        """
        Args:
            boundaries: Sorted list of world-Y thresholds separating lanes.
        """
        self.boundaries: List[float] = sorted(boundaries)

    def classify(self, world_y: float) -> int:
        """Return the lane index (0-based) for a world Y coordinate.

        Args:
            world_y: Vehicle world Y position in metres.

        Returns:
            Lane index integer.
        """
        for i, threshold in enumerate(self.boundaries):
            if world_y < threshold:
                return i
        return len(self.boundaries)

    def n_lanes(self) -> int:
        """Total number of lanes (one more than the number of boundaries)."""
        return len(self.boundaries) + 1

    def save(self, path: str) -> None:
        """Persist lane boundaries to JSON."""
        config = {
            "lane_boundaries_world_y": self.boundaries,
            "n_lanes": self.n_lanes(),
        }
        Path(path).write_text(json.dumps(config, indent=2))

    @classmethod
    def from_lane_width(
        cls,
        n_lanes: int,
        lane_width_m: float,
        start_y_m: float = 0.0,
        save_path: Optional[str] = None,
    ) -> "LaneClassifier":
        """Build a LaneClassifier from known equal-width lane geometry.

        This is the preferred method when the lane width is measured in the
        field (e.g. 3.5 m per lane).  It is more reliable than image-based
        detection because it uses the calibrated homography coordinate system
        directly.

        Boundaries are placed at:
            start_y_m + lane_width_m,
            start_y_m + 2 * lane_width_m,
            ...  (n_lanes - 1 boundaries total)

        Args:
            n_lanes      : Total number of lanes (e.g. 3).
            lane_width_m : Width of each lane in world metres (e.g. 3.5).
            start_y_m    : World Y of the first lane's left edge (default 0.0).
            save_path    : If given, persist the boundaries to this JSON path.

        Returns:
            LaneClassifier with (n_lanes - 1) equally-spaced boundaries.

        Example:
            LaneClassifier.from_lane_width(3, 3.5)
            # → boundaries = [3.5, 7.0]
            # → Lane 0: Y < 3.5 m
            # → Lane 1: 3.5 ≤ Y < 7.0 m
            # → Lane 2: Y ≥ 7.0 m
        """
        boundaries = [
            round(start_y_m + (i + 1) * lane_width_m, 3)
            for i in range(n_lanes - 1)
        ]
        obj = cls(boundaries)
        if save_path:
            obj.save(save_path)
        return obj

    @classmethod
    def load(cls, path: str) -> "LaneClassifier":
        """Load a LaneClassifier from a JSON file.

        Args:
            path: Path to lane_config.json.

        Raises:
            FileNotFoundError: If the file does not exist.
                Run from_lane_width() or from_frame() first.
        """
        data       = json.loads(Path(path).read_text())
        boundaries = data["lane_boundaries_world_y"]
        return cls(boundaries)

    @classmethod
    def from_frame(
        cls,
        frame: np.ndarray,
        H: np.ndarray,
        save_path: Optional[str] = None,
        y_range: tuple[float, float] = (0.0, 11.0),
        x_range: tuple[float, float] = (0.0, 30.0),
    ) -> "LaneClassifier":
        """Convenience: detect lanes from a frame and return a LaneClassifier.

        Optionally saves the result so it can be loaded on the next run.

        Args:
            frame    : BGR video frame with visible lane markings.
            H        : Homography matrix (pixel → world).
            save_path: If given, persist the detected boundaries here.
            y_range  : (y_min, y_max) world Y extent in metres. Adjust to
                       match your road width; default 0–11 m covers two lanes.
            x_range  : (x_min, x_max) world X extent for the BEV warp.

        Returns:
            LaneClassifier instance.
        """
        bev, ppm   = _warp_to_bev(frame, H, x_range=x_range, y_range=y_range)
        boundaries = _detect_lane_y_boundaries(bev, ppm, y_range)
        obj        = cls(boundaries)
        if save_path:
            obj.save(save_path)
        return obj
