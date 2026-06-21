"""Zone-based detection blur.

Blurs everything outside the active zone + a 5 m entry buffer so YOLO
focuses on the measurement area and the approach corridor where new
vehicles appear.

Pipeline position: applied to the raw frame BEFORE detector.predict_with_pose().
Nothing is stored — the blurred frame is used only for inference.

Entry buffer
------------
The measurement zone starts at the C0/C6 gate (world X ≈ 28 m).  Vehicles
approach from beyond that edge (higher world X).  We extend the unblurred
region 5 m upstream to catch incoming vehicles before they cross the gate.

The extension is computed in world space and projected back to pixels with
the inverse homography, so it is camera-calibration-aware and correct.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _world_to_pixel(
    H_inv: np.ndarray,
    world_pts: np.ndarray,
) -> np.ndarray:
    """Project (N, 2) world-coordinate points to pixel coordinates.

    Args:
        H_inv    : (3, 3) inverse homography  (world → pixel).
        world_pts: (N, 2) float array of (world_x, world_y) in metres.

    Returns:
        (N, 2) integer pixel coordinates.
    """
    pts_h  = np.array([world_pts], dtype=np.float32)   # (1, N, 2)
    pix_h  = cv2.perspectiveTransform(pts_h, H_inv.astype(np.float32))
    return pix_h[0].astype(np.int32)                   # (N, 2)


def build_entry_buffer(
    H: np.ndarray,
    named_points: List[dict],
    extension_m: float = 5.0,
) -> Optional[np.ndarray]:
    """Build a pixel polygon for the vehicle entry corridor.

    Finds C0 and C6 in named_points, projects them to pixel space, then
    creates a quadrilateral [C0_px, C6_px, C6_ext_px, C0_ext_px] that
    covers the 5 m strip upstream of the measurement zone gate.

    Args:
        H            : (3, 3) homography matrix (pixel → world).
        named_points : List of calibration point dicts with keys
                       'name', 'world_x', 'world_y'.
        extension_m  : Distance to extend beyond C0/C6 in metres.

    Returns:
        (4, 2) int32 pixel polygon, or None if C0/C6 are not found.
    """
    c0 = next((p for p in named_points if p["name"] == "C0"), None)
    c6 = next((p for p in named_points if p["name"] == "C6"), None)
    if c0 is None or c6 is None:
        log.warning(
            "build_entry_buffer: C0/C6 not found in named_points %s — "
            "entry buffer disabled; approach vehicles will be blurred.",
            [p["name"] for p in named_points],
        )
        return None

    H_inv = np.linalg.inv(H)

    # C0/C6 are at world X ≈ 28 m (the far measurement line).
    # Traffic approaches from higher X, so the buffer extends in +X.
    world_pts = np.array([
        [c0["world_x"],                c0["world_y"]],   # C0
        [c6["world_x"],                c6["world_y"]],   # C6
        [c6["world_x"] + extension_m,  c6["world_y"]],   # C6 + 5 m
        [c0["world_x"] + extension_m,  c0["world_y"]],   # C0 + 5 m
    ], dtype=np.float64)

    return _world_to_pixel(H_inv, world_pts)


def apply_detection_blur(
    frame: np.ndarray,
    zone_polygon: Optional[List[Tuple[float, float]]],
    entry_buffer: Optional[np.ndarray],
    blur_ksize: int = 51,
) -> np.ndarray:
    """Return a copy of frame with everything outside the detection zone blurred.

    The detection zone is the union of:
      - zone_polygon  : user-defined measurement zone (pixel coordinates)
      - entry_buffer  : 5 m approach corridor beyond C0/C6 (pixel coordinates)

    A Gaussian blur with kernel size ``blur_ksize`` is applied to masked-out
    pixels.  This directs YOLO's attention to the relevant area without
    cropping or distorting the image geometry.

    Args:
        frame        : BGR image (not modified in-place).
        zone_polygon : List of (x, y) pixel vertices from zone_config.json.
                       Pass None to skip blurring.
        entry_buffer : (4, 2) int32 polygon from build_entry_buffer().
                       Pass None to blur only outside the zone with no buffer.
        blur_ksize   : Gaussian kernel size (must be odd and positive).

    Returns:
        New BGR frame with blur applied outside the unmasked region.
    """
    if zone_polygon is None:
        return frame.copy()

    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    h, w  = frame.shape[:2]
    mask  = np.zeros((h, w), dtype=np.uint8)

    # Fill zone polygon
    zone_pts = np.array(zone_polygon, dtype=np.int32)
    cv2.fillPoly(mask, [zone_pts], 255)

    # Fill entry buffer
    if entry_buffer is not None:
        cv2.fillPoly(mask, [entry_buffer.reshape(-1, 1, 2)], 255)

    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    result  = frame.copy()
    result[mask == 0] = blurred[mask == 0]
    return result
