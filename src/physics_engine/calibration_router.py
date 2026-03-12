"""
Calibration API Router.

Provides endpoints for the web-based homography calibration tool:
  GET  /calibrate/videos          — lists available videos in data/raw_videos/
  GET  /calibrate/frame           — extracts a single frame as a base64 JPEG
  POST /calibrate/compute         — computes the homography matrix, saves
                                    calibration.yaml, and returns per-point
                                    reprojection errors.
  GET  /calibrate/status          — checks whether calibration.yaml exists.
"""

import base64
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

VIDEOS_DIR = Path("data/raw_videos")
CALIBRATION_FILE = Path("calibration.yaml")
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}

router = APIRouter(prefix="/calibrate", tags=["calibration"])


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PointPair(BaseModel):
    """One pixel point paired with its known real-world ground coordinate."""
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float


class ComputeRequest(BaseModel):
    """Payload sent by the frontend when the user clicks Compute."""
    video_path: str
    frame_idx: int
    point_pairs: List[PointPair]


class PointError(BaseModel):
    """Per-point reprojection result returned to the frontend."""
    index: int
    pixel_x: float
    pixel_y: float
    projected_x: float
    projected_y: float
    error_px: float


class ComputeResponse(BaseModel):
    """Full response from /calibrate/compute."""
    rmse: float
    point_errors: List[PointError]
    saved_to: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_frame(video_path: str, frame_idx: int) -> tuple[np.ndarray, int]:
    """
    Opens the video, seeks to frame_idx, and returns (frame_bgr, total_frames).

    Raises:
        HTTPException: If the file cannot be opened or the frame cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clamped_idx = max(0, min(frame_idx, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Failed to read frame from video.")

    return frame, total_frames


def _frame_to_base64(frame_bgr: np.ndarray) -> str:
    """Encodes a BGR frame as a base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame as JPEG.")
    return base64.b64encode(buffer).decode("utf-8")


def _compute_reprojection_errors(
    image_pts: np.ndarray,
    world_pts: np.ndarray,
    H: np.ndarray,
) -> list[PointError]:
    """
    Projects each image point through H and computes the Euclidean distance
    to the corresponding known world point (in world-coordinate space).
    """
    errors = []
    for i, (img_pt, world_pt) in enumerate(zip(image_pts, world_pts)):
        projected = cv2.perspectiveTransform(
            img_pt.reshape(1, 1, 2).astype(np.float32), H
        )[0][0]
        error = float(np.linalg.norm(projected - world_pt))
        errors.append(
            PointError(
                index=i,
                pixel_x=float(img_pt[0]),
                pixel_y=float(img_pt[1]),
                projected_x=float(projected[0]),
                projected_y=float(projected[1]),
                error_px=round(error, 4),
            )
        )
    return errors


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/videos")
def list_videos() -> JSONResponse:
    """
    Returns a list of video filenames found in data/raw_videos/.
    Used by the frontend to populate the video selector dropdown.
    """
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    videos = [
        p.name
        for p in sorted(VIDEOS_DIR.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return JSONResponse({"videos": videos})


@router.get("/frame")
def get_frame(
    video_path: str = Query(..., description="Filename inside data/raw_videos/"),
    frame_idx: int = Query(0, ge=0, description="Zero-based frame index"),
) -> JSONResponse:
    """
    Extracts one frame from a video and returns it as a base64-encoded JPEG
    along with the total frame count (used to set the scrubber range).
    """
    full_path = str(VIDEOS_DIR / video_path)
    frame, total_frames = _extract_frame(full_path, frame_idx)
    return JSONResponse({
        "frame_b64": _frame_to_base64(frame),
        "total_frames": total_frames,
        "frame_idx": frame_idx,
        "width": frame.shape[1],
        "height": frame.shape[0],
    })


@router.post("/compute", response_model=ComputeResponse)
def compute_homography(request: ComputeRequest) -> ComputeResponse:
    """
    Computes the homography matrix from the provided pixel ↔ world point pairs,
    saves the result to calibration.yaml, and returns per-point reprojection
    errors so the user can evaluate calibration quality.

    Requires at least 4 point pairs (minimum for a valid homography).
    """
    if len(request.point_pairs) < 4:
        raise HTTPException(
            status_code=422,
            detail="At least 4 point pairs are required to compute a homography.",
        )

    image_pts = np.array(
        [[p.pixel_x, p.pixel_y] for p in request.point_pairs], dtype=np.float32
    )
    world_pts = np.array(
        [[p.world_x, p.world_y] for p in request.point_pairs], dtype=np.float32
    )

    H, mask = cv2.findHomography(image_pts, world_pts, method=cv2.RANSAC)
    if H is None:
        raise HTTPException(
            status_code=422,
            detail="Homography computation failed. Check that points are not collinear.",
        )

    point_errors = _compute_reprojection_errors(image_pts, world_pts, H)
    rmse = float(np.sqrt(np.mean([e.error_px ** 2 for e in point_errors])))

    calibration_data = {
        "homography": H.tolist(),
        "image_points": image_pts.tolist(),
        "world_points": world_pts.tolist(),
        "video_path": request.video_path,
        "frame_idx": request.frame_idx,
        "rmse_meters": round(rmse, 4),
    }
    with open(CALIBRATION_FILE, "w") as f:
        yaml.dump(calibration_data, f)

    return ComputeResponse(
        rmse=round(rmse, 4),
        point_errors=point_errors,
        saved_to=str(CALIBRATION_FILE.resolve()),
    )


@router.get("/status")
def calibration_status() -> JSONResponse:
    """
    Returns whether a valid calibration.yaml exists at the project root.
    The frontend uses this to warn the user before starting video processing.
    """
    exists = CALIBRATION_FILE.exists()
    detail = {}
    if exists:
        try:
            with open(CALIBRATION_FILE) as f:
                data = yaml.safe_load(f)
            detail = {
                "video_path": data.get("video_path"),
                "frame_idx": data.get("frame_idx"),
                "rmse_meters": data.get("rmse_meters"),
                "num_points": len(data.get("image_points", [])),
            }
        except Exception:
            pass
    return JSONResponse({"calibrated": exists, "detail": detail})
