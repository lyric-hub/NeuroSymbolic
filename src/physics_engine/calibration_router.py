"""
Calibration API Router.

Provides endpoints for the web-based homography calibration tool:
  GET  /calibrate/videos          — lists available videos in data/raw_videos/
  GET  /calibrate/frame           — extracts a single frame as a base64 JPEG
  POST /calibrate/compute         — computes the homography matrix, saves
                                    calibration.yaml, and returns per-point
                                    reprojection errors.
  GET  /calibrate/status          — checks whether calibration.yaml exists.
  POST /calibrate/import-kml      — parses a Google Earth KML/KMZ export and
                                    converts Placemark lat/lon to local metres
                                    (ready to use as world_points).
"""

import base64
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
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
    Returns a list of video paths (relative to data/raw_videos/) found
    recursively inside that directory.  Subdirectories are included so
    videos stored in data/raw_videos/ulloor/ etc. are all discoverable.
    """
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    videos = [
        str(p.relative_to(VIDEOS_DIR))
        for p in sorted(VIDEOS_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
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


# ---------------------------------------------------------------------------
# KML import — Google Earth placemark → local metric world coordinates
# ---------------------------------------------------------------------------

class KMLPoint(BaseModel):
    """One Google Earth placemark converted to local metric ground coordinates."""
    name: str
    lat: float
    lon: float
    world_x: float   # metres East  from origin placemark
    world_y: float   # metres North from origin placemark


class KMLImportResponse(BaseModel):
    origin_name: str
    origin_lat: float
    origin_lon: float
    points: List[KMLPoint]


def _latlon_to_local_metres(
    lat: float, lon: float,
    origin_lat: float, origin_lon: float,
) -> tuple[float, float]:
    """
    Flat-earth approximation: converts (lat, lon) to metres relative to an
    origin point. Accurate to < 0.1 % for areas smaller than ~1 km² (any
    single intersection or road segment).

    Returns (east_metres, north_metres).
    """
    d_lat = lat - origin_lat
    d_lon = lon - origin_lon
    north = d_lat * 111_320.0
    east  = d_lon * 111_320.0 * math.cos(math.radians(origin_lat))
    return round(east, 4), round(north, 4)


def _parse_kml_bytes(kml_bytes: bytes) -> list[tuple[str, float, float]]:
    """
    Parses raw KML XML and returns a list of (name, lat, lon) for every
    Placemark that contains a Point coordinate.  Handles both the default
    KML namespace and namespace-free documents.
    """
    root = ET.fromstring(kml_bytes)

    # KML namespace varies; try with and without.
    NS = "http://www.opengis.net/kml/2.2"
    ns_map = {"kml": NS}

    placemarks = root.findall(".//kml:Placemark", ns_map)
    if not placemarks:
        placemarks = root.findall(".//Placemark")   # no-namespace fallback

    results: list[tuple[str, float, float]] = []
    for pm in placemarks:
        # Name
        name_el = pm.find("kml:name", ns_map) or pm.find("name")
        name = name_el.text.strip() if name_el is not None and name_el.text else f"pt{len(results)}"

        # Coordinates  — "lon,lat[,alt]"
        coord_el = (
            pm.find(".//kml:Point/kml:coordinates", ns_map)
            or pm.find(".//Point/coordinates")
        )
        if coord_el is None or not coord_el.text:
            continue

        parts = coord_el.text.strip().split(",")
        if len(parts) < 2:
            continue

        lon, lat = float(parts[0]), float(parts[1])
        results.append((name, lat, lon))

    return results


@router.post("/import-kml", response_model=KMLImportResponse)
async def import_kml(file: UploadFile = File(...)) -> KMLImportResponse:
    """
    Accepts a Google Earth KML or KMZ export and converts every Placemark
    to local metric (East, North) coordinates relative to the first placemark
    (which becomes the coordinate origin).

    Workflow:
      1. In Google Earth, place one Placemark on each identifiable road feature
         visible in your video frame (lane markings, kerb corners, stop lines).
      2. Export as KML (File → Save Place As → .kml) or KMZ.
      3. Upload here.  The response gives world_x / world_y in metres for each
         placemark — paste these as world_points in the /calibrate/compute call.

    The first placemark in the file becomes (0, 0).  If you want a specific
    point as origin, name it "origin" and it will be used preferentially.
    """
    raw = await file.read()

    # KMZ is a ZIP that contains doc.kml
    filename = (file.filename or "").lower()
    if filename.endswith(".kmz"):
        try:
            with zipfile.ZipFile(BytesIO(raw)) as zf:
                kml_name = next(
                    (n for n in zf.namelist() if n.lower().endswith(".kml")),
                    None,
                )
                if kml_name is None:
                    raise HTTPException(status_code=422, detail="No .kml file found inside KMZ.")
                raw = zf.read(kml_name)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=422, detail="Uploaded file is not a valid KMZ archive.")

    try:
        placemarks = _parse_kml_bytes(raw)
    except ET.ParseError as exc:
        raise HTTPException(status_code=422, detail=f"KML parse error: {exc}")

    if not placemarks:
        raise HTTPException(
            status_code=422,
            detail="No Point Placemarks found in the KML. Make sure you placed markers (not paths or polygons).",
        )

    # Choose origin: prefer a placemark explicitly named "origin", else use first.
    origin_idx = next(
        (i for i, (n, _, _) in enumerate(placemarks) if n.lower() == "origin"),
        0,
    )
    origin_name, origin_lat, origin_lon = placemarks[origin_idx]

    points: list[KMLPoint] = []
    for name, lat, lon in placemarks:
        east, north = _latlon_to_local_metres(lat, lon, origin_lat, origin_lon)
        points.append(KMLPoint(
            name=name,
            lat=lat,
            lon=lon,
            world_x=east,
            world_y=north,
        ))

    return KMLImportResponse(
        origin_name=origin_name,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        points=points,
    )
