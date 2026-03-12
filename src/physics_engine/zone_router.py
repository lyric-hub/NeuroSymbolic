"""
FastAPI router for zone configuration management.

Endpoints
---------
GET  /zone/status    — check whether zone_config.json exists + summary.
GET  /zone/config    — return the current zone_config.json contents.
POST /zone/save      — validate and save a new zone configuration.

Frame loading for the drawing UI reuses the existing calibration endpoint:
    GET /calibrate/frame?video_path=&frame_idx=
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ZONE_CONFIG_PATH = "zone_config.json"

router = APIRouter(prefix="/zone", tags=["zone"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class GateModel(BaseModel):
    name: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]


class ZoneSaveRequest(BaseModel):
    zone_id: str
    polygon: List[Tuple[float, float]]
    gates: List[GateModel]


class ZoneStatusResponse(BaseModel):
    exists: bool
    zone_id: str = ""
    gate_count: int = 0
    polygon_point_count: int = 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status", response_model=ZoneStatusResponse)
async def zone_status():
    """Returns whether a zone_config.json exists and a brief summary."""
    p = Path(ZONE_CONFIG_PATH)
    if not p.exists():
        return ZoneStatusResponse(exists=False)
    try:
        data = json.loads(p.read_text())
        return ZoneStatusResponse(
            exists=True,
            zone_id=data.get("zone_id", ""),
            gate_count=len(data.get("gates", [])),
            polygon_point_count=len(data.get("polygon", [])),
        )
    except Exception:
        return ZoneStatusResponse(exists=True)


@router.get("/config")
async def get_zone_config():
    """Returns the raw zone_config.json as a JSON object."""
    p = Path(ZONE_CONFIG_PATH)
    if not p.exists():
        raise HTTPException(status_code=404, detail="zone_config.json not found.")
    return json.loads(p.read_text())


@router.post("/save")
async def save_zone_config(request: ZoneSaveRequest):
    """
    Validates and saves the zone configuration to zone_config.json.

    Requires at least 3 polygon points and at least 1 gate.
    """
    if len(request.polygon) < 3:
        raise HTTPException(
            status_code=400,
            detail="A zone polygon needs at least 3 points.",
        )
    if len(request.gates) < 1:
        raise HTTPException(
            status_code=400,
            detail="At least one gate is required.",
        )
    if not request.zone_id.strip():
        raise HTTPException(status_code=400, detail="zone_id must not be empty.")

    data = {
        "zone_id": request.zone_id.strip(),
        "polygon": [list(pt) for pt in request.polygon],
        "gates": [
            {"name": g.name, "p1": list(g.p1), "p2": list(g.p2)}
            for g in request.gates
        ],
    }
    Path(ZONE_CONFIG_PATH).write_text(json.dumps(data, indent=2))
    return {
        "saved_to": ZONE_CONFIG_PATH,
        "zone_id": data["zone_id"],
        "gate_count": len(data["gates"]),
    }
