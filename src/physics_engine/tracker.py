# src/physics_engine/tracker.py
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union, Literal

import numpy as np
import torch
import yaml  # type: ignore
from boxmot import TRACKERS, create_tracker  # type: ignore
from boxmot.utils import TRACKER_CONFIGS  # type: ignore

from src.utils import get_device

# --- Types ---
PathLike = Union[str, Path]
ReIdArg = Union[PathLike, Literal["auto"], None]
DeviceArg = Optional[Union[str, int, torch.device]]

_APPEARANCE_TRACKERS = {
    "deepocsort",
    "strongsort",
    "botsort",
    "hybridsort",
    "boosttrack",
    "imprassoc",
}

_DEFAULT_REID = os.getenv("BOXMOT_DEFAULT_REID", "osnet_x0_25_msmt17.pt")


def _resolve_device_str(device_arg: DeviceArg = None) -> str:
    """Resolves the provided device argument to a string compatible with BoxMOT."""
    if device_arg is None or (isinstance(device_arg, str) and device_arg == "auto"):
        device_obj = get_device()
    elif isinstance(device_arg, torch.device):
        device_obj = device_arg
    elif isinstance(device_arg, int):
        device_obj = torch.device(f"cuda:{device_arg}")
    else:
        device_string_lower = str(device_arg).strip().lower()
        if device_string_lower == "cpu":
            device_obj = torch.device("cpu")
        elif device_string_lower in ("cuda", "gpu"):
            device_obj = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        elif device_string_lower.startswith("cuda:"):
            device_obj = torch.device(device_string_lower)
        elif device_string_lower.isdigit():
            device_obj = torch.device(f"cuda:{device_string_lower}")
        else:
            device_obj = get_device()

    if device_obj.type == "cpu":
        return "cpu"
    return str(device_obj.index) if device_obj.index is not None else "0"


def _to_path(path_like_input: Optional[PathLike]) -> Optional[Path]:
    return None if path_like_input is None else Path(path_like_input)


@lru_cache(maxsize=64)
def _load_yaml(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _deep_find_reid_value(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and "reid" in k.lower():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            found = _deep_find_reid_value(v)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _deep_find_reid_value(item)
            if found:
                return found
    return None


def _deep_merge(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            _deep_merge(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def load_tracker(
    name: str,
    *,
    config: Optional[PathLike] = None,
    config_dir: Optional[PathLike] = None,
    reid: ReIdArg = "auto",
    device: DeviceArg = None,
    half: bool = False,
    per_class: bool = False,
    warmup: bool = True,
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Factory function to create a BoxMOT tracker instance."""
    method = name.strip().lower()
    if method not in TRACKERS:
        raise ValueError(f"Unsupported tracker {name!r}. Supported: {sorted(TRACKERS)}")

    if config is not None:
        cfg_path = Path(config)
    else:
        cfg_dir = Path(config_dir) if config_dir is not None else Path(TRACKER_CONFIGS)
        cfg_path = cfg_dir / f"{method}.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {cfg_path}")

    cfg_for_create = cfg_path
    tmp_path: Optional[Path] = None

    if config_overrides:
        base = dict(_load_yaml(str(cfg_path)))
        merged = _deep_merge(base, dict(config_overrides))

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            yaml.safe_dump(merged, tmp, sort_keys=False)
            tmp.flush()
            tmp_path = Path(tmp.name)
            cfg_for_create = tmp_path
        finally:
            tmp.close()

    dev_str = _resolve_device_str(device)
    reid_weights: Optional[Path] = None

    if reid == "auto":
        if method in _APPEARANCE_TRACKERS:
            config_y = _load_yaml(str(cfg_path))
            val = _deep_find_reid_value(config_y) or _DEFAULT_REID
            candidate = Path(val)

            if not candidate.exists():
                raise FileNotFoundError(
                    f"ReID weights not found: {candidate}\n"
                    "Set BOXMOT_DEFAULT_REID env var or pass reid= explicitly."
                )
            reid_weights = candidate

    elif reid is None:
        reid_weights = None
    else:
        candidate = Path(reid)
        if not candidate.exists():
            raise FileNotFoundError(f"ReID weights not found: {candidate}")
        reid_weights = candidate

    try:
        tracker = create_tracker(
            method,
            cfg_for_create,
            reid_weights,
            dev_str,
            half,
            per_class,
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if warmup:
        tracker_model = getattr(tracker, "model", None)
        if tracker_model is not None and hasattr(tracker_model, "warmup"):
            tracker_model.warmup()

    return tracker


class VehicleTracker:
    """
    Stateful execution wrapper for processing video frames iteratively.
    Connects the YoloDetector output to the BoxMOT tracking algorithms.
    """
    def __init__(self, tracker_name: str = "bytetrack", device: DeviceArg = None):
        """
        Initializes the tracking execution loop. ByteTrack is the default tracker —
        it is fast, appearance-free (no ReID weights required), and robust for
        dense traffic scenes.
        """
        self.tracker = load_tracker(name=tracker_name, device=device, half=True)
    
    def update(self, raw_detections: Any, frame: np.ndarray) -> np.ndarray:
        """
        Takes raw predictions from `detector.py` and the current video frame (H, W, C),
        and updates the tracker state.
        
        Args:
            raw_detections: Ultralytics Results object or formatted array [x1, y1, x2, y2, conf, cls]
            frame: Raw numpy array of the current video frame.
            
        Returns:
            np.ndarray: Tracked bounding boxes in the format
                [x1, y1, x2, y2, track_id, conf, cls, ind] — shape (N, 8).
                ``ind`` is the index of the matched detection in the input array.
        """
        # Ensure detections are in the required numpy format for BoxMOT: (N, 6)
        # Expected format per row: [x1, y1, x2, y2, conf, cls]
        if hasattr(raw_detections, "boxes"):
            # Extract from Ultralytics Results object
            dets_np = raw_detections.boxes.data.cpu().numpy()
        else:
            dets_np = np.array(raw_detections)

        # If no detections exist in the frame, return an empty array
        if len(dets_np) == 0:
            return np.empty((0, 8))

        # Update tracker (BoxMOT handles the Kalman filtering and ReID matching internally here)
        tracked_objects = self.tracker.update(dets_np, frame)

        # tracked_objects format: [x1, y1, x2, y2, track_id, conf, cls, ind]
        return tracked_objects