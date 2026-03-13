"""
Shared pytest fixtures for the TrafficAgent test suite.

All fixtures use synthetic data — no GPU, video files, or running
Ollama instance is required to run the unit tests.
"""

import math
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic kinematics helpers
# ---------------------------------------------------------------------------

def make_state_vectors(
    track_ids: List[int],
    n_frames: int = 60,
    fps: float = 30.0,
    speed_ms: float = 5.0,
) -> Dict[int, List[float]]:
    """
    Build synthetic state_vectors dict mimicking KinematicEstimator output.

    Returns:
        {track_id: [x, y, vel_x, vel_y, accel_x, accel_y]}
    """
    result = {}
    for i, tid in enumerate(track_ids):
        # Place vehicles along the X axis, moving right at constant speed.
        x = float(i * 10.0 + n_frames * speed_ms / fps)
        y = float(i * 3.0)
        result[tid] = [x, y, speed_ms, 0.0, 0.0, 0.0]
    return result


def make_trajectory_df(
    track_id: int = 1,
    n_frames: int = 90,
    fps: float = 30.0,
    speed_ms: float = 5.0,
    accel_ms2: float = 0.0,
) -> pd.DataFrame:
    """
    Build a pandas DataFrame that matches the schema returned by
    DuckDBClient.get_trajectory_window().

    Args:
        track_id:   Vehicle ID.
        n_frames:   Number of synthetic frames.
        fps:        Frames per second (determines dt).
        speed_ms:   Constant initial speed in X direction (m/s).
        accel_ms2:  Constant signed acceleration applied in X direction (m/s²).
    """
    dt = 1.0 / fps
    timestamps, pos_x, pos_y = [], [], []
    vel_x, vel_y, accel_x, accel_y = [], [], [], []

    vx = speed_ms
    x = 0.0

    for f in range(n_frames):
        t = f * dt
        timestamps.append(t)
        pos_x.append(x)
        pos_y.append(0.0)
        vel_x.append(vx)
        vel_y.append(0.0)
        accel_x.append(accel_ms2)
        accel_y.append(0.0)

        # Simple Euler integration for next frame
        vx = vx + accel_ms2 * dt
        x = x + vx * dt

    return pd.DataFrame({
        "timestamp": timestamps,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "vel_x": vel_x,
        "vel_y": vel_y,
        "accel_x": accel_x,
        "accel_y": accel_y,
    })


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_dir(tmp_path: Path) -> Path:
    """Temporary directory for database files — cleaned up after each test."""
    return tmp_path


@pytest.fixture
def straight_trajectory() -> pd.DataFrame:
    """90-frame synthetic trajectory at 5 m/s constant speed."""
    return make_trajectory_df(track_id=1, n_frames=90, speed_ms=5.0)


@pytest.fixture
def speeding_trajectory() -> pd.DataFrame:
    """90-frame trajectory at 20 m/s — well above the 13.89 m/s threshold."""
    return make_trajectory_df(track_id=2, n_frames=90, speed_ms=20.0)


@pytest.fixture
def hard_braking_trajectory() -> pd.DataFrame:
    """60-frame trajectory starting at 12 m/s with -5 m/s² deceleration."""
    return make_trajectory_df(
        track_id=3, n_frames=60, speed_ms=12.0, accel_ms2=-5.0
    )


@pytest.fixture
def aggressive_accel_trajectory() -> pd.DataFrame:
    """60-frame trajectory starting at 0 m/s with +4 m/s² acceleration."""
    return make_trajectory_df(
        track_id=4, n_frames=60, speed_ms=0.0, accel_ms2=4.0
    )


@pytest.fixture
def multi_vehicle_state_vectors() -> Dict[int, List[float]]:
    """Synthetic state_vectors for 3 vehicles at moderate speed."""
    return make_state_vectors(track_ids=[1, 2, 3], speed_ms=5.0)


@pytest.fixture
def close_proximity_state_vectors() -> Dict[int, List[float]]:
    """
    Two vehicles that are 1.5 m apart and closing on each other
    (triggers PROXIMITY_WARNING in AlertEngine).
    """
    return {
        1: [0.0, 0.0,  3.0, 0.0, 0.0, 0.0],   # moving right
        2: [1.5, 0.0, -3.0, 0.0, 0.0, 0.0],   # moving left → closing
    }


@pytest.fixture
def speeding_state_vectors() -> Dict[int, List[float]]:
    """Single vehicle at 20 m/s — above SPEEDING_THRESHOLD_MS = 13.89."""
    return {10: [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]}
