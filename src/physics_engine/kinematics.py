"""
Kinematic State Vector Extractor.

This module processes raw, real-world Bird's-Eye View coordinates over time
to estimate smooth velocity and acceleration vectors. It relies on classical
signal processing (Savitzky-Golay filtering) to eliminate bounding-box jitter
without requiring neural networks.

Gap handling
------------
Trackers regularly lose a detection for a small number of frames due to
occlusion or detector misses.  Without special handling, even a 1-frame gap
would wipe the trajectory history and force a 15-frame Savitzky-Golay warm-up
on every re-association.  This module solves that with two mechanisms:

  1. **Grace period** – a track's history is kept alive for up to
     ``MAX_MISSED_FRAMES`` consecutive absent frames before being purged.

  2. **Linear interpolation** – when a track reappears after N ≤
     ``MAX_MISSED_FRAMES`` absent frames, N synthetic positions are inserted
     between the last known position and the new one.  This keeps the
     Savitzky-Golay window temporally continuous.

Cold-start transparency
-----------------------
Until a track accumulates ``window_length`` samples it cannot use the full
Savitzky-Golay estimator.  The ``warm_tracks`` attribute exposes the set of
track IDs that have reliable estimates so that callers (e.g. the VLM prompt
builder) can label initialising tracks differently instead of reporting a
misleading "speed = 0 m/s".
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.signal import savgol_filter

# Number of consecutive frames a track may be absent before its trajectory
# history is discarded.  5 frames ≈ 167 ms at 30 fps — covers typical
# detector misses and short occlusions while ByteTrack still reassigns the
# same track ID.
MAX_MISSED_FRAMES: int = 5


class KinematicEstimator:
    """Estimates position, velocity, and acceleration for each tracked vehicle."""

    def __init__(
        self,
        fps: float = 30.0,
        window_length: int = 15,
        polyorder: int = 3,
    ) -> None:
        """
        Initialises the kinematic estimator.

        Args:
            fps: Frame rate of the video (used to compute Δt).
            window_length: Rolling window size for the Savitzky-Golay filter.
                Must be odd.  Larger values smooth more but add lag.
            polyorder: Polynomial order for the SG filter.  Must be less than
                ``window_length``.
        """
        self.fps = fps
        self.dt = 1.0 / fps

        # Ensure window_length is odd as required by scipy
        self.window_length = (
            window_length if window_length % 2 != 0 else window_length + 1
        )
        self.polyorder = polyorder

        # Per-track position history.  Size-limited to window_length so old
        # positions are automatically evicted.
        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.window_length)
        )

        # Consecutive missed-frame counter per track.  Tracks are kept alive
        # in self.trajectories until this exceeds MAX_MISSED_FRAMES.
        self._missed: Dict[int, int] = {}

        # Track IDs whose Savitzky-Golay window is full and whose velocity /
        # acceleration estimates are therefore reliable.  Callers (e.g. the
        # VLM prompt builder) use this to mark initialising tracks.
        self.warm_tracks: Set[int] = set()

    def update(
        self, current_coords: Dict[int, Tuple[float, float]]
    ) -> Dict[int, List[float]]:
        """
        Updates kinematic state for the current frame.

        Args:
            current_coords: Mapping of ``{track_id: (real_x, real_y)}`` from
                ``homography.py``.

        Returns:
            Mapping of ``{track_id: [x, y, v_x, v_y, a_x, a_y]}`` for all
            tracks visible in the current frame.
        """
        active_ids = set(current_coords.keys())

        # --- Step 1: increment missed counter for absent tracks ---
        for tid in list(self._missed.keys()):
            if tid not in active_ids:
                self._missed[tid] += 1

        # --- Step 2: process active tracks ---
        for tid, (curr_x, curr_y) in current_coords.items():
            gap = self._missed.get(tid, 0)

            if gap > 0 and len(self.trajectories[tid]) > 0:
                # Track reappeared after a short absence.  Linearly interpolate
                # the missing positions so the SG window sees a continuous
                # signal instead of a sudden position jump.
                last_x, last_y = self.trajectories[tid][-1]
                for i in range(1, gap + 1):
                    alpha = i / (gap + 1)
                    self.trajectories[tid].append((
                        last_x + alpha * (curr_x - last_x),
                        last_y + alpha * (curr_y - last_y),
                    ))

            # Register new tracks or reset the missed counter.
            self._missed[tid] = 0

        # --- Step 3: purge tracks absent too long ---
        stale_ids = [
            tid for tid, count in self._missed.items()
            if count > MAX_MISSED_FRAMES
        ]
        for tid in stale_ids:
            del self._missed[tid]
            self.trajectories.pop(tid, None)
            self.warm_tracks.discard(tid)

        # --- Step 4: compute state vectors for active tracks ---
        state_vectors: Dict[int, List[float]] = {}

        for track_id, (x, y) in current_coords.items():
            self.trajectories[track_id].append((x, y))
            history = self.trajectories[track_id]

            if len(history) < self.window_length:
                # Cold start: not enough frames for the SG filter yet.
                # Use simple finite differences for velocity; leave
                # acceleration as zero.  The caller should check
                # ``warm_tracks`` and treat these values as provisional.
                v_x, v_y, a_x, a_y = 0.0, 0.0, 0.0, 0.0
                if len(history) > 1:
                    prev_x, prev_y = history[-2]
                    v_x = (x - prev_x) / self.dt
                    v_y = (y - prev_y) / self.dt
                state_vectors[track_id] = [x, y, v_x, v_y, a_x, a_y]
                continue

            # Full Savitzky-Golay path — estimates are reliable.
            self.warm_tracks.add(track_id)
            pts = np.array(history)
            xs, ys = pts[:, 0], pts[:, 1]

            v_xs = savgol_filter(
                xs, self.window_length, self.polyorder, deriv=1, delta=self.dt
            )
            v_ys = savgol_filter(
                ys, self.window_length, self.polyorder, deriv=1, delta=self.dt
            )
            a_xs = savgol_filter(
                xs, self.window_length, self.polyorder, deriv=2, delta=self.dt
            )
            a_ys = savgol_filter(
                ys, self.window_length, self.polyorder, deriv=2, delta=self.dt
            )

            state_vectors[track_id] = [
                x, y,
                float(v_xs[-1]), float(v_ys[-1]),
                float(a_xs[-1]), float(a_ys[-1]),
            ]

        return state_vectors
