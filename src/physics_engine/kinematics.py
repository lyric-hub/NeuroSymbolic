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
     ``max_missed_frames`` consecutive absent frames before being purged.
     The default (30) is intentionally aligned with ByteTrack's
     ``track_buffer`` so the estimator never discards history for a track
     that ByteTrack can still re-associate to the same track_id.

  2. **Linear interpolation + SG backfill** – when a track reappears after
     N ≤ ``max_missed_frames`` absent frames, N synthetic positions are
     inserted between the last known position and the new one.  For warm
     tracks the Savitzky-Golay filter is then run over the full window
     (which now includes the synthetic positions), and the estimated
     velocity / acceleration at each synthetic sample is extracted from the
     SG output array.  For cold tracks simple finite differences are used.

     Critically, the synthetic rows are *returned to the caller* as a list
     of dicts with their own timestamps and frame IDs so the DuckDB layer
     can persist them (flagged ``interpolated=True``).  This eliminates
     the temporal gaps that would otherwise appear in SQL queries.

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

# Backward-compatible constant — retained so existing imports/tests don't break.
# New code should use KinematicEstimator(max_missed_frames=N) instead.
MAX_MISSED_FRAMES: int = 5


class KinematicEstimator:
    """Estimates position, velocity, and acceleration for each tracked vehicle."""

    def __init__(
        self,
        fps: float = 30.0,
        window_length: int = 15,
        polyorder: int = 3,
        max_missed_frames: int = 30,
    ) -> None:
        """
        Initialises the kinematic estimator.

        Args:
            fps:               Frame rate of the video (used to compute Δt).
            window_length:     Rolling window size for the Savitzky-Golay filter.
                               Must be odd.  Larger values smooth more but add lag.
            polyorder:         Polynomial order for the SG filter.  Must be less
                               than ``window_length``.
            max_missed_frames: Number of consecutive frames a track may be absent
                               before its trajectory history is purged.
                               **Must equal ByteTrack's track_buffer** so the
                               estimator never discards history for a track that
                               ByteTrack can still re-associate to the same ID.
                               Default 30 matches ByteTrack's default track_buffer.
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_missed_frames = max_missed_frames

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
        # in self.trajectories until this exceeds max_missed_frames.
        self._missed: Dict[int, int] = {}

        # Track IDs whose Savitzky-Golay window is full and whose velocity /
        # acceleration estimates are therefore reliable.  Callers (e.g. the
        # VLM prompt builder) use this to mark initialising tracks.
        self.warm_tracks: Set[int] = set()

        # Last-seen timestamp and frame_id per track — needed to compute the
        # correct timestamp / frame_id for each synthetic interpolated row.
        self._last_timestamp: Dict[int, float] = {}
        self._last_frame_id: Dict[int, int] = {}

    def update(
        self,
        current_coords: Dict[int, Tuple[float, float]],
        timestamp: float,
        frame_id: int,
    ) -> Tuple[Dict[int, List[float]], List[dict]]:
        """
        Updates kinematic state for the current frame.

        Args:
            current_coords: Mapping of ``{track_id: (real_x, real_y)}`` from
                ``homography.py``.
            timestamp:      Current video timestamp in seconds.
            frame_id:       Current video frame index.

        Returns:
            A 2-tuple:
              state_vectors   — ``{track_id: [x, y, v_x, v_y, a_x, a_y]}`` for
                                all tracks visible in the current frame.
              synthetic_rows  — list of dicts for gap-filled frames that were
                                not detected but have been interpolated.  Each
                                dict has keys: timestamp, frame_id, track_id,
                                pos_x, pos_y, vel_x, vel_y, accel_x, accel_y.
                                Write these to DuckDB with ``interpolated=True``.
        """
        active_ids = set(current_coords.keys())
        synthetic_rows: List[dict] = []

        # ── Step 1: increment missed counter for absent tracks ────────────
        for tid in list(self._missed.keys()):
            if tid not in active_ids:
                self._missed[tid] += 1

        # ── Step 2: process active tracks ────────────────────────────────
        for tid, (curr_x, curr_y) in current_coords.items():
            gap = self._missed.get(tid, 0)

            if gap > 0 and len(self.trajectories[tid]) > 0:
                # Track reappeared after a short absence.  Linearly interpolate
                # the missing positions so the SG window sees a continuous
                # signal instead of a sudden position jump.
                last_x, last_y = self.trajectories[tid][-1]
                for i in range(gap):
                    alpha = (i + 1) / (gap + 1)
                    self.trajectories[tid].append((
                        last_x + alpha * (curr_x - last_x),
                        last_y + alpha * (curr_y - last_y),
                    ))
                # Synthetic rows are built after SG (or finite-diff for cold
                # tracks) later in this iteration so derivatives are accurate.

            # Register new tracks or reset the missed counter.
            self._missed[tid] = 0

        # ── Step 3: purge tracks absent too long ──────────────────────────
        stale_ids = [
            tid for tid, count in self._missed.items()
            if count > self.max_missed_frames
        ]
        for tid in stale_ids:
            del self._missed[tid]
            self.trajectories.pop(tid, None)
            self.warm_tracks.discard(tid)
            self._last_timestamp.pop(tid, None)
            self._last_frame_id.pop(tid, None)

        # ── Step 4: compute state vectors and synthetic rows ──────────────
        state_vectors: Dict[int, List[float]] = {}

        for track_id, (x, y) in current_coords.items():
            gap = self._missed.get(track_id, 0)  # already reset to 0 above
            # Retrieve gap size BEFORE it was reset: use difference in frame IDs.
            last_fid = self._last_frame_id.get(track_id)
            actual_gap = (frame_id - last_fid - 1) if last_fid is not None else 0

            self.trajectories[track_id].append((x, y))
            history = self.trajectories[track_id]

            if len(history) < self.window_length:
                # ── Cold start: SG not yet usable ─────────────────────────
                v_x, v_y, a_x, a_y = 0.0, 0.0, 0.0, 0.0
                if len(history) > 1:
                    prev_x, prev_y = history[-2]
                    v_x = (x - prev_x) / self.dt
                    v_y = (y - prev_y) / self.dt
                state_vectors[track_id] = [x, y, v_x, v_y, a_x, a_y]

                # Synthetic rows for cold tracks: finite-difference velocities
                if actual_gap > 0 and last_fid is not None:
                    last_ts = self._last_timestamp[track_id]
                    last_x_real, last_y_real = (
                        history[-actual_gap - 2]
                        if len(history) > actual_gap + 1
                        else (x, y)
                    )
                    for k in range(actual_gap):
                        alpha_k   = (k + 1) / (actual_gap + 1)
                        alpha_k1  = (k + 2) / (actual_gap + 1)
                        syn_x = last_x_real + alpha_k  * (x - last_x_real)
                        nxt_x = last_x_real + alpha_k1 * (x - last_x_real)
                        syn_y = last_y_real + alpha_k  * (y - last_y_real)
                        nxt_y = last_y_real + alpha_k1 * (y - last_y_real)
                        syn_vx = (nxt_x - syn_x) / self.dt
                        syn_vy = (nxt_y - syn_y) / self.dt
                        frames_ahead = k + 1
                        synthetic_rows.append({
                            "timestamp": last_ts + frames_ahead * self.dt,
                            "frame_id":  last_fid + frames_ahead,
                            "track_id":  track_id,
                            "pos_x":     syn_x,
                            "pos_y":     syn_y,
                            "vel_x":     syn_vx,
                            "vel_y":     syn_vy,
                            "accel_x":   0.0,
                            "accel_y":   0.0,
                        })
            else:
                # ── Warm path: full Savitzky-Golay ────────────────────────
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

                # Synthetic rows for warm tracks: extract SG derivatives at
                # the interpolated positions inside the window.
                # After backfill + append-current, the pts array layout is:
                #   pts[-actual_gap-1] = last real position
                #   pts[-actual_gap:-1] = synthetic positions (gap frames)
                #   pts[-1]            = current real position
                if actual_gap > 0 and last_fid is not None:
                    last_ts = self._last_timestamp[track_id]
                    for k in range(actual_gap):
                        # Index into pts: synthetic frames are at
                        # [-(actual_gap - k)] counting from the end
                        # (not including the last current-frame entry).
                        idx = -(actual_gap - k)
                        frames_ahead = k + 1
                        synthetic_rows.append({
                            "timestamp": last_ts + frames_ahead * self.dt,
                            "frame_id":  last_fid + frames_ahead,
                            "track_id":  track_id,
                            "pos_x":     float(pts[idx, 0]),
                            "pos_y":     float(pts[idx, 1]),
                            "vel_x":     float(v_xs[idx]),
                            "vel_y":     float(v_ys[idx]),
                            "accel_x":   float(a_xs[idx]),
                            "accel_y":   float(a_ys[idx]),
                        })

            # Update last-seen records for this track.
            self._last_timestamp[track_id] = timestamp
            self._last_frame_id[track_id] = frame_id

        return state_vectors, synthetic_rows
