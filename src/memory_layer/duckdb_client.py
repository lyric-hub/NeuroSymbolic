import json as _json
from pathlib import Path
from typing import Dict, List
import duckdb
import pandas as pd

FLUSH_EVERY_N_FRAMES = 100


def _to_dicts(cursor) -> list:
    """
    Convert a DuckDB cursor result to a list of dicts without the Arrow bridge.

    DuckDB's .df() method routes through an Arrow C++ bridge that conflicts with
    the Kuzu graph-DB C++ allocator in the same process (SIGSEGV).
    This helper uses plain fetchall() + description to build dicts in Python,
    completely sidestepping that bridge.
    """
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _to_df(cursor) -> pd.DataFrame:
    """
    Build a pandas DataFrame from a DuckDB cursor without the Arrow bridge.

    Equivalent to cursor.df() but uses fetchall() so it is safe when Kuzu
    is also imported in the same process.
    """
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=cols)

class DuckDBClient:
    """
    Manages the high-frequency time-series storage of vehicle state vectors.
    Optimized for local edge deployment on the DGX Spark.
    """
    def __init__(self, db_path: str = "data/duckdb_storage/physics_vectors.duckdb"):
        # Ensure the data directory exists based on your project structure
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path

        # Connect to the persistent database file
        self.conn = duckdb.connect(self.db_path)
        self._initialize_schema()

        # Row buffer: accumulate rows and flush in bulk every N frames.
        # DuckDB 1.x removed the Appender class; executemany is the equivalent.
        self._buffer: list = []
        self._frames_since_flush = 0

    def _initialize_schema(self) -> None:
        """Creates tables and indexes for both physics and zone-crossing data."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_trajectories (
                timestamp            DOUBLE,
                frame_id             UINTEGER,
                track_id             UINTEGER,
                pos_x                DOUBLE,
                pos_y                DOUBLE,
                vel_x                DOUBLE,
                vel_y                DOUBLE,
                accel_x              DOUBLE,
                accel_y              DOUBLE,
                class_label          VARCHAR,
                interpolated         BOOLEAN,
                detection_confidence DOUBLE,
                speed_ms             DOUBLE,
                measurement_used     BOOLEAN,
                predicted_only       BOOLEAN,
                outlier_rejected     BOOLEAN,
                track_warm           BOOLEAN,
                trusted_for_rules    BOOLEAN,
                association_iou      DOUBLE,
                inside_calibration_core BOOLEAN,
                strong_association   BOOLEAN
            )
        """)
        # Migrations: add columns to existing databases that pre-date them.
        for migration_sql in [
            "ALTER TABLE vehicle_trajectories ADD COLUMN class_label VARCHAR DEFAULT 'unknown'",
            "ALTER TABLE vehicle_trajectories ADD COLUMN interpolated BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN detection_confidence DOUBLE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN speed_ms DOUBLE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN measurement_used BOOLEAN DEFAULT TRUE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN predicted_only BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN outlier_rejected BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN track_warm BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN trusted_for_rules BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN association_iou DOUBLE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN inside_calibration_core BOOLEAN DEFAULT TRUE",
            "ALTER TABLE vehicle_trajectories ADD COLUMN strong_association BOOLEAN DEFAULT TRUE",
        ]:
            try:
                self.conn.execute(migration_sql)
            except Exception:
                pass  # column already exists

        # Agent reasoning trace — stores every tool call made during a query
        # session so the agent's investigation path is fully auditable.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                session_id     VARCHAR,
                step_number    UINTEGER,
                tool_name      VARCHAR,
                input_args     VARCHAR,
                output_excerpt VARCHAR,
                timestamp_s    DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_time_track
            ON vehicle_trajectories(track_id, timestamp)
        """)

        # Zone crossing events — written once per event (rare), not per frame,
        # so a direct INSERT is used instead of the Appender pattern.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS zone_crossings (
                timestamp   DOUBLE,
                frame_id    UINTEGER,
                track_id    UINTEGER,
                zone_id     VARCHAR,
                gate_name   VARCHAR,
                direction   VARCHAR,
                confidence  VARCHAR,
                pixel_x     DOUBLE,
                pixel_y     DOUBLE,
                real_x      DOUBLE,
                real_y      DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_zone_gate
            ON zone_crossings(zone_id, gate_name, timestamp)
        """)

        # Fix 4: persist real-time alerts so they survive beyond the in-memory
        # deque(maxlen=200) in api.py and are queryable after processing.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS traffic_alerts (
                timestamp  DOUBLE,
                frame_id   UINTEGER,
                alert_type VARCHAR,
                severity   VARCHAR,
                track_id   INTEGER,
                message    VARCHAR,
                evidence   VARCHAR
            )
        """)

        # Calibration metadata — stores all named calibration landmarks so that
        # positions can be described relative to the nearest known landmark.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_metadata (
                name      VARCHAR,
                world_x   DOUBLE,
                world_y   DOUBLE,
                is_origin BOOLEAN
            )
        """)

    def insert_state_vectors(
        self,
        timestamp: float,
        frame_id: int,
        state_vectors: Dict[int, List[float]],
        class_labels: Dict[int, str] = None,
        detection_confidences: Dict[int, float] = None,
        state_quality: Dict[int, dict] = None,
    ):
        """
        Rapidly ingests the dictionary output from kinematics.py.
        Uses a persistent DuckDB Appender that stays open across frames,
        flushing to disk every FLUSH_EVERY_N_FRAMES frames.

        Args:
            timestamp:              The current video timestamp (seconds).
            frame_id:               The current video frame number.
            state_vectors:          Format {track_id: [x, y, v_x, v_y, a_x, a_y]}
            class_labels:           Optional {track_id: class_name} from YOLO detections
                                    (e.g. "car", "motorcycle", "bus", "truck").
            detection_confidences:  Optional {track_id: confidence} from YOLO (0.0–1.0).
                                    Stored so data quality reports can report mean
                                    detection confidence per vehicle/window.
            state_quality:          Optional per-track quality flags from
                                    KinematicEstimator.update().
        """
        if not state_vectors:
            return

        _labels = class_labels or {}
        _confs  = detection_confidences or {}
        _quality = state_quality or {}
        for track_id, sv in state_vectors.items():
            q = _quality.get(track_id, {})
            self._buffer.append((
                timestamp, frame_id, track_id,
                sv[0], sv[1], sv[2], sv[3], sv[4], sv[5],
                _labels.get(track_id, "unknown"),
                bool(q.get("interpolated", False)),
                _confs.get(track_id, None),
                q.get("speed_ms"),
                bool(q.get("measurement_used", True)),
                bool(q.get("predicted_only", False)),
                bool(q.get("outlier_rejected", False)),
                bool(q.get("track_warm", False)),
                bool(q.get("trusted_for_rules", False)),
                q.get("association_iou"),
                bool(q.get("inside_calibration_core", True)),
                bool(q.get("strong_association", True)),
            ))

        self._frames_since_flush += 1
        if self._frames_since_flush >= FLUSH_EVERY_N_FRAMES:
            self._flush()

    def _flush(self) -> None:
        """Bulk-inserts buffered rows and clears the buffer."""
        if not self._buffer:
            return
        self.conn.executemany(
            "INSERT INTO vehicle_trajectories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self._buffer,
        )
        self._buffer.clear()
        self._frames_since_flush = 0

    def insert_interpolated_rows(
        self,
        rows: List[dict],
        class_labels: Dict[int, str] = None,
    ) -> None:
        """
        Persists synthetic gap-filled rows produced by KinematicEstimator.

        These rows were never actually detected — they are linearly interpolated
        positions (with SG-smoothed derivatives for warm tracks) covering the
        frames where a vehicle was absent from the detector output.

        Flagged ``interpolated=TRUE`` in the database so queries can optionally
        exclude them (e.g. ``WHERE NOT interpolated``) when only measured data
        is acceptable.  By default they are included so time-window queries have
        a continuous signal with no temporal gaps.

        Args:
            rows:         List of dicts from ``KinematicEstimator.update()``,
                          each with keys: timestamp, frame_id, track_id,
                          pos_x, pos_y, vel_x, vel_y, accel_x, accel_y.
            class_labels: Optional ``{track_id: class_name}`` mapping.
        """
        if not rows:
            return
        _labels = class_labels or {}
        for row in rows:
            self._buffer.append((
                row["timestamp"], row["frame_id"], row["track_id"],
                row["pos_x"], row["pos_y"],
                row["vel_x"], row["vel_y"],
                row["accel_x"], row["accel_y"],
                _labels.get(row["track_id"], "unknown"),
                True,   # interpolated = True
                None,   # detection_confidence = NULL (synthetic row, no YOLO detection)
                row.get("speed_ms"),
                bool(row.get("measurement_used", False)),
                bool(row.get("predicted_only", False)),
                bool(row.get("outlier_rejected", False)),
                bool(row.get("track_warm", False)),
                bool(row.get("trusted_for_rules", False)),
                row.get("association_iou"),
                bool(row.get("inside_calibration_core", False)),
                bool(row.get("strong_association", False)),
            ))
        # Flush immediately: synthetic rows should be persisted before the next
        # real frame's data so the timeline stays ordered in DuckDB.
        self._flush()

    def get_trajectory_window(
        self,
        start_time: float,
        end_time: float,
        track_id: int,
        trusted_only: bool = False,
    ):
        """
        Tool for the LangGraph Agent: Retrieves the smoothed physics data 
        for a specific vehicle during a specific semantic event window.
        
        Returns:
            A Pandas DataFrame containing the trajectory.
        """
        trust_clause = "AND trusted_for_rules" if trusted_only else ""
        query = f"""
            SELECT
                timestamp, pos_x, pos_y, vel_x, vel_y, accel_x, accel_y,
                speed_ms, class_label, interpolated, detection_confidence,
                measurement_used, predicted_only, outlier_rejected,
                track_warm, trusted_for_rules, association_iou,
                inside_calibration_core, strong_association
            FROM vehicle_trajectories
            WHERE track_id =? AND timestamp >=? AND timestamp <=? {trust_clause}
            ORDER BY timestamp ASC
        """
        return _to_df(self.conn.execute(query, (track_id, start_time, end_time)))

    def get_vehicles_by_class(self, class_label: str) -> List[dict]:
        """
        Returns all distinct track_ids whose YOLO class label matches the query.
        Uses a LIKE match so partial strings work (e.g. "motor" matches "motorcycle").

        Args:
            class_label: Vehicle type string — "motorcycle", "car", "bus", "truck",
                         "bicycle", "person", or any partial substring.

        Returns:
            List of dicts with track_id, class_label, first_seen, last_seen, frame_count.
        """
        self._flush()
        result = self.conn.execute(
            """
            SELECT
                track_id,
                class_label,
                MIN(timestamp) AS first_seen,
                MAX(timestamp) AS last_seen,
                COUNT(*)       AS frame_count
            FROM vehicle_trajectories
            WHERE LOWER(class_label) LIKE LOWER(?)
            GROUP BY track_id, class_label
            ORDER BY track_id
            """,
            (f"%{class_label}%",),
        )
        return _to_dicts(result)

    def get_behavior_summary(
        self,
        track_ids: List[int],
        current_time: float,
        window_secs: float = 5.0,
    ) -> str:
        """
        Queries the last ``window_secs`` of trajectory data for each track and
        returns a compact, change-only narrative — not raw rows.

        Algorithm:
        1. Classify each row into a behaviour state (STOPPED/BRAKING/COASTING/ACCELERATING).
        2. Run-length encode consecutive identical states (removes repetition).
        3. Build one human-readable sentence per vehicle.

        Args:
            track_ids:    Vehicle IDs visible in the current frame.
            current_time: Timestamp of the current frame (seconds).
            window_secs:  How far back to look (default 5 s).

        Returns:
            Multi-line string ready to inject into the VLM prompt.
        """
        if not track_ids:
            return ""

        # Flush the buffer so the current frame's data is visible to the query.
        # Without this, rows written since the last periodic flush are invisible.
        self._flush()

        start_time = max(0.0, current_time - window_secs)
        lines = []

        for tid in sorted(track_ids):
            df = self.get_trajectory_window(
                start_time, current_time, tid, trusted_only=True
            )
            if df.empty:
                lines.append(f"  Vehicle {tid}: no trusted history yet")
                continue

            # Compute scalar speed and signed acceleration
            df = df.copy()
            df["speed"]       = df["speed_ms"].fillna(
                (df["vel_x"]**2 + df["vel_y"]**2).pow(0.5)
            )
            df["accel_mag"]   = (df["accel_x"]**2 + df["accel_y"]**2).pow(0.5)
            dot               = df["vel_x"]*df["accel_x"] + df["vel_y"]*df["accel_y"]
            df["signed_accel"] = df["accel_mag"].where(dot >= 0, -df["accel_mag"])

            # Classify each row into a behaviour label
            def _label(row):
                if row["speed"] < 0.5:
                    return "STOPPED"
                if row["signed_accel"] < -2.0:
                    return "BRAKING"
                if row["signed_accel"] > 1.5:
                    return "ACCELERATING"
                return "MOVING"

            df["state"] = df.apply(_label, axis=1)

            # Run-length encode — only keep state transitions
            segments = []
            prev_state, seg_start = None, df["timestamp"].iloc[0]
            for _, row in df.iterrows():
                if row["state"] != prev_state:
                    if prev_state is not None:
                        segments.append((prev_state, seg_start, row["timestamp"]))
                    prev_state, seg_start = row["state"], row["timestamp"]
            segments.append((prev_state, seg_start, df["timestamp"].iloc[-1]))

            # Build narrative from segments (skip single-frame blips < 0.2s)
            parts = []
            for state, t0, t1 in segments:
                dur = t1 - t0
                if dur < 0.5 and len(segments) > 1:
                    continue
                label_map = {
                    "STOPPED":      f"stationary for {dur:.1f}s",
                    "BRAKING":      f"braking for {dur:.1f}s",
                    "ACCELERATING": f"accelerating for {dur:.1f}s",
                    "MOVING":       f"moving for {dur:.1f}s",
                }
                parts.append(label_map[state])

            # Append current speed/accel + nearest landmark at the last row
            last = df.iloc[-1]
            lm_name, lm_dist = self.nearest_landmark(float(last["pos_x"]), float(last["pos_y"]))
            current_str = (
                f"now: speed={last['speed']:.1f} m/s, "
                f"accel={last['signed_accel']:+.1f} m/s², "
                f"position={lm_dist:.1f}m from {lm_name}"
            )
            narrative = " → ".join(parts) + f" | {current_str}"
            lines.append(f"  Vehicle {tid}: {narrative}")

        return "\n".join(lines)

    def insert_crossing_event(self, event) -> None:
        """
        Persists a single gate-crossing event from ZoneManager.

        Args:
            event: A ``CrossingEvent`` dataclass instance.
        """
        self.conn.execute(
            """
            INSERT INTO zone_crossings
                (timestamp, frame_id, track_id, zone_id, gate_name, direction,
                 confidence, pixel_x, pixel_y, real_x, real_y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.timestamp, event.frame_id, event.track_id,
                event.zone_id, event.gate_name, event.direction,
                event.confidence,
                event.pixel_x, event.pixel_y, event.real_x, event.real_y,
            ),
        )

    def query_zone_flow(
        self,
        zone_id: str = "",
        gate_name: str = "",
        start_time: float = 0.0,
        end_time: float = 9_999_999.0,
    ) -> dict:
        """
        Returns entry/exit counts per gate and OD pairs for the specified zone.

        Args:
            zone_id:    Filter to a specific zone (empty = all zones).
            gate_name:  Filter to a specific gate (empty = all gates).
            start_time: Lower bound on event timestamp.
            end_time:   Upper bound on event timestamp.

        Returns:
            Dict with keys ``gate_counts`` and ``od_pairs``.
        """
        # Build optional WHERE clauses dynamically
        filters = ["timestamp >= ? AND timestamp <= ?"]
        params: list = [start_time, end_time]

        if zone_id:
            filters.append("zone_id = ?")
            params.append(zone_id)
        if gate_name:
            filters.append("gate_name = ?")
            params.append(gate_name)

        where = " AND ".join(filters)

        # Per-gate enter/exit counts
        counts_rows = _to_dicts(self.conn.execute(
            f"""
            SELECT gate_name, direction, COUNT(*) AS cnt
            FROM zone_crossings
            WHERE {where}
            GROUP BY gate_name, direction
            ORDER BY gate_name, direction
            """,
            params,
        ))

        gate_counts: dict = {}
        for row in counts_rows:
            g = row["gate_name"]
            if g not in gate_counts:
                gate_counts[g] = {"enter": 0, "exit": 0}
            gate_counts[g][row["direction"]] = int(row["cnt"])

        # OD pairs — join each vehicle's first entry with its last exit.
        # confidence = 'confirmed' only if BOTH entry and exit were confirmed.
        od_pairs = _to_dicts(self.conn.execute(
            f"""
            WITH enters AS (
                SELECT
                    track_id,
                    gate_name  AS origin_gate,
                    confidence AS entry_confidence,
                    MIN(timestamp) AS entry_time
                FROM zone_crossings
                WHERE direction = 'enter' AND {where}
                GROUP BY track_id, gate_name, confidence
            ),
            exits AS (
                SELECT
                    track_id,
                    gate_name  AS dest_gate,
                    confidence AS exit_confidence,
                    MAX(timestamp) AS exit_time
                FROM zone_crossings
                WHERE direction = 'exit' AND {where}
                GROUP BY track_id, gate_name, confidence
            )
            SELECT
                e.track_id,
                e.origin_gate,
                x.dest_gate,
                e.entry_time,
                x.exit_time,
                ROUND(x.exit_time - e.entry_time, 2) AS dwell_time_seconds,
                e.entry_confidence,
                x.exit_confidence,
                CASE
                    WHEN e.entry_confidence = 'confirmed'
                     AND x.exit_confidence = 'confirmed'
                    THEN 'confirmed' ELSE 'estimated'
                END AS od_confidence
            FROM enters e
            JOIN exits x ON e.track_id = x.track_id
            ORDER BY e.entry_time ASC
            """,
            params * 2,
        ))

        return {"gate_counts": gate_counts, "od_pairs": od_pairs}

    def insert_alert(self, alert) -> None:
        """
        Persists a single TrafficAlert to the traffic_alerts table immediately
        (no buffering — alerts are rare and must survive a crash).

        Args:
            alert: A TrafficAlert dataclass instance from AlertEngine.
        """
        self.conn.execute(
            "INSERT INTO traffic_alerts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                alert.timestamp,
                alert.frame_id,
                alert.alert_type,
                alert.severity,
                alert.track_id,
                alert.message,
                _json.dumps(alert.evidence),
            ),
        )

    def set_named_points(
        self,
        named_points: list,
        reference_name: str,
    ) -> None:
        """
        Persists all named calibration landmarks from the CoordinateTransformer.

        Called once at pipeline startup. Replaces any existing entries.

        Args:
            named_points:   List of dicts with keys: name, world_x, world_y.
            reference_name: Name of the (0, 0) origin point.
        """
        self.conn.execute("DELETE FROM calibration_metadata")
        for pt in named_points:
            self.conn.execute(
                "INSERT INTO calibration_metadata VALUES (?, ?, ?, ?)",
                (
                    pt["name"],
                    pt["world_x"],
                    pt["world_y"],
                    pt["name"] == reference_name,
                ),
            )

    def get_reference_name(self) -> str:
        """
        Returns the name of the origin (is_origin=true) landmark.
        Falls back to 'Origin' if not yet set.
        """
        row = self.conn.execute(
            "SELECT name FROM calibration_metadata WHERE is_origin = true LIMIT 1"
        ).fetchone()
        return row[0] if row else "Origin"

    def get_named_points(self) -> List[dict]:
        """Returns all named calibration landmarks from calibration_metadata."""
        rows = self.conn.execute(
            "SELECT name, world_x, world_y FROM calibration_metadata ORDER BY is_origin DESC"
        ).fetchall()
        return [{"name": r[0], "world_x": r[1], "world_y": r[2]} for r in rows]

    def nearest_landmark(self, x: float, y: float) -> tuple:
        """
        Returns (landmark_name, distance_metres) of the closest named calibration point.
        Falls back to ('Origin', distance) when no landmarks are stored.
        """
        points = self.get_named_points()
        if not points:
            return "Origin", round((x ** 2 + y ** 2) ** 0.5, 2)
        best_name, best_dist = "Origin", float("inf")
        for pt in points:
            dx = x - pt["world_x"]
            dy = y - pt["world_y"]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_name = pt["name"]
        return best_name, round(best_dist, 2)

    # ------------------------------------------------------------------
    # Explainability: data quality + reasoning trace
    # ------------------------------------------------------------------

    def get_data_quality_report(
        self,
        track_id: int = -1,
        start_time: float = 0.0,
        end_time: float = 9_999_999.0,
    ) -> dict:
        """
        Returns a data quality summary for a vehicle (or all vehicles) in a
        time window.

        Covers three quality dimensions:
          1. Frame coverage — how many rows are real vs predicted vs interpolated.
          2. Detection confidence — mean YOLO confidence for real frames.
          3. Gate crossing confidence — confirmed vs estimated crossings.

        Use this BEFORE citing evaluate_traffic_rules findings — a violation
        based on 95% real frames is much stronger evidence than one based on
        30% real frames with the rest synthesised by the SG interpolator.

        Args:
            track_id:   Vehicle ID, or -1 for all vehicles.
            start_time: Window start in seconds.
            end_time:   Window end in seconds.

        Returns:
            Dict with: track_filter, time_window, total_frames, real_frames,
            interpolated_frames, coverage_pct, mean_detection_confidence,
            confirmed_gate_crossings, estimated_gate_crossings,
            data_quality_rating (HIGH / MEDIUM / LOW).
        """
        self._flush()
        track_filter = "" if track_id < 0 else "AND track_id = ?"
        params_traj = [start_time, end_time] + ([track_id] if track_id >= 0 else [])

        row = self.conn.execute(
            f"""
            SELECT
                COUNT(*)                                          AS total_frames,
                SUM(CASE WHEN measurement_used AND NOT predicted_only AND NOT interpolated
                         THEN 1 ELSE 0 END) AS real_frames,
                SUM(CASE WHEN predicted_only THEN 1 ELSE 0 END) AS predicted_frames,
                SUM(CASE WHEN interpolated   THEN 1 ELSE 0 END) AS interp_frames,
                SUM(CASE WHEN trusted_for_rules THEN 1 ELSE 0 END) AS trusted_frames,
                AVG(CASE WHEN measurement_used AND NOT predicted_only AND NOT interpolated
                         THEN detection_confidence END)           AS mean_conf
            FROM vehicle_trajectories
            WHERE timestamp >= ? AND timestamp <= ? {track_filter}
            """,
            params_traj,
        ).fetchone()

        total     = int(row[0] or 0)
        real      = int(row[1] or 0)
        predicted = int(row[2] or 0)
        interp    = int(row[3] or 0)
        trusted   = int(row[4] or 0)
        mean_conf = round(float(row[5]), 3) if row[5] is not None else None
        coverage  = round(100.0 * real / total, 1) if total > 0 else 0.0

        # Gate crossing confidence breakdown
        params_zc = [start_time, end_time] + ([track_id] if track_id >= 0 else [])
        zc_filter = "" if track_id < 0 else "AND track_id = ?"
        zc_row = self.conn.execute(
            f"""
            SELECT
                SUM(CASE WHEN confidence = 'confirmed'  THEN 1 ELSE 0 END),
                SUM(CASE WHEN confidence = 'estimated'  THEN 1 ELSE 0 END)
            FROM zone_crossings
            WHERE timestamp >= ? AND timestamp <= ? {zc_filter}
            """,
            params_zc,
        ).fetchone()
        confirmed_crossings = int(zc_row[0] or 0)
        estimated_crossings = int(zc_row[1] or 0)

        # Rating heuristic: HIGH ≥ 85% real + conf ≥ 0.6; LOW < 60% real or conf < 0.4
        if coverage >= 85.0 and (mean_conf is None or mean_conf >= 0.6):
            rating = "HIGH"
        elif coverage >= 60.0 and (mean_conf is None or mean_conf >= 0.4):
            rating = "MEDIUM"
        else:
            rating = "LOW"

        return {
            "track_filter": track_id if track_id >= 0 else "all",
            "time_window": f"{start_time}-{end_time}",
            "total_frames": total,
            "real_frames": real,
            "predicted_only_frames": predicted,
            "interpolated_frames": interp,
            "trusted_frames": trusted,
            "coverage_pct": coverage,
            "mean_detection_confidence": mean_conf,
            "confirmed_gate_crossings": confirmed_crossings,
            "estimated_gate_crossings": estimated_crossings,
            "data_quality_rating": rating,
            "_interpretation": (
                f"{coverage:.1f}% of frames are real YOLO detections "
                f"({'confidence ' + str(mean_conf) if mean_conf else 'confidence not recorded'}). "
                f"Rating: {rating}. "
                + ("Findings from this window are RELIABLE. " if rating == "HIGH"
                   else "Treat findings as INDICATIVE — review raw frames. " if rating == "MEDIUM"
                   else "LOW data quality — DO NOT cite rule violations as confirmed facts. ")
            ),
        }

    def insert_reasoning_trace(self, session_id: str, steps: list) -> None:
        """
        Persists the agent's investigation steps for a query session to DuckDB.

        Each step records which tool was called, with what arguments, and what
        the tool returned (truncated to 300 chars). This makes the agent's
        full reasoning path auditable after the fact — queryable via the
        get_reasoning_trace tool.

        Args:
            session_id: UUID string identifying the query session.
            steps:      List of dicts with keys: step, tool, args, output_excerpt.
        """
        import time as _time
        ts = _time.time()
        for step in steps:
            self.conn.execute(
                "INSERT INTO analysis_sessions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    int(step.get("step", 0)),
                    str(step.get("tool", "")),
                    str(step.get("args", {}))[:500],
                    str(step.get("output_excerpt", ""))[:300],
                    ts,
                ),
            )

    def get_reasoning_trace(self, session_id: str) -> list:
        """
        Retrieves the full investigation trace for a prior query session.

        Args:
            session_id: UUID returned in the original query response.

        Returns:
            Ordered list of tool call steps with args and output excerpts.
        """
        rows = self.conn.execute(
            """
            SELECT step_number, tool_name, input_args, output_excerpt, timestamp_s
            FROM analysis_sessions
            WHERE session_id = ?
            ORDER BY step_number ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            {
                "step": r[0],
                "tool": r[1],
                "args": r[2],
                "output_excerpt": r[3],
                "recorded_at": r[4],
            }
            for r in rows
        ]

    def get_trajectory_path(
        self,
        track_id: int,
        start_time: float,
        end_time: float,
        max_rows: int = 50,
    ) -> List[dict]:
        """
        Returns a sampled sequence of (timestamp, pos_x, pos_y, speed, nearest_landmark)
        for a vehicle over a time window.

        Used by get_vehicle_trajectory tool for spatial reconstruction.

        Args:
            track_id:   Vehicle track ID.
            start_time: Window start in seconds.
            end_time:   Window end in seconds.
            max_rows:   Maximum number of rows to return (sampled evenly).

        Returns:
            List of dicts: timestamp, pos_x, pos_y, speed_ms,
                           nearest_landmark, distance_to_landmark_m.
        """
        self._flush()
        df = self.get_trajectory_window(
            start_time, end_time, track_id, trusted_only=True
        )
        if df.empty:
            return []
        step = max(1, len(df) // max_rows)
        sampled = df.iloc[::step].copy()
        sampled["_speed"] = sampled["speed_ms"].fillna(
            (sampled["vel_x"] ** 2 + sampled["vel_y"] ** 2) ** 0.5
        )
        result = []
        for _, row in sampled.iterrows():
            lm, dist = self.nearest_landmark(float(row["pos_x"]), float(row["pos_y"]))
            result.append({
                "timestamp": round(float(row["timestamp"]), 2),
                "pos_x": round(float(row["pos_x"]), 2),
                "pos_y": round(float(row["pos_y"]), 2),
                "speed_ms": round(float(row["_speed"]), 2),
                "nearest_landmark": lm,
                "distance_to_landmark_m": dist,
            })
        return result

    # Standard half-lengths (metres) per YOLO class — used to convert
    # centre-to-centre distance into the physically meaningful tail-to-head gap.
    _HALF_LENGTH: Dict[str, float] = {
        "car":        2.25,
        "motorcycle": 1.00,
        "bicycle":    0.90,
        "bus":        6.00,
        "truck":      4.00,
        "person":     0.45,
    }
    _DEFAULT_HALF_LENGTH = 2.25  # fallback if class unknown

    def _half_length(self, track_id: int, start_time: float, end_time: float) -> float:
        """Returns the half-length (m) for a vehicle based on its YOLO class label."""
        row = self.conn.execute(
            """
            SELECT class_label FROM vehicle_trajectories
            WHERE track_id = ? AND timestamp >= ? AND timestamp <= ?
            LIMIT 1
            """,
            (track_id, start_time, end_time),
        ).fetchone()
        label = (row[0] or "").lower() if row else ""
        for key, half in self._HALF_LENGTH.items():
            if key in label:
                return half
        return self._DEFAULT_HALF_LENGTH

    def get_vehicle_proximity(
        self,
        track_id_a: int,
        track_id_b: int,
        start_time: float,
        end_time: float,
    ) -> dict:
        """
        Finds the minimum tail-to-head gap between two vehicles within a time window.

        Uses centre-to-centre distance minus each vehicle's half-length so the
        result represents the actual clearance between vehicle bodies — not the
        distance between their geometric centres.

        gap = centre_distance - half_length_A - half_length_B
        gap ≤ 0  → collision (bodies overlapping)
        gap < 1m → near-miss

        Uses pandas merge_asof to align trajectories on nearest timestamp.

        Returns:
            Dict with min_gap_m, centre_distance_m, timestamp_s,
            vehicle_a_pos, vehicle_b_pos, collision_confirmed.
            On missing data: {"error": "..."}.
        """
        import pandas as pd
        self._flush()
        df_a = self.get_trajectory_window(
            start_time, end_time, track_id_a, trusted_only=True
        )
        df_b = self.get_trajectory_window(
            start_time, end_time, track_id_b, trusted_only=True
        )
        if df_a.empty or df_b.empty:
            return {"error": f"No data for one or both vehicles (ids={track_id_a},{track_id_b}) in t={start_time}-{end_time}s."}

        half_a = self._half_length(track_id_a, start_time, end_time)
        half_b = self._half_length(track_id_b, start_time, end_time)

        df_a = df_a.sort_values("timestamp").reset_index(drop=True)
        df_b = df_b.sort_values("timestamp").reset_index(drop=True)
        merged = pd.merge_asof(df_a, df_b, on="timestamp", suffixes=("_a", "_b"), direction="nearest")
        merged["_centre_dist"] = ((merged["pos_x_a"] - merged["pos_x_b"]) ** 2 + (merged["pos_y_a"] - merged["pos_y_b"]) ** 2) ** 0.5
        merged["_gap"] = merged["_centre_dist"] - half_a - half_b

        min_idx = merged["_gap"].idxmin()
        min_row = merged.loc[min_idx]
        min_gap = float(min_row["_gap"])

        return {
            "min_gap_m": round(min_gap, 2),
            "centre_distance_m": round(float(min_row["_centre_dist"]), 2),
            "collision_confirmed": bool(min_gap <= 0.0),
            "timestamp_s": round(float(min_row["timestamp"]), 2),
            "vehicle_a_half_length_m": half_a,
            "vehicle_b_half_length_m": half_b,
            "vehicle_a_pos": {"x": round(float(min_row["pos_x_a"]), 2), "y": round(float(min_row["pos_y_a"]), 2)},
            "vehicle_b_pos": {"x": round(float(min_row["pos_x_b"]), 2), "y": round(float(min_row["pos_y_b"]), 2)},
        }

    # ------------------------------------------------------------------
    # Interval-based trajectory sampling
    # ------------------------------------------------------------------

    def get_sampled_trajectory(
        self,
        track_id: int,
        interval_secs: float = 0.5,
        start_time: float = 0.0,
        end_time: float = 9_999_999.0,
    ) -> List[dict]:
        """
        Returns one data snapshot per time interval for a single vehicle.

        Instead of returning every frame (~30 rows/sec), this picks the
        first row that falls inside each fixed-width time bucket:

            bucket_start = FLOOR(timestamp / interval) * interval

        So for interval=0.5 you get rows at t≈0.0, 0.5, 1.0, 1.5 …
        exactly one row per bucket, regardless of source frame rate.

        Each row is enriched with derived fields (speed, signed acceleration,
        nearest landmark) so the output is ready for export or display without
        further processing.

        Args:
            track_id:      Vehicle track ID.
            interval_secs: Bucket width in seconds (default 0.5 s).
                           Valid range: 0.05 s – 3600 s.
            start_time:    Start of the query window (seconds).
            end_time:      End of the query window (seconds).

        Returns:
            List of dicts, one per interval bucket, keys:
                time_bucket, timestamp, track_id, class_label,
                pos_x, pos_y, vel_x, vel_y, accel_x, accel_y,
                speed_ms, signed_accel_ms2,
                nearest_landmark, distance_to_landmark_m.
        """
        interval_secs = float(interval_secs)
        if not (0.05 <= interval_secs <= 3600.0):
            raise ValueError(f"interval_secs must be between 0.05 and 3600, got {interval_secs}")

        self._flush()

        # FLOOR bucketing: pick the earliest sample in each time bucket.
        # arg_min(col, timestamp) returns the value of col at the row
        # where timestamp is minimum — deterministic, no random tie-breaking.
        iv = interval_secs  # float literal — safe for numeric f-string injection
        query = f"""
            SELECT
                FLOOR(timestamp / {iv}) * {iv}  AS time_bucket,
                arg_min(timestamp,  timestamp)   AS timestamp,
                arg_min(pos_x,      timestamp)   AS pos_x,
                arg_min(pos_y,      timestamp)   AS pos_y,
                arg_min(vel_x,      timestamp)   AS vel_x,
                arg_min(vel_y,      timestamp)   AS vel_y,
                arg_min(accel_x,    timestamp)   AS accel_x,
                arg_min(accel_y,    timestamp)   AS accel_y,
                arg_min(class_label,timestamp)   AS class_label
            FROM vehicle_trajectories
            WHERE track_id = ?
              AND timestamp >= ?
              AND timestamp <= ?
              AND trusted_for_rules
            GROUP BY time_bucket
            ORDER BY time_bucket
        """
        df = _to_df(self.conn.execute(query, (track_id, start_time, end_time)))
        if df.empty:
            return []

        df["speed_ms"] = (df["vel_x"] ** 2 + df["vel_y"] ** 2) ** 0.5
        accel_mag = (df["accel_x"] ** 2 + df["accel_y"] ** 2) ** 0.5
        dot = df["vel_x"] * df["accel_x"] + df["vel_y"] * df["accel_y"]
        df["signed_accel_ms2"] = accel_mag.where(dot >= 0, -accel_mag)

        result = []
        for _, row in df.iterrows():
            lm, dist = self.nearest_landmark(float(row["pos_x"]), float(row["pos_y"]))
            result.append({
                "time_bucket":              round(float(row["time_bucket"]), 3),
                "timestamp":                round(float(row["timestamp"]), 3),
                "track_id":                 track_id,
                "class_label":              row["class_label"],
                "pos_x":                    round(float(row["pos_x"]), 3),
                "pos_y":                    round(float(row["pos_y"]), 3),
                "vel_x":                    round(float(row["vel_x"]), 3),
                "vel_y":                    round(float(row["vel_y"]), 3),
                "accel_x":                  round(float(row["accel_x"]), 3),
                "accel_y":                  round(float(row["accel_y"]), 3),
                "speed_ms":                 round(float(row["speed_ms"]), 3),
                "signed_accel_ms2":         round(float(row["signed_accel_ms2"]), 3),
                "nearest_landmark":         lm,
                "distance_to_landmark_m":   dist,
            })
        return result

    def get_all_vehicles_sampled(
        self,
        interval_secs: float = 0.5,
        start_time: float = 0.0,
        end_time: float = 9_999_999.0,
    ) -> List[dict]:
        """
        Returns one data snapshot per (vehicle, time bucket) for all vehicles.

        Same bucketing logic as ``get_sampled_trajectory`` but covers every
        track_id present in the given time window in a single SQL query.

        Useful for:
        - Building a time-series table for a dashboa or export (CSV/Excel).
        - Comparing all vehicles' positions at the same timestamps.
        - Feeding a downstream ML pipeline that expects fixed-rate inputs.

        Args:
            interval_secs: Bucket width in seconds (default 0.5 s).
            start_time:    Start of the query window (seconds).
            end_time:      End of the query window (seconds).

        Returns:
            Flat list of dicts sorted by (time_bucket, track_id).
            Each dict has the same keys as ``get_sampled_trajectory`` rows.
        """
        interval_secs = float(interval_secs)
        if not (0.05 <= interval_secs <= 3600.0):
            raise ValueError(f"interval_secs must be between 0.05 and 3600, got {interval_secs}")

        self._flush()

        iv = interval_secs
        query = f"""
            SELECT
                track_id,
                FLOOR(timestamp / {iv}) * {iv}  AS time_bucket,
                arg_min(timestamp,  timestamp)   AS timestamp,
                arg_min(pos_x,      timestamp)   AS pos_x,
                arg_min(pos_y,      timestamp)   AS pos_y,
                arg_min(vel_x,      timestamp)   AS vel_x,
                arg_min(vel_y,      timestamp)   AS vel_y,
                arg_min(accel_x,    timestamp)   AS accel_x,
                arg_min(accel_y,    timestamp)   AS accel_y,
                arg_min(class_label,timestamp)   AS class_label
            FROM vehicle_trajectories
            WHERE timestamp >= ?
              AND timestamp <= ?
              AND trusted_for_rules
            GROUP BY track_id, time_bucket
            ORDER BY time_bucket, track_id
        """
        df = _to_df(self.conn.execute(query, (start_time, end_time)))
        if df.empty:
            return []

        df["speed_ms"] = (df["vel_x"] ** 2 + df["vel_y"] ** 2) ** 0.5
        accel_mag = (df["accel_x"] ** 2 + df["accel_y"] ** 2) ** 0.5
        dot = df["vel_x"] * df["accel_x"] + df["vel_y"] * df["accel_y"]
        df["signed_accel_ms2"] = accel_mag.where(dot >= 0, -accel_mag)

        result = []
        for _, row in df.iterrows():
            lm, dist = self.nearest_landmark(float(row["pos_x"]), float(row["pos_y"]))
            result.append({
                "time_bucket":              round(float(row["time_bucket"]), 3),
                "timestamp":                round(float(row["timestamp"]), 3),
                "track_id":                 int(row["track_id"]),
                "class_label":              row["class_label"],
                "pos_x":                    round(float(row["pos_x"]), 3),
                "pos_y":                    round(float(row["pos_y"]), 3),
                "vel_x":                    round(float(row["vel_x"]), 3),
                "vel_y":                    round(float(row["vel_y"]), 3),
                "accel_x":                  round(float(row["accel_x"]), 3),
                "accel_y":                  round(float(row["accel_y"]), 3),
                "speed_ms":                 round(float(row["speed_ms"]), 3),
                "signed_accel_ms2":         round(float(row["signed_accel_ms2"]), 3),
                "nearest_landmark":         lm,
                "distance_to_landmark_m":   dist,
            })
        return result

    # ------------------------------------------------------------------
    # Time-to-Collision (TTC) analysis
    # ------------------------------------------------------------------

    def compute_ttc(
        self,
        track_id_a: int,
        track_id_b: int,
        start_time: float,
        end_time: float,
    ) -> dict:
        """
        Computes Time-to-Collision (TTC) between two vehicles across a window.

        TTC = gap / closing_speed, where:
          gap           = centre_distance - half_length_A - half_length_B  (metres)
          closing_speed = dot(relative_velocity_A_minus_B, unit_vector_A_to_B)

        TTC is only defined when closing_speed > 0 (vehicles approaching).
        When vehicles are diverging (closing_speed ≤ 0) TTC = infinity.

        Standard conflict thresholds (SSAM / HCM):
          TTC < 1.5 s  → critical conflict
          TTC < 3.0 s  → serious conflict

        Returns:
            Dict with min_ttc_s, conflict_level, timestamp_s, gap_at_min_ttc_m,
            closing_speed_ms, vehicle_a/b_pos, and a sampled ttc_series list.
            On missing data: {"error": "..."}.
        """
        import pandas as pd
        self._flush()
        df_a = self.get_trajectory_window(
            start_time, end_time, track_id_a, trusted_only=True
        )
        df_b = self.get_trajectory_window(
            start_time, end_time, track_id_b, trusted_only=True
        )
        if df_a.empty or df_b.empty:
            return {
                "error": (
                    f"No data for one or both vehicles (ids={track_id_a},{track_id_b}) "
                    f"in t={start_time}-{end_time}s."
                )
            }

        half_a = self._half_length(track_id_a, start_time, end_time)
        half_b = self._half_length(track_id_b, start_time, end_time)

        df_a = df_a.sort_values("timestamp").reset_index(drop=True)
        df_b = df_b.sort_values("timestamp").reset_index(drop=True)
        merged = pd.merge_asof(
            df_a, df_b, on="timestamp",
            suffixes=("_a", "_b"), direction="nearest",
        )

        dx = merged["pos_x_b"] - merged["pos_x_a"]
        dy = merged["pos_y_b"] - merged["pos_y_a"]
        centre_dist = (dx ** 2 + dy ** 2) ** 0.5
        # Unit vector from A to B (avoid divide-by-zero at coincident positions)
        eps = 1e-6
        ux = dx / (centre_dist + eps)
        uy = dy / (centre_dist + eps)

        rel_vx = merged["vel_x_a"] - merged["vel_x_b"]
        rel_vy = merged["vel_y_a"] - merged["vel_y_b"]
        closing_speed = rel_vx * ux + rel_vy * uy   # positive = approaching

        gap = centre_dist - half_a - half_b
        # TTC only defined for approaching pairs with positive gap
        ttc = gap / closing_speed.clip(lower=eps)
        ttc = ttc.where((closing_speed > 0) & (gap > 0), other=float("inf"))
        merged["ttc_s"] = ttc
        merged["gap_m"] = gap
        merged["closing_speed_ms"] = closing_speed

        finite_mask = merged["ttc_s"] < 1e8
        if not finite_mask.any():
            return {
                "track_id_a": track_id_a,
                "track_id_b": track_id_b,
                "time_window": f"{start_time}-{end_time}",
                "min_ttc_s": None,
                "conflict_level": "NONE",
                "note": "Vehicles were never approaching each other in this window.",
            }

        min_idx = merged.loc[finite_mask, "ttc_s"].idxmin()
        min_row = merged.loc[min_idx]
        min_ttc = float(min_row["ttc_s"])

        if min_ttc < 1.5:
            conflict = "CRITICAL"
        elif min_ttc < 3.0:
            conflict = "SERIOUS"
        else:
            conflict = "LOW"

        # Sample up to 20 points from the TTC series for the agent
        sample_step = max(1, len(merged) // 20)
        series = []
        for _, row in merged.iloc[::sample_step].iterrows():
            series.append({
                "timestamp": round(float(row["timestamp"]), 2),
                "ttc_s": round(float(row["ttc_s"]), 2) if row["ttc_s"] < 1e8 else None,
                "gap_m": round(float(row["gap_m"]), 2),
                "closing_speed_ms": round(float(row["closing_speed_ms"]), 2),
            })

        return {
            "track_id_a": track_id_a,
            "track_id_b": track_id_b,
            "time_window": f"{start_time}-{end_time}",
            "min_ttc_s": round(min_ttc, 2),
            "conflict_level": conflict,
            "timestamp_of_min_ttc_s": round(float(min_row["timestamp"]), 2),
            "gap_at_min_ttc_m": round(float(min_row["gap_m"]), 2),
            "closing_speed_ms": round(float(min_row["closing_speed_ms"]), 2),
            "vehicle_a_pos": {
                "x": round(float(min_row["pos_x_a"]), 2),
                "y": round(float(min_row["pos_y_a"]), 2),
            },
            "vehicle_b_pos": {
                "x": round(float(min_row["pos_x_b"]), 2),
                "y": round(float(min_row["pos_y_b"]), 2),
            },
            "ttc_series": series,
        }

    # ------------------------------------------------------------------
    # Speed statistics (85th percentile, mean, max)
    # ------------------------------------------------------------------

    def get_speed_statistics(
        self,
        track_id: int = -1,
        start_time: float = 0.0,
        end_time: float = 9_999_999.0,
    ) -> List[dict]:
        """
        Computes speed statistics including the 85th percentile speed.

        The 85th percentile speed (V85) is the standard design speed metric
        used in road safety audits and speed limit policy — it represents the
        speed below which 85% of vehicles travel.

        Args:
            track_id:   -1 = all vehicles (one row per vehicle);
                        ≥ 0 = single vehicle.
            start_time: Window start (seconds).
            end_time:   Window end (seconds).

        Returns:
            List of dicts per vehicle: track_id, class_label,
            v85_speed_kmh, mean_speed_kmh, max_speed_kmh, data_points.
        """
        self._flush()
        track_filter = "" if track_id < 0 else f"AND track_id = {int(track_id)}"
        query = f"""
            WITH speeds AS (
                SELECT
                    track_id,
                    class_label,
                    speed_ms
                FROM vehicle_trajectories
                WHERE timestamp >= ? AND timestamp <= ?
                  AND trusted_for_rules
                  {track_filter}
            )
            SELECT
                track_id,
                MIN(class_label) AS class_label,
                PERCENTILE_CONT(0.85) WITHIN GROUP (ORDER BY speed_ms) AS v85_ms,
                AVG(speed_ms)  AS mean_ms,
                MAX(speed_ms)  AS max_ms,
                COUNT(*)       AS data_points
            FROM speeds
            GROUP BY track_id
            ORDER BY v85_ms DESC
        """
        rows = _to_dicts(self.conn.execute(query, (start_time, end_time)))
        return [
            {
                "track_id":       int(row["track_id"]),
                "class_label":    row["class_label"],
                "v85_speed_kmh":  round(float(row["v85_ms"]) * 3.6, 1),
                "mean_speed_kmh": round(float(row["mean_ms"]) * 3.6, 1),
                "max_speed_kmh":  round(float(row["max_ms"]) * 3.6, 1),
                "data_points":    int(row["data_points"]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Vehicle count by time period
    # ------------------------------------------------------------------

    def get_vehicle_count_by_period(
        self,
        period_secs: float = 3600.0,
        start_time: float = 0.0,
        end_time: float = 9_999_999.0,
    ) -> List[dict]:
        """
        Counts distinct vehicles observed per fixed time period.

        Used for flow rate analysis: how many vehicles pass per hour/minute.
        Also computes Peak Hour Factor (PHF) = total_count / (4 × peak_15min_count)
        when period_secs = 900 (15 minutes).

        Args:
            period_secs: Bucket width in seconds (default 3600 = 1 hour).
            start_time:  Window start (seconds).
            end_time:    Window end (seconds).

        Returns:
            List of dicts: period_start_s, period_end_s, vehicle_count,
            vehicles_per_hour (flow rate normalised to hourly).
        """
        self._flush()
        pf = float(period_secs)
        query = f"""
            SELECT
                FLOOR(timestamp / {pf}) * {pf}  AS period_start,
                COUNT(DISTINCT track_id)          AS vehicle_count
            FROM vehicle_trajectories
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY period_start
            ORDER BY period_start
        """
        rows = _to_dicts(self.conn.execute(query, (start_time, end_time)))
        hourly_factor = 3600.0 / pf
        return [
            {
                "period_start_s":    round(float(row["period_start"]), 1),
                "period_end_s":      round(float(row["period_start"]) + pf, 1),
                "vehicle_count":     int(row["vehicle_count"]),
                "vehicles_per_hour": round(int(row["vehicle_count"]) * hourly_factor, 1),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Queue / congestion detection
    # ------------------------------------------------------------------

    def detect_queues(
        self,
        start_time: float = 0.0,
        end_time: float = 9_999_999.0,
        min_vehicles: int = 3,
        max_speed_ms: float = 1.0,
        period_secs: float = 2.0,
    ) -> List[dict]:
        """
        Detects congestion / queue events — time windows where ≥ N vehicles
        are simultaneously near-stationary (speed < max_speed_ms).

        Algorithm:
          1. Bucket timestamps into period_secs intervals.
          2. For each bucket, count distinct vehicles with speed < max_speed_ms.
          3. Merge adjacent buckets into continuous queue episodes.
          4. Return episodes with ≥ min_vehicles slow vehicles.

        This is a macroscopic shockwave detector: it flags when a queue has
        formed but does not yet pinpoint its spatial extent (use
        get_vehicle_data_at_interval for spatial positions at those timestamps).

        Args:
            start_time:    Window start (seconds).
            end_time:      Window end (seconds).
            min_vehicles:  Minimum simultaneous slow vehicles to flag.
            max_speed_ms:  Speed threshold below which a vehicle is "queued".
            period_secs:   Time bucket width for aggregation.

        Returns:
            List of queue episode dicts: start_s, end_s, duration_s,
            peak_slow_vehicle_count, mean_slow_vehicle_count.
        """
        self._flush()
        pf = float(period_secs)
        spd_sq = max_speed_ms ** 2

        query = f"""
            SELECT
                FLOOR(timestamp / {pf}) * {pf} AS bucket,
                COUNT(DISTINCT track_id)         AS slow_count
            FROM vehicle_trajectories
            WHERE timestamp >= ?
              AND timestamp <= ?
              AND trusted_for_rules
              AND (vel_x * vel_x + vel_y * vel_y) < {spd_sq}
            GROUP BY bucket
            HAVING slow_count >= ?
            ORDER BY bucket
        """
        rows = _to_dicts(self.conn.execute(query, (start_time, end_time, min_vehicles)))
        if not rows:
            return []

        # Merge adjacent buckets (gap ≤ 1 bucket apart) into episodes
        episodes = []
        ep_start = ep_end = float(rows[0]["bucket"])
        peak = mean_acc = int(rows[0]["slow_count"])
        count_in_ep = 1

        for row in rows[1:]:
            bucket = float(row["bucket"])
            slow = int(row["slow_count"])
            if bucket - ep_end <= pf * 1.5:  # contiguous
                ep_end = bucket
                peak = max(peak, slow)
                mean_acc += slow
                count_in_ep += 1
            else:
                episodes.append({
                    "start_s":                  round(ep_start, 1),
                    "end_s":                    round(ep_end + pf, 1),
                    "duration_s":               round(ep_end + pf - ep_start, 1),
                    "peak_slow_vehicle_count":  peak,
                    "mean_slow_vehicle_count":  round(mean_acc / count_in_ep, 1),
                })
                ep_start = ep_end = bucket
                peak = mean_acc = slow
                count_in_ep = 1

        episodes.append({
            "start_s":                  round(ep_start, 1),
            "end_s":                    round(ep_end + pf, 1),
            "duration_s":               round(ep_end + pf - ep_start, 1),
            "peak_slow_vehicle_count":  peak,
            "mean_slow_vehicle_count":  round(mean_acc / count_in_ep, 1),
        })
        return episodes

    # ------------------------------------------------------------------
    # Turning Movement Count (TMC)
    # ------------------------------------------------------------------

    def get_turning_movement_counts(
        self,
        start_time: float = 0.0,
        end_time: float = 9_999_999.0,
    ) -> List[dict]:
        """
        Computes a Turning Movement Count (TMC) matrix from zone gate crossings.

        Each vehicle's first confirmed/estimated entry gate is its approach arm;
        its last exit gate is its departure arm.  The (entry, exit) pair is the
        turn movement (e.g. North→South = through, North→East = right turn).

        Vehicle class is joined from vehicle_trajectories so counts are broken
        down by vehicle type (car, truck, motorcycle, etc.).

        Args:
            start_time: Window start (seconds).
            end_time:   Window end (seconds).

        Returns:
            List of dicts: origin_gate, destination_gate, class_label,
            vehicle_count, sorted by count descending.
        """
        self._flush()
        query = """
            WITH entries AS (
                SELECT
                    track_id,
                    gate_name  AS origin_gate,
                    MIN(timestamp) AS entry_time
                FROM zone_crossings
                WHERE direction = 'enter'
                  AND timestamp >= ? AND timestamp <= ?
                GROUP BY track_id, gate_name
            ),
            exits AS (
                SELECT
                    track_id,
                    gate_name  AS dest_gate,
                    MAX(timestamp) AS exit_time
                FROM zone_crossings
                WHERE direction = 'exit'
                  AND timestamp >= ? AND timestamp <= ?
                GROUP BY track_id, gate_name
            ),
            classes AS (
                SELECT track_id, MIN(class_label) AS class_label
                FROM vehicle_trajectories
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY track_id
            )
            SELECT
                e.origin_gate,
                x.dest_gate,
                COALESCE(c.class_label, 'unknown') AS class_label,
                COUNT(*)                            AS vehicle_count
            FROM entries e
            JOIN exits    x ON e.track_id = x.track_id
            LEFT JOIN classes c ON e.track_id = c.track_id
            GROUP BY e.origin_gate, x.dest_gate, c.class_label
            ORDER BY vehicle_count DESC
        """
        return _to_dicts(self.conn.execute(
            query,
            (start_time, end_time, start_time, end_time, start_time, end_time),
        ))

    def close(self) -> None:
        """Flushes any remaining buffered rows and closes the database connection."""
        self._flush()
        self.conn.close()
