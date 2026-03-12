from pathlib import Path
from typing import Dict, List
import duckdb

FLUSH_EVERY_N_FRAMES = 100

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

        # Persistent appender: stays open for the lifetime of the pipeline.
        # Avoids the open/flush/close overhead on every single frame.
        self._appender = self.conn.appender("vehicle_trajectories")
        self._frames_since_flush = 0

    def _initialize_schema(self) -> None:
        """Creates tables and indexes for both physics and zone-crossing data."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_trajectories (
                timestamp DOUBLE,
                frame_id  UINTEGER,
                track_id  UINTEGER,
                pos_x     DOUBLE,
                pos_y     DOUBLE,
                vel_x     DOUBLE,
                vel_y     DOUBLE,
                accel_x   DOUBLE,
                accel_y   DOUBLE
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

    def insert_state_vectors(self, timestamp: float, frame_id: int, state_vectors: Dict[int, List[float]]):
        """
        Rapidly ingests the dictionary output from kinematics.py.
        Uses a persistent DuckDB Appender that stays open across frames,
        flushing to disk every FLUSH_EVERY_N_FRAMES frames.

        Args:
            timestamp: The current video timestamp (seconds).
            frame_id: The current video frame number.
            state_vectors: Format {track_id: [x, y, v_x, v_y, a_x, a_y]}
        """
        if not state_vectors:
            return

        for track_id, sv in state_vectors.items():
            # sv corresponds to [pos_x, pos_y, vel_x, vel_y, accel_x, accel_y]
            self._appender.append_row(
                timestamp, frame_id, track_id,
                sv[0], sv[1], sv[2], sv[3], sv[4], sv[5]
            )

        self._frames_since_flush += 1
        if self._frames_since_flush >= FLUSH_EVERY_N_FRAMES:
            self._appender.flush()
            self._frames_since_flush = 0

    def get_trajectory_window(self, start_time: float, end_time: float, track_id: int):
        """
        Tool for the LangGraph Agent: Retrieves the smoothed physics data 
        for a specific vehicle during a specific semantic event window.
        
        Returns:
            A Pandas DataFrame containing the trajectory.
        """
        query = """
            SELECT timestamp, pos_x, pos_y, vel_x, vel_y, accel_x, accel_y 
            FROM vehicle_trajectories 
            WHERE track_id =? AND timestamp >=? AND timestamp <=?
            ORDER BY timestamp ASC
        """
        # Returns a pandas dataframe for easy analytical processing by the agent
        return self.conn.execute(query, (track_id, start_time, end_time)).df()

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
        counts_df = self.conn.execute(
            f"""
            SELECT gate_name, direction, COUNT(*) AS cnt
            FROM zone_crossings
            WHERE {where}
            GROUP BY gate_name, direction
            ORDER BY gate_name, direction
            """,
            params,
        ).df()

        gate_counts: dict = {}
        for _, row in counts_df.iterrows():
            g = row["gate_name"]
            if g not in gate_counts:
                gate_counts[g] = {"enter": 0, "exit": 0}
            gate_counts[g][row["direction"]] = int(row["cnt"])

        # OD pairs — join each vehicle's first entry with its last exit.
        # confidence = 'confirmed' only if BOTH entry and exit were confirmed.
        od_df = self.conn.execute(
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
        ).df()

        od_pairs = od_df.to_dict(orient="records")

        return {"gate_counts": gate_counts, "od_pairs": od_pairs}

    def close(self) -> None:
        """Flushes any remaining buffered rows and closes the database connection."""
        self._appender.flush()
        self._appender.close()
        self.conn.close()