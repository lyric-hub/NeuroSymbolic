"""
LangGraph Agent Tools — Neuro-Symbolic Bridge
==============================================
Twenty LangChain tools that expose the hybrid memory layer to the ReAct agent.
Each tool maps to a distinct reasoning modality:

  search_semantic_events      → Milvus ANN search (event-level)      (neural)
  search_entity_profiles      → Milvus ANN search (vehicle-level)    (neural)
  query_graph_relationships   → Kùzu Cypher traversal                (symbolic)
  verify_physics_math         → DuckDB kinematic stats + landmark     (symbolic)
  evaluate_traffic_rules      → Rule Engine (auto zone config)       (symbolic / deterministic)
  query_zone_flow             → DuckDB OD analysis                   (symbolic)
  get_vehicle_trajectory      → DuckDB spatial path reconstruction   (symbolic)
  get_vehicle_proximity       → DuckDB min-distance between vehicles (symbolic)
  compare_vehicle_kinematics  → DuckDB multi-vehicle stats           (symbolic)
  get_vehicle_data_at_interval→ DuckDB interval-based snapshots      (symbolic)
  compute_ttc                 → Time-to-Collision analysis            (symbolic)
  get_speed_statistics        → 85th-percentile + mean + max speed   (symbolic)
  get_vehicle_count_report    → Flow rate / volume per time period   (symbolic)
  detect_traffic_queue        → Queue / congestion episode detection  (symbolic)
  get_turning_movement_counts → Gate-level TMC matrix                (symbolic)

  --- Explainability tools ---
  get_data_quality_report     → Real vs interpolated frames, detection confidence (symbolic)
  explain_threshold           → Source standard behind each rule threshold        (symbolic)
  link_violation_to_conflict  → Write CAUSES edge to Kùzu causal graph           (symbolic)
  get_reasoning_trace         → Retrieve prior session's tool call audit log      (symbolic)

Tools are lazy-initialised: DB connections open on first use, not at
import time, so the FastAPI server starts cleanly even when a DB is not
yet populated.
"""

import json
import logging

from langchain_core.tools import tool

from src.memory_layer.milvus_client import SemanticVectorStore
from src.memory_layer.graph_client import GraphClient
from src.memory_layer.duckdb_client import DuckDBClient
from src.symbolic_engine.rule_engine import TrafficRuleEngine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy DB singletons — created once, reused across tool calls.
# ---------------------------------------------------------------------------
_milvus_db: SemanticVectorStore | None = None
_graph_db: GraphClient | None = None
_duckdb_db: DuckDBClient | None = None


def _get_milvus() -> SemanticVectorStore:
    global _milvus_db
    if _milvus_db is None:
        _milvus_db = SemanticVectorStore()
    return _milvus_db


def _get_graph() -> GraphClient:
    global _graph_db
    if _graph_db is None:
        _graph_db = GraphClient()
    return _graph_db


def _get_duckdb() -> DuckDBClient:
    global _duckdb_db
    if _duckdb_db is None:
        _duckdb_db = DuckDBClient()
    return _duckdb_db


# ---------------------------------------------------------------------------
# Tool 1 — Neural semantic search (Milvus)
# ---------------------------------------------------------------------------

@tool
def search_entity_profiles(query: str) -> str:
    """
    Behavioral Profile Search over per-vehicle longitudinal summaries (Milvus entity_profiles).

    Use this tool for global behavioral queries that span the FULL video, such as:
      - "Which vehicle was speeding the most?"
      - "Find the most aggressive driver."
      - "Which vehicle hard-braked the most times?"

    Unlike search_semantic_events (which finds specific frame-level events),
    this tool searches longitudinal summaries accumulated over the entire video,
    one per vehicle.  It answers ENTITY-LEVEL questions, not EVENT-LEVEL questions.

    Pass the user's natural language behavioral description directly.

    Returns a list with: similarity_score, track_id, summary, first_seen, last_seen.
    """
    log.info("Tool: search_entity_profiles | query='%s'", query)
    results = _get_milvus().search_entity_profiles(query, top_k=3)

    if not results:
        return (
            "No entity profiles found. "
            "Ensure the video has been processed and vehicles were observed."
        )

    return json.dumps(results, indent=2)


@tool
def search_semantic_events(query: str) -> str:
    """
    Semantic Search over VLM-generated event descriptions (Milvus vector DB).

    ALWAYS call this tool FIRST when the user asks about an event in the
    video (e.g. 'a crash', 'speeding in the rain', 'near-miss').
    Pass the user's natural language query directly.

    Returns a list of matching events and their 'time_window_pointer'
    (e.g. '10.0-20.0'), which is needed to query the graph and physics tools.
    """
    log.info("Tool: search_semantic_events | query='%s'", query)
    results = _get_milvus().search_semantic_events(query, top_k=2)

    if not results:
        return "No matching events found in the video."

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Tool 2 — Symbolic graph traversal (Kùzu)
# ---------------------------------------------------------------------------

@tool
def query_graph_relationships(cypher_query: str) -> str:
    """
    Structural/Relational Search over the typed entity graph (Kùzu, Cypher).

    Use this tool AFTER search_semantic_events to find which entities
    interacted during a specific time window.

    Node labels: Vehicle, Pedestrian, Infrastructure.
    Relationship: INTERACTS_WITH with properties predicate, timestamp,
                  trajectory_time_window, motion_state, phase.

    Example Cypher:
      MATCH (s)-[r:INTERACTS_WITH]->(o)
      WHERE r.trajectory_time_window = '10.0-20.0'
      RETURN s.name, r.predicate, o.name

    Temporal query (PRECEDES chain):
      MATCH (v:Vehicle {name:'Vehicle 4'})-[r:PRECEDES*1..3]->(v)
      RETURN r[*].from_window, r[*].to_window
    """
    log.info("Tool: query_graph_relationships | cypher='%s'", cypher_query[:120])
    results = _get_graph().query_graph(cypher_query)

    if not results or (isinstance(results[0], dict) and "error" in results[0]):
        return f"Graph query returned no results or failed: {results}"

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Tool 3 — Symbolic kinematic verification (DuckDB)
# ---------------------------------------------------------------------------

@tool
def verify_physics_math(start_time: float, end_time: float, track_id: int) -> str:
    """
    Deterministic Math Verification using raw kinematic data (DuckDB).

    Retrieves high-frequency Savitzky-Golay smoothed trajectory data for a
    specific vehicle and computes:
      - Peak speed (m/s and km/h) as the true vector magnitude max
      - Minimum signed acceleration (negative = braking)
      - Hard-braking flag (signed accel < -4.0 m/s²)

    Args:
        start_time: Window start in seconds.
        end_time:   Window end in seconds.
        track_id:   Integer vehicle ID (e.g. 4 for 'Vehicle 4').
    """
    log.info(
        "Tool: verify_physics_math | vehicle=%d, t=%.1f-%.1f",
        track_id, start_time, end_time,
    )
    df = _get_duckdb().get_trajectory_window(
        start_time, end_time, track_id, trusted_only=True
    )

    if df.empty:
        return (
            f"No physics data found for Vehicle {track_id} "
            f"in t={start_time}–{end_time}s."
        )

    # True speed magnitude per row, then take row-wise max.
    # Bug note: taking max(|vel_x|) and max(|vel_y|) separately and combining
    # overestimates — the maxima may occur at different frames.
    df = df.copy()
    df["_speed"] = df["speed_ms"].fillna(
        (df["vel_x"] ** 2 + df["vel_y"] ** 2) ** 0.5
    )

    # Signed acceleration: positive = speeding up, negative = braking.
    df["_accel_mag"] = (df["accel_x"] ** 2 + df["accel_y"] ** 2) ** 0.5
    dot = df["vel_x"] * df["accel_x"] + df["vel_y"] * df["accel_y"]
    df["_signed_accel"] = df["_accel_mag"].where(dot >= 0, -df["_accel_mag"])

    max_speed = float(df["_speed"].max())
    min_signed_accel = float(df["_signed_accel"].min())

    # Position and nearest landmark at the moment of peak speed.
    peak_row = df.loc[df["_speed"].idxmax()]
    landmark, lm_dist = _get_duckdb().nearest_landmark(
        float(peak_row["pos_x"]), float(peak_row["pos_y"])
    )

    analysis = {
        "vehicle_id": track_id,
        "time_window": f"{start_time}-{end_time}",
        "max_speed_ms": round(max_speed, 2),
        "max_speed_kmh": round(max_speed * 3.6, 1),
        "min_signed_accel_ms2": round(min_signed_accel, 2),
        "hard_braking_detected": bool(min_signed_accel < -4.0),
        "position_at_peak_speed": f"{lm_dist:.1f}m from {landmark}",
        "data_points": len(df),
    }

    return json.dumps(analysis, indent=2)


# ---------------------------------------------------------------------------
# Tool 4 — Zone flow / OD analysis (DuckDB)
# ---------------------------------------------------------------------------

@tool
def query_zone_flow(
    gate_name: str = "",
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
) -> str:
    """
    Zone Flow and Origin-Destination (OD) Analysis (DuckDB zone_crossings).

    Use when the user asks about vehicle counts, flow, entry/exit events,
    origin-destination pairs, or dwell times for a defined zone.

    Args:
        gate_name:  Filter to a specific gate name (e.g. 'North'). Leave
                    empty to include all gates.
        start_time: Window start in seconds (default: 0.0).
        end_time:   Window end in seconds (default: end of video).

    Returns JSON with:
      gate_counts — per-gate enter/exit counts.
      od_pairs    — list of (track_id, origin_gate, dest_gate, dwell_time_s).
    """
    log.info(
        "Tool: query_zone_flow | gate='%s', t=%.1f-%.1f",
        gate_name, start_time, end_time,
    )
    result = _get_duckdb().query_zone_flow(
        gate_name=gate_name,
        start_time=start_time,
        end_time=end_time,
    )

    if not result["gate_counts"] and not result["od_pairs"]:
        return (
            "No zone crossing data found. "
            "Ensure zone_config.json exists and the video has been processed."
        )

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 5 — Vehicle class lookup (DuckDB)
# ---------------------------------------------------------------------------

@tool
def search_vehicles_by_type(vehicle_type: str) -> str:
    """
    Look up all track_ids matching a specific vehicle class detected by YOLO.

    Use this tool FIRST when the user asks about a category of vehicles, such as:
      - "two-wheelers" / "motorcycles" / "bikes"  → pass "motorcycle"
      - "bicycles"                                 → pass "bicycle"
      - "cars"                                     → pass "car"
      - "buses"                                    → pass "bus"
      - "trucks"                                   → pass "truck"
      - "pedestrians" / "people"                   → pass "person"

    For two-wheelers, call this tool TWICE: once with "motorcycle" and once
    with "bicycle" to capture both sub-types.

    Returns track_id, class_label, first_seen, last_seen, frame_count for every
    matching vehicle. Use the returned track_ids with verify_physics_math and
    evaluate_traffic_rules to analyse their behaviour.
    """
    log.info("Tool: search_vehicles_by_type | type='%s'", vehicle_type)
    results = _get_duckdb().get_vehicles_by_class(vehicle_type)

    if not results:
        return (
            f"No vehicles of type '{vehicle_type}' found. "
            "Either none were detected or the video has not been processed yet."
        )

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Tool 6 — Symbolic Rule Engine (deterministic)
# ---------------------------------------------------------------------------

@tool
def evaluate_traffic_rules(
    track_id: int,
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
    collision_timestamp: float = -1.0,
) -> str:
    """
    Symbolic Rule Engine — deterministic, auditable traffic violation detection.

    Evaluates explicit safety rules against kinematic data from DuckDB.
    Each result includes the exact numeric evidence that triggered the rule,
    making findings fully citable.

    Rules evaluated:
      SPEEDING               — speed > posted zone limit (default 50 km/h)
      HARD_BRAKING           — signed deceleration < −3.0 m/s²
      AGGRESSIVE_ACCELERATION — signed forward acceleration > 3.0 m/s²
      STATIONARY_IN_LANE     — stopped < 0.5 m/s for > 10 s in live lane
      WRONG_WAY              — heading > 90° from zone flow_direction_deg
                               (only fires when zone_config.json sets flow_direction_deg)

    Args:
        track_id:            Integer vehicle ID (e.g. 4 for 'Vehicle 4').
        start_time:          Window start in seconds (default: 0.0).
        end_time:            Window end in seconds (default: end of video).
        collision_timestamp: If the vehicle was involved in a collision, pass the
                             collision time here (seconds). STATIONARY_IN_LANE and
                             HARD_BRAKING violations that occur at or after this
                             timestamp are suppressed — they are collision consequences,
                             not driver violations. Pass -1.0 (default) to disable.
    """
    log.info(
        "Tool: evaluate_traffic_rules | vehicle=%d, t=%.1f-%.1f",
        track_id, start_time, end_time,
    )
    df = _get_duckdb().get_trajectory_window(
        start_time, end_time, track_id, trusted_only=True
    )

    if df.empty:
        return (
            f"No kinematic data found for Vehicle {track_id} "
            f"in t={start_time}–{end_time}s."
        )

    # Auto-load per-zone thresholds from zone_config.json if it exists.
    # Enables school zones (30 km/h), motorways (130 km/h), etc. automatically.
    speed_limit_ms = None
    flow_direction_deg = None
    try:
        from src.physics_engine.zone_manager import ZoneConfig as _ZC
        zc = _ZC.from_json("zone_config.json")
        speed_limit_ms = zc.speed_limit_kmh / 3.6
        flow_direction_deg = zc.flow_direction_deg
    except Exception:
        pass  # no zone config or malformed — fall back to global defaults

    engine = TrafficRuleEngine()
    violations = engine.evaluate(
        df, track_id,
        speed_limit_ms=speed_limit_ms,
        flow_direction_deg=flow_direction_deg,
        collision_timestamp=collision_timestamp if collision_timestamp >= 0 else None,
    )

    # Collect assumptions from the first violation (all share the same assumptions)
    # or from the evaluation context directly.
    eval_assumptions = []
    if speed_limit_ms is None:
        eval_assumptions.append(
            "speed_limit_kmh defaulted to 50 km/h — zone_config.json not found or field absent"
        )
    if flow_direction_deg is None:
        eval_assumptions.append(
            "WRONG_WAY rule disabled — flow_direction_deg not set in zone_config.json"
        )

    active_limit_kmh = round(speed_limit_ms * 3.6, 0) if speed_limit_ms else 50.0

    if not violations:
        return json.dumps({
            "vehicle_id":        track_id,
            "result":            "NO_VIOLATIONS",
            "speed_limit_kmh":   active_limit_kmh,
            "message": (
                f"No traffic rule violations detected in this time window "
                f"(speed limit applied: {active_limit_kmh:.0f} km/h)."
            ),
            "_assumptions": eval_assumptions,
        }, indent=2)

    # Feedback loop — write violations back to Kùzu so the graph has
    # persistent memory of rule verdicts, queryable via Cypher.
    time_window = f"{start_time}-{end_time}"
    graph = _get_graph()
    for v in violations:
        vd = v.to_dict()
        graph.insert_violation(
            track_id=track_id,
            violation_type=vd.get("rule_id", "UNKNOWN"),
            time_window=time_window,
            evidence=vd.get("evidence", {}),
        )

    return json.dumps({
        "vehicle_id":      track_id,
        "violation_count": len(violations),
        "speed_limit_kmh": active_limit_kmh,
        "violations":      [v.to_dict() for v in violations],
        "_assumptions":    eval_assumptions,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 7 — Spatial trajectory path (DuckDB)
# ---------------------------------------------------------------------------

@tool
def get_vehicle_trajectory(
    track_id: int,
    start_time: float,
    end_time: float,
) -> str:
    """
    Spatial Trajectory Reconstruction — full position sequence for a vehicle.

    Returns a sampled sequence of (timestamp, pos_x, pos_y, speed, nearest_landmark)
    so the agent can reconstruct WHERE a vehicle was at each moment, not just
    peak statistics. Essential for accident spatial analysis.

    Use AFTER verify_physics_math when you need to know the vehicle's path,
    not just its peak speed — e.g. "was it in the intersection when it braked?"

    Args:
        track_id:   Integer vehicle ID.
        start_time: Window start in seconds.
        end_time:   Window end in seconds.
    """
    log.info(
        "Tool: get_vehicle_trajectory | vehicle=%d, t=%.1f-%.1f",
        track_id, start_time, end_time,
    )
    path = _get_duckdb().get_trajectory_path(track_id, start_time, end_time)

    if not path:
        return (
            f"No trajectory data found for Vehicle {track_id} "
            f"in t={start_time}–{end_time}s."
        )

    return json.dumps({
        "vehicle_id": track_id,
        "time_window": f"{start_time}-{end_time}",
        "point_count": len(path),
        "path": path,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 8 — Vehicle proximity (DuckDB)
# ---------------------------------------------------------------------------

@tool
def get_vehicle_proximity(
    track_id_a: int,
    track_id_b: int,
    start_time: float,
    end_time: float,
) -> str:
    """
    Proximity Analysis — minimum tail-to-head gap between two vehicles.

    Answers: "How close did Vehicle 4 get to Vehicle 9? Did they actually collide?"

    Computes gap = centre_distance - half_length_A - half_length_B using
    class-based vehicle dimensions (car≈4.5m, bus≈12m, motorcycle≈2m).
    gap ≤ 0 means collision confirmed (bodies overlapping).
    gap < 1m means near-miss.

    Use AFTER query_graph_relationships identifies involved vehicles.

    Args:
        track_id_a: First vehicle ID.
        track_id_b: Second vehicle ID.
        start_time: Window start in seconds.
        end_time:   Window end in seconds.

    Returns JSON with: min_gap_m, centre_distance_m, collision_confirmed,
    timestamp_s, vehicle_a/b_half_length_m, vehicle_a/b_pos.
    """
    log.info(
        "Tool: get_vehicle_proximity | vehicles=%d,%d, t=%.1f-%.1f",
        track_id_a, track_id_b, start_time, end_time,
    )
    result = _get_duckdb().get_vehicle_proximity(
        track_id_a, track_id_b, start_time, end_time
    )

    if "error" in result:
        return result["error"]

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 9 — Multi-vehicle kinematic comparison (DuckDB)
# ---------------------------------------------------------------------------

@tool
def compare_vehicle_kinematics(
    track_ids_csv: str,
    start_time: float,
    end_time: float,
) -> str:
    """
    Multi-Vehicle Kinematic Comparison — side-by-side stats for multiple vehicles.

    Returns speed and acceleration statistics for all specified vehicles in
    one call, aligned to the same time window. Essential for comparing
    involved parties at the exact moment of an incident.

    Use when you need to compare Vehicle 4 and Vehicle 9 simultaneously
    rather than calling verify_physics_math separately for each.

    Args:
        track_ids_csv: Comma-separated vehicle IDs, e.g. "4,9" or "4,9,12".
        start_time:    Window start in seconds.
        end_time:      Window end in seconds.
    """
    log.info(
        "Tool: compare_vehicle_kinematics | ids='%s', t=%.1f-%.1f",
        track_ids_csv, start_time, end_time,
    )
    try:
        ids = [int(x.strip()) for x in track_ids_csv.split(",") if x.strip()]
    except ValueError:
        return "Invalid track_ids_csv format. Use comma-separated integers e.g. '4,9'."

    if not ids:
        return "No vehicle IDs provided."

    db = _get_duckdb()
    results = []

    for track_id in ids:
        df = db.get_trajectory_window(
            start_time, end_time, track_id, trusted_only=True
        )
        if df.empty:
            results.append({
                "vehicle_id": track_id,
                "status": "no_data",
                "time_window": f"{start_time}-{end_time}",
            })
            continue

        df = df.copy()
        df["_speed"] = df["speed_ms"].fillna(
            (df["vel_x"] ** 2 + df["vel_y"] ** 2) ** 0.5
        )
        df["_accel_mag"] = (df["accel_x"] ** 2 + df["accel_y"] ** 2) ** 0.5
        dot = df["vel_x"] * df["accel_x"] + df["vel_y"] * df["accel_y"]
        df["_signed_accel"] = df["_accel_mag"].where(dot >= 0, -df["_accel_mag"])

        max_speed = float(df["_speed"].max())
        min_accel = float(df["_signed_accel"].min())
        peak_row = df.loc[df["_speed"].idxmax()]
        landmark, lm_dist = db.nearest_landmark(float(peak_row["pos_x"]), float(peak_row["pos_y"]))

        results.append({
            "vehicle_id": track_id,
            "time_window": f"{start_time}-{end_time}",
            "max_speed_ms": round(max_speed, 2),
            "max_speed_kmh": round(max_speed * 3.6, 1),
            "min_signed_accel_ms2": round(min_accel, 2),
            "hard_braking_detected": bool(min_accel < -4.0),
            "position_at_peak_speed": f"{lm_dist:.1f}m from {landmark}",
            "data_points": len(df),
        })

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Tool 10 — Interval-based data snapshots (DuckDB)
# ---------------------------------------------------------------------------

@tool
def get_vehicle_data_at_interval(
    interval_secs: float,
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
    track_id: int = -1,
) -> str:
    """
    Interval-Based Data Snapshots — one row per vehicle per time bucket.

    Returns kinematic data sampled at a fixed time interval instead of
    every raw frame.  Useful when the user needs data at a specific rate,
    e.g. "give me data every 0.5 seconds" or "show me where each vehicle
    was every second".

    For each bucket the tool picks the **first** measured sample that falls
    inside it (deterministic — no averaging or interpolation), enriched with:
      speed_ms, signed_accel_ms2, nearest_landmark, distance_to_landmark_m.

    Args:
        interval_secs: Sampling interval in seconds (e.g. 0.5, 1.0, 2.0).
                       Minimum 0.05 s, maximum 3600 s.
        start_time:    Window start in seconds (default: 0.0).
        end_time:      Window end in seconds (default: end of video).
        track_id:      Specific vehicle ID to query, or -1 (default) to
                       return data for ALL vehicles in the window.

    Returns JSON list of snapshot rows.  Each row:
        time_bucket, timestamp, track_id, class_label,
        pos_x, pos_y, speed_ms, signed_accel_ms2,
        nearest_landmark, distance_to_landmark_m.
    """
    log.info(
        "Tool: get_vehicle_data_at_interval | interval=%.2fs, vehicle=%s, t=%.1f-%.1f",
        interval_secs, track_id if track_id >= 0 else "ALL", start_time, end_time,
    )
    db = _get_duckdb()
    try:
        if track_id >= 0:
            rows = db.get_sampled_trajectory(track_id, interval_secs, start_time, end_time)
        else:
            rows = db.get_all_vehicles_sampled(interval_secs, start_time, end_time)
    except ValueError as exc:
        return f"Invalid interval: {exc}"

    if not rows:
        scope = f"Vehicle {track_id}" if track_id >= 0 else "any vehicle"
        return (
            f"No data found for {scope} in t={start_time}–{end_time}s. "
            "Ensure the video has been processed."
        )

    return json.dumps({
        "interval_secs": interval_secs,
        "time_window": f"{start_time}-{end_time}",
        "vehicle_filter": track_id if track_id >= 0 else "all",
        "row_count": len(rows),
        "data": rows,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 11 — Time-to-Collision (TTC) analysis (DuckDB)
# ---------------------------------------------------------------------------

@tool
def compute_ttc(
    track_id_a: int,
    track_id_b: int,
    start_time: float,
    end_time: float,
) -> str:
    """
    Time-to-Collision (TTC) Analysis — industry-standard traffic conflict metric.

    TTC = gap / closing_speed.
    Only defined when two vehicles are approaching (closing_speed > 0).
    When vehicles are diverging, TTC = infinity (no conflict).

    Conflict severity thresholds (SSAM / Highway Capacity Manual):
      TTC < 1.5 s  → CRITICAL conflict (imminent crash risk)
      TTC < 3.0 s  → SERIOUS conflict  (significant safety hazard)
      TTC ≥ 3.0 s  → LOW (acceptable gap)

    Use this tool to quantify how close two vehicles came to a collision —
    complementing get_vehicle_proximity (which gives minimum gap) by also
    accounting for relative speed and approach trajectory.

    Use AFTER get_vehicle_proximity confirms vehicles were close, to determine
    whether the situation was actually dangerous (fast approach) or benign
    (slow crawl at close range).

    Args:
        track_id_a:  First vehicle (typically the following vehicle).
        track_id_b:  Second vehicle (typically the lead vehicle).
        start_time:  Window start in seconds.
        end_time:    Window end in seconds.

    Returns JSON with: min_ttc_s, conflict_level, timestamp_of_min_ttc_s,
    gap_at_min_ttc_m, closing_speed_ms, vehicle_a/b_pos, ttc_series.
    """
    log.info(
        "Tool: compute_ttc | vehicles=%d,%d, t=%.1f-%.1f",
        track_id_a, track_id_b, start_time, end_time,
    )
    result = _get_duckdb().compute_ttc(track_id_a, track_id_b, start_time, end_time)

    if "error" in result:
        return result["error"]

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 12 — Speed statistics / 85th percentile speed (DuckDB)
# ---------------------------------------------------------------------------

@tool
def get_speed_statistics(
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
    track_id: int = -1,
) -> str:
    """
    Speed Statistics — 85th percentile (V85), mean, and maximum speed per vehicle.

    The 85th percentile speed (V85) is the key design speed metric used in road
    safety audits and speed limit review.  It represents the speed below which
    85% of vehicles travel — if V85 exceeds the posted limit, the limit may need
    enforcement or re-assessment.

    Use to answer:
      - "What is the 85th percentile speed on this road section?"
      - "Are vehicles generally complying with the speed limit?"
      - "Which vehicle class drives fastest on average?"

    Args:
        start_time: Window start in seconds (default: full video).
        end_time:   Window end in seconds (default: full video).
        track_id:   Specific vehicle ID, or -1 (default) for all vehicles.

    Returns JSON list per vehicle with:
        track_id, class_label, v85_speed_kmh, mean_speed_kmh, max_speed_kmh,
        data_points.
    """
    log.info(
        "Tool: get_speed_statistics | vehicle=%s, t=%.1f-%.1f",
        track_id if track_id >= 0 else "ALL", start_time, end_time,
    )
    rows = _get_duckdb().get_speed_statistics(track_id, start_time, end_time)

    if not rows:
        return (
            "No speed data found. "
            "Ensure the video has been processed."
        )

    return json.dumps({
        "time_window": f"{start_time}-{end_time}",
        "vehicle_filter": track_id if track_id >= 0 else "all",
        "vehicle_count": len(rows),
        "statistics": rows,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 13 — Vehicle count / flow rate by time period (DuckDB)
# ---------------------------------------------------------------------------

@tool
def get_vehicle_count_report(
    period_secs: float = 3600.0,
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
) -> str:
    """
    Vehicle Count and Flow Rate Report — vehicles observed per time period.

    Counts distinct vehicles detected in each fixed-width time bucket and
    normalises to vehicles-per-hour (flow rate).  Essential for:
      - Volume studies (AADT estimation)
      - Peak Hour Factor (PHF) analysis — use period_secs=900 (15-min buckets)
      - Identifying congestion onset times

    PHF = total_peak_hour_count / (4 × peak_15min_count).
    PHF close to 1.0 means steady flow; low PHF means heavy bursts.

    Args:
        period_secs: Bucket width in seconds.
                     Common values: 60 (per minute), 900 (15-min for PHF),
                     3600 (hourly, default).
        start_time:  Window start in seconds (default: full video).
        end_time:    Window end in seconds (default: full video).

    Returns JSON list of time periods with:
        period_start_s, period_end_s, vehicle_count, vehicles_per_hour.
    """
    log.info(
        "Tool: get_vehicle_count_report | period=%.0fs, t=%.1f-%.1f",
        period_secs, start_time, end_time,
    )
    rows = _get_duckdb().get_vehicle_count_by_period(period_secs, start_time, end_time)

    if not rows:
        return "No vehicle data found in the specified window."

    total = sum(r["vehicle_count"] for r in rows)
    peak = max(rows, key=lambda r: r["vehicle_count"])

    return json.dumps({
        "period_secs": period_secs,
        "time_window": f"{start_time}-{end_time}",
        "total_vehicles": total,
        "peak_period": peak,
        "periods": rows,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 14 — Queue / congestion detection (DuckDB)
# ---------------------------------------------------------------------------

@tool
def detect_traffic_queue(
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
    min_vehicles: int = 3,
    max_speed_kmh: float = 3.6,
) -> str:
    """
    Traffic Queue / Congestion Detection — finds episodes where ≥ N vehicles
    are simultaneously near-stationary (speed < threshold).

    Detects shockwave formation (queue back-propagation) and sustained
    congestion events.  Returns continuous queue episodes with duration
    and peak slow-vehicle count.

    Use to answer:
      - "Was there a queue at t=60s?"
      - "How long did the congestion last?"
      - "What is the worst congestion episode in the video?"

    After finding a queue episode, use get_vehicle_data_at_interval with the
    episode start/end times to get vehicle positions inside the queue.

    Args:
        start_time:     Window start in seconds (default: full video).
        end_time:       Window end in seconds (default: full video).
        min_vehicles:   Minimum simultaneous slow vehicles to declare a queue
                        (default 3).
        max_speed_kmh:  Speed threshold below which a vehicle is "queued"
                        (default 3.6 km/h = 1 m/s — nearly stopped).

    Returns JSON list of queue episodes with:
        start_s, end_s, duration_s,
        peak_slow_vehicle_count, mean_slow_vehicle_count.
    """
    log.info(
        "Tool: detect_traffic_queue | min_vehicles=%d, max_speed=%.1f km/h, t=%.1f-%.1f",
        min_vehicles, max_speed_kmh, start_time, end_time,
    )
    max_speed_ms = max_speed_kmh / 3.6
    episodes = _get_duckdb().detect_queues(
        start_time=start_time,
        end_time=end_time,
        min_vehicles=min_vehicles,
        max_speed_ms=max_speed_ms,
    )

    if not episodes:
        return (
            f"No queue episodes found (≥{min_vehicles} vehicles at < {max_speed_kmh} km/h "
            f"simultaneously) in t={start_time}–{end_time}s."
        )

    return json.dumps({
        "time_window": f"{start_time}-{end_time}",
        "min_vehicles_threshold": min_vehicles,
        "max_speed_threshold_kmh": max_speed_kmh,
        "episode_count": len(episodes),
        "episodes": episodes,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 15 — Turning Movement Count (TMC) matrix (DuckDB)
# ---------------------------------------------------------------------------

@tool
def get_turning_movement_counts(
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
) -> str:
    """
    Turning Movement Count (TMC) Matrix — gate-level intersection flow breakdown.

    Pairs each vehicle's entry gate (approach arm) with its exit gate (departure
    arm) to produce the standard TMC matrix used in:
      - Intersection capacity analysis (HCM methodology)
      - Signal timing optimisation
      - Conflict point identification (left-turn vs through conflicts)
      - Mode split analysis (trucks vs cars vs motorcycles)

    Gate naming conventions (example for 4-arm intersection):
      North→South = through movement from North approach
      North→East  = right turn
      North→West  = left turn
      North→North = U-turn

    Requires zone_config.json with named gates.

    Args:
        start_time: Window start in seconds (default: full video).
        end_time:   Window end in seconds (default: full video).

    Returns JSON with a TMC matrix list, each row:
        origin_gate, destination_gate, class_label, vehicle_count.
    Sorted by vehicle_count descending.
    """
    log.info(
        "Tool: get_turning_movement_counts | t=%.1f-%.1f", start_time, end_time,
    )
    rows = _get_duckdb().get_turning_movement_counts(start_time, end_time)

    if not rows:
        return (
            "No turning movement data found. "
            "Ensure zone_config.json defines named gates and the video has been processed."
        )

    total = sum(r["vehicle_count"] for r in rows)
    return json.dumps({
        "time_window": f"{start_time}-{end_time}",
        "total_movements": total,
        "tmc_matrix": rows,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 16 — Data quality report (DuckDB)
# ---------------------------------------------------------------------------

@tool
def get_data_quality_report(
    track_id: int = -1,
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
) -> str:
    """
    Data Quality Report — real vs interpolated frames, YOLO detection confidence,
    and gate crossing confidence for a vehicle or the full video.

    ALWAYS call this tool BEFORE citing evaluate_traffic_rules findings in
    your final answer.  A violation backed by HIGH quality data is a confirmed
    fact; one backed by LOW quality data is only indicative.

    Quality rating:
      HIGH   — ≥ 85% real frames AND mean YOLO confidence ≥ 0.6
      MEDIUM — ≥ 60% real frames AND mean YOLO confidence ≥ 0.4
      LOW    — below MEDIUM thresholds (treat findings as unconfirmed)

    Args:
        track_id:   Vehicle ID, or -1 (default) for all vehicles in the window.
        start_time: Window start in seconds (default: full video).
        end_time:   Window end in seconds (default: full video).

    Returns JSON with: total_frames, real_frames, interpolated_frames,
    coverage_pct, mean_detection_confidence, confirmed/estimated gate crossings,
    data_quality_rating, and a plain-English _interpretation.
    """
    log.info(
        "Tool: get_data_quality_report | vehicle=%s, t=%.1f-%.1f",
        track_id if track_id >= 0 else "ALL", start_time, end_time,
    )
    result = _get_duckdb().get_data_quality_report(track_id, start_time, end_time)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 17 — Threshold provenance / explain threshold (registry)
# ---------------------------------------------------------------------------

@tool
def explain_threshold(rule_name: str) -> str:
    """
    Threshold Provenance — explains the engineering standard behind a rule threshold.

    Use when the user asks WHY a specific numeric threshold was chosen, or when
    citing a rule violation and you need to justify the threshold to a traffic
    engineer audience.

    Available rule names:
      HARD_BRAKING, AGGRESSIVE_ACCELERATION, SPEEDING_DEFAULT,
      TTC_CRITICAL, TTC_SERIOUS, WRONG_WAY_ANGLE, WRONG_WAY_MIN_SPEED,
      WRONG_WAY_MIN_DURATION, STATIONARY_IN_LANE_DURATION, STATIONARY_SPEED,
      V85_DESIGN_SPEED

    Args:
        rule_name: Rule name (case-insensitive, e.g. "HARD_BRAKING" or "ttc critical").

    Returns JSON with: rule, value, unit, source standard, full description.
    """
    log.info("Tool: explain_threshold | rule='%s'", rule_name)
    from src.symbolic_engine.threshold_registry import explain_threshold as _explain
    result = _explain(rule_name)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 18 — Causal link writer (Kùzu CAUSES edge)
# ---------------------------------------------------------------------------

@tool
def link_violation_to_conflict(
    cause_track_id: int,
    cause_rule: str,
    effect_track_id: int,
    effect_type: str,
    time_window: str,
    confidence: float = 0.8,
) -> str:
    """
    Causal Attribution — writes a CAUSES edge in the Kùzu graph linking a rule
    violation to a downstream conflict or event.

    Use this tool ONLY when you have strong evidence (from evaluate_traffic_rules
    + get_vehicle_proximity or compute_ttc) that one vehicle's violation directly
    caused a conflict with another vehicle.  This converts correlation into
    explicit causal memory queryable via Cypher:

        MATCH (c:Vehicle)-[r:CAUSES]->(e:Vehicle)
        RETURN c.name, r.cause_rule, r.effect_type, e.name, r.confidence

    Args:
        cause_track_id:  Track ID of the violating vehicle (the cause).
        cause_rule:      Rule that fired, e.g. 'SPEEDING', 'WRONG_WAY'.
        effect_track_id: Track ID of the vehicle that experienced the effect.
        effect_type:     Type of downstream event: 'TTC_CRITICAL', 'TTC_SERIOUS',
                         'COLLISION', 'HARD_BRAKING', 'NEAR_MISS'.
        time_window:     Time range string, e.g. '5.0-20.0'.
        confidence:      Your confidence in this causal attribution (0.0–1.0).
                         Use 0.9 when physics + rules both corroborate;
                         0.6 when based on sequence inference only.

    Returns confirmation JSON with the inserted edge details.
    """
    log.info(
        "Tool: link_violation_to_conflict | cause=%d [%s] → effect=%d [%s] conf=%.2f",
        cause_track_id, cause_rule, effect_track_id, effect_type, confidence,
    )
    _get_graph().insert_causal_link(
        cause_track_id=cause_track_id,
        cause_rule=cause_rule,
        effect_track_id=effect_track_id,
        effect_type=effect_type,
        time_window=time_window,
        confidence=float(confidence),
        severity_score=severity_score,
    )
    return json.dumps({
        "status": "CAUSES edge written",
        "cause": f"Vehicle {cause_track_id}",
        "cause_rule": cause_rule,
        "effect": f"Vehicle {effect_track_id}",
        "effect_type": effect_type,
        "time_window": time_window,
        "confidence": confidence,
        "cypher_to_query": (
            f"MATCH (c:Vehicle {{name:'Vehicle {cause_track_id}'}})"
            f"-[r:CAUSES]->(e:Vehicle) RETURN c.name, r.cause_rule, r.effect_type, "
            f"e.name, r.confidence"
        ),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 19 — Reasoning trace retrieval (DuckDB analysis_sessions)
# ---------------------------------------------------------------------------

@tool
def get_reasoning_trace(session_id: str) -> str:
    """
    Reasoning Trace — retrieves the full tool-call audit log for a prior query
    session, showing exactly which tools were called, with what arguments, and
    what they returned.

    Use when the user asks "how did you reach that conclusion?" or "show me your
    reasoning for the last analysis."  The session_id is returned in every
    final_summary under the key [Session ID: ...].

    Args:
        session_id: UUID string from a prior analysis (shown in final_summary).

    Returns ordered list of steps: step_number, tool_name, input_args,
    output_excerpt (first 300 chars of tool result), recorded_at timestamp.
    """
    log.info("Tool: get_reasoning_trace | session='%s'", session_id)
    steps = _get_duckdb().get_reasoning_trace(session_id)

    if not steps:
        return (
            f"No reasoning trace found for session '{session_id}'. "
            "Traces are written at the end of each analysis. "
            "Ensure the session_id is correct."
        )

    return json.dumps({
        "session_id": session_id,
        "step_count": len(steps),
        "trace": steps,
    }, indent=2)
