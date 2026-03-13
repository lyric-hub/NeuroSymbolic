"""
LangGraph Agent Tools — Neuro-Symbolic Bridge
==============================================
Five LangChain tools that expose the hybrid memory layer to the ReAct agent.
Each tool maps to a distinct reasoning modality:

  search_semantic_events    → Milvus ANN search       (neural)
  query_graph_relationships → Kùzu Cypher traversal   (symbolic)
  verify_physics_math       → DuckDB kinematic stats  (symbolic)
  evaluate_traffic_rules    → Rule Engine              (symbolic / deterministic)
  query_zone_flow           → DuckDB OD analysis      (symbolic)

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
    df = _get_duckdb().get_trajectory_window(start_time, end_time, track_id)

    if df.empty:
        return (
            f"No physics data found for Vehicle {track_id} "
            f"in t={start_time}–{end_time}s."
        )

    # True speed magnitude per row, then take row-wise max.
    # Bug note: taking max(|vel_x|) and max(|vel_y|) separately and combining
    # overestimates — the maxima may occur at different frames.
    df = df.copy()
    df["_speed"] = (df["vel_x"] ** 2 + df["vel_y"] ** 2) ** 0.5

    # Signed acceleration: positive = speeding up, negative = braking.
    df["_accel_mag"] = (df["accel_x"] ** 2 + df["accel_y"] ** 2) ** 0.5
    dot = df["vel_x"] * df["accel_x"] + df["vel_y"] * df["accel_y"]
    df["_signed_accel"] = df["_accel_mag"].where(dot >= 0, -df["_accel_mag"])

    max_speed = float(df["_speed"].max())
    min_signed_accel = float(df["_signed_accel"].min())

    analysis = {
        "vehicle_id": track_id,
        "time_window": f"{start_time}-{end_time}",
        "max_speed_ms": round(max_speed, 2),
        "max_speed_kmh": round(max_speed * 3.6, 1),
        "min_signed_accel_ms2": round(min_signed_accel, 2),
        "hard_braking_detected": bool(min_signed_accel < -4.0),
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
# Tool 5 — Symbolic Rule Engine (deterministic)
# ---------------------------------------------------------------------------

@tool
def evaluate_traffic_rules(
    track_id: int,
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
) -> str:
    """
    Symbolic Rule Engine — deterministic, auditable traffic violation detection.

    Evaluates explicit safety rules against kinematic data from DuckDB.
    Each result includes the exact numeric evidence that triggered the rule,
    making findings fully citable.

    Rules evaluated:
      SPEEDING               — peak speed > 50 km/h (13.89 m/s)
      HARD_BRAKING           — signed deceleration < −4.0 m/s²
      AGGRESSIVE_ACCELERATION — acceleration magnitude > 3.5 m/s²

    Args:
        track_id:   Integer vehicle ID (e.g. 4 for 'Vehicle 4').
        start_time: Window start in seconds (default: 0.0).
        end_time:   Window end in seconds (default: end of video).
    """
    log.info(
        "Tool: evaluate_traffic_rules | vehicle=%d, t=%.1f-%.1f",
        track_id, start_time, end_time,
    )
    df = _get_duckdb().get_trajectory_window(start_time, end_time, track_id)

    if df.empty:
        return (
            f"No kinematic data found for Vehicle {track_id} "
            f"in t={start_time}–{end_time}s."
        )

    engine = TrafficRuleEngine()
    violations = engine.evaluate(df, track_id)

    if not violations:
        return json.dumps({
            "vehicle_id": track_id,
            "result": "NO_VIOLATIONS",
            "message": "No traffic rule violations detected in this time window.",
        }, indent=2)

    return json.dumps({
        "vehicle_id": track_id,
        "violation_count": len(violations),
        "violations": [v.to_dict() for v in violations],
    }, indent=2)
