import json
from langchain_core.tools import tool

# Import the database clients we built in the memory_layer
from src.memory_layer.milvus_client import SemanticVectorStore
from src.memory_layer.graph_client import GraphClient
from src.memory_layer.duckdb_client import DuckDBClient

# Import the symbolic rule engine — the deterministic reasoning layer
from src.symbolic_engine.rule_engine import TrafficRuleEngine

# Lazy initialization: connections are created on first use, not at import time.
# This prevents the server from crashing if a DB dependency isn't ready yet.
_milvus_db = None
_graph_db = None
_duckdb_db = None

def _get_milvus():
    global _milvus_db
    if _milvus_db is None:
        _milvus_db = SemanticVectorStore()
    return _milvus_db

def _get_graph():
    global _graph_db
    if _graph_db is None:
        _graph_db = GraphClient()
    return _graph_db

def _get_duckdb():
    global _duckdb_db
    if _duckdb_db is None:
        _duckdb_db = DuckDBClient()
    return _duckdb_db

@tool
def search_semantic_events(query: str) -> str:
    """
    Step 1 in the Sequential Pipeline: Semantic Search.
    Always use this tool FIRST when asked about an event in the video (e.g., 'a crash', 'speeding in the rain').
    Pass the user's natural language query directly to this tool.
    It returns a list of events and their critical 'time_window_pointer' (e.g., '10.0-20.0').
    """
    print(f"🔧 Tool Call: Searching Vector DB for '{query}'...")
    results = _get_milvus().search_semantic_events(query, top_k=2)
    
    if not results:
        return "No matching events found in the video."
        
    return json.dumps(results, indent=2)

@tool
def query_graph_relationships(cypher_query: str) -> str:
    """
    Step 2 in the Sequential Pipeline: Structural/Relational Search.
    Use this tool SECOND after you have obtained a 'time_window_pointer'.
    Write a Cypher query to find which entities interacted during that specific time window.
    Node labels are typed: Vehicle, Pedestrian, or Infrastructure.
    Example Cypher: MATCH (s)-[r:INTERACTS_WITH]->(o) WHERE r.trajectory_time_window = '10.0-20.0' RETURN s.name, r.predicate, o.name
    To filter by type: MATCH (s:Vehicle)-[r:INTERACTS_WITH]->(o:Infrastructure) RETURN s.name, r.predicate, o.name
    """
    print(f"🔧 Tool Call: Traversing Graph DB with Cypher...")
    results = _get_graph().query_graph(cypher_query)
    
    if not results or (isinstance(results[0], dict) and "error" in results[0]):
        return f"Graph query failed or returned no results: {results}"
        
    return json.dumps(results, indent=2)

@tool
def verify_physics_math(start_time: float, end_time: float, track_id: int) -> str:
    """
    Step 3 in the Sequential Pipeline: Deterministic Math Verification.
    Use this tool LAST to verify physical facts (speed, acceleration, braking).
    You must provide the start_time, end_time, and the specific integer track_id (e.g., 4 for 'Vehicle 4').
    This retrieves high-frequency kinematic data and calculates the maximum speed and acceleration.
    """
    print(f"🔧 Tool Call: Querying DuckDB for Vehicle {track_id} physics...")
    df = _get_duckdb().get_trajectory_window(start_time, end_time, track_id)
    
    if df.empty:
        return f"No physics data found for Vehicle {track_id} in that time window."
    
    # Calculate simple deterministic metrics from the Savitzky-Golay smoothed data
    max_speed_x = df['vel_x'].abs().max()
    max_speed_y = df['vel_y'].abs().max()
    approx_max_speed = (max_speed_x**2 + max_speed_y**2)**0.5  # Pythagorean theorem for magnitude
    
    min_accel = df['accel_y'].min() # Negative acceleration in Y usually indicates braking
    
    analysis = {
        "vehicle_id": track_id,
        "max_speed_meters_per_second": round(approx_max_speed, 2),
        "hard_braking_detected": bool(min_accel < -4.0), # Assuming < -4 m/s^2 is hard braking
        "minimum_acceleration": round(min_accel, 2)
    }
    
    return json.dumps(analysis, indent=2)


@tool
def query_zone_flow(
    gate_name: str = "",
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
) -> str:
    """
    Zone Flow & OD Analysis tool.
    Use this tool when the user asks about vehicle counts, flow, entry/exit,
    origin-destination pairs, or dwell times for the defined zone.

    Args:
        gate_name:  Filter results to a specific gate name (e.g. 'North', 'South').
                    Leave empty to get data for all gates.
        start_time: Start of the time window in seconds (default: 0.0).
        end_time:   End of the time window in seconds (default: end of video).

    Returns a JSON object with:
      - gate_counts: per-gate enter/exit counts
      - od_pairs: list of (track_id, origin_gate, destination_gate, dwell_time_seconds)
    """
    print(f"🔧 Tool Call: Querying zone flow (gate='{gate_name}', t={start_time}-{end_time})...")
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


@tool
def evaluate_traffic_rules(
    track_id: int,
    start_time: float = 0.0,
    end_time: float = 9_999_999.0,
) -> str:
    """
    Symbolic Rule Engine — deterministic traffic violation detection.

    Use this tool to determine whether a vehicle VIOLATED a traffic safety
    rule (speeding, hard braking, aggressive acceleration).

    This is the symbolic reasoning layer: rules are explicit, auditable, and
    independent of any neural model.  Every result includes the exact numeric
    evidence that caused the rule to fire, so findings can be cited precisely.

    Rules evaluated:
      - SPEEDING               : peak speed > 50 km/h (13.89 m/s)
      - HARD_BRAKING           : deceleration < −4.0 m/s²
      - AGGRESSIVE_ACCELERATION: acceleration magnitude > 3.5 m/s²

    Args:
        track_id:   Integer track ID of the vehicle (e.g. 4 for 'Vehicle 4').
        start_time: Start of the time window in seconds (default: 0.0).
        end_time:   End of the time window in seconds (default: end of video).
    """
    print(f"🔧 Tool Call: Symbolic Rule Engine for Vehicle {track_id}...")
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