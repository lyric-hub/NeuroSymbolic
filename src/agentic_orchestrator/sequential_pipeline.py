"""
Multi-Agent Neuro-Symbolic Pipeline
=====================================

Architecture
------------

    route_query ──► planner ──► initialize ──► agent ◄──► tools
                                                  │
                                               finalize ──► END

Agents and their roles
----------------------
1. route_query (Symbolic)   Classifies query intent with cosine similarity.
                            Determines which tool set and system prompt to use.

2. planner     (Symbolic)   Decomposes the query into an ordered investigation
                            plan BEFORE the ReAct loop starts.  This is the
                            meta-reasoning / planning layer.  Only activated
                            for full_analysis queries.

3. initialize  (Symbolic)   Seeds the message history with the system prompt
                            (including the plan) and the user's query.

4. agent       (Neural)     Core ReAct loop — the LLM selects and calls tools,
                            observes results, and repeats until it can answer.

5. tools       (Neuro-Symbolic)  Executes the tool the LLM called:
                 search_semantic_events     → Milvus ANN (event-level)        (neural)
                 search_entity_profiles     → Milvus ANN (vehicle-level)    (neural)
                 query_graph_relationships  → Kùzu Cypher                   (symbolic)
                 verify_physics_math        → DuckDB kinematic stats         (symbolic)
                 evaluate_traffic_rules     → Rule engine                    (symbolic)
                 query_zone_flow            → DuckDB OD analysis             (symbolic)
                 get_vehicle_trajectory     → DuckDB spatial path            (symbolic)
                 get_vehicle_proximity      → DuckDB min-distance            (symbolic)
                 compare_vehicle_kinematics → DuckDB multi-vehicle stats     (symbolic)

6. finalize    (Symbolic)   Extracts the last AIMessage as the final answer
                            and exposes it as state['final_summary'].

Neuro-Symbolic separation
--------------------------
Neural  : YOLO + ByteTrack + VLM + sentence embeddings + LLM reasoning
Symbolic: Savitzky-Golay + homography + Kùzu graph + DuckDB + ZoneManager +
          TrafficRuleEngine (explicit, deterministic, auditable rules)
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv(Path(__file__).parent.parent.parent / ".env")
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage  # HumanMessage used in initialize

from .langgraph_state import AgentState
from .tools import (
    search_semantic_events,
    search_entity_profiles,
    query_graph_relationships,
    verify_physics_math,
    query_zone_flow,
    evaluate_traffic_rules,
    search_vehicles_by_type,
    get_vehicle_trajectory,
    get_vehicle_proximity,
    compare_vehicle_kinematics,
    get_vehicle_data_at_interval,
    compute_ttc,
    get_speed_statistics,
    get_vehicle_count_report,
    detect_traffic_queue,
    get_turning_movement_counts,
    get_data_quality_report,
    explain_threshold,
    link_violation_to_conflict,
    get_reasoning_trace,
)

# ---------------------------------------------------------------------------
# Tool sets — the router selects which LLM binding to use.
# ToolNode must register all tools so it can execute whichever the LLM calls.
# ---------------------------------------------------------------------------
TOOLS_FULL = [
    search_semantic_events,
    search_entity_profiles,
    query_graph_relationships,
    verify_physics_math,
    evaluate_traffic_rules,
    query_zone_flow,
    search_vehicles_by_type,       # Class-based vehicle lookup (motorcycle, car, bus…)
    get_vehicle_trajectory,        # Full spatial path reconstruction per vehicle
    get_vehicle_proximity,         # Min distance between two vehicles in a window
    compare_vehicle_kinematics,    # Side-by-side stats for multiple vehicles
    get_vehicle_data_at_interval,  # Fixed-rate snapshots (e.g. every 0.5 s)
    compute_ttc,                   # Time-to-Collision (TTC) conflict metric
    get_speed_statistics,          # 85th-percentile speed (V85) + mean + max
    get_vehicle_count_report,      # Volume / flow rate per time period
    detect_traffic_queue,          # Queue / congestion episode detection
    get_turning_movement_counts,   # Gate-level TMC matrix
    # --- Explainability tools ---
    get_data_quality_report,       # Real vs interpolated frames + detection confidence
    explain_threshold,             # Source standard behind each rule threshold
    link_violation_to_conflict,    # Write CAUSES edge to Kùzu causal graph
    get_reasoning_trace,           # Retrieve prior session's tool call audit log
]
TOOLS_SEMANTIC = [search_semantic_events, search_entity_profiles]

# ---------------------------------------------------------------------------
# System prompt building
# ---------------------------------------------------------------------------
# Per-plan system prompts expose only the 3–6 tool descriptions relevant to
# the active plan rather than all 19.  This reduces token usage by ~70% per
# request, which is critical for staying within API token-per-day limits.
# ---------------------------------------------------------------------------

_TOOL_CALL_MANDATE = (
    "MANDATORY RULE: You MUST call at least one tool before producing any answer. "
    "Never write a final answer without first calling a tool. "
    "If the tool returns no data, state that and stop — do not invent facts. "
    "Follow the Analysis Plan step by step, calling each tool in order.\n\n"
)

_TOOL_DESCRIPTIONS: dict = {
    "search_semantic_events": (
        "search_semantic_events — Semantic search over VLM-generated EVENT descriptions (Milvus). "
        "Finds specific events in specific time windows. "
        "Returns time_window_pointer (e.g. '10.0-15.0') for further queries. "
        "Use for: 'what happened at t=10s?', 'find a near-miss event'."
    ),
    "search_entity_profiles": (
        "search_entity_profiles — Behavioral profile search over per-VEHICLE longitudinal summaries. "
        "Searches summaries accumulated over the full video. "
        "Use for: 'which vehicle was most aggressive?', 'find the speeder', "
        "'which vehicle hard-braked the most throughout the video?'. "
        "Returns: track_id, summary, first_seen, last_seen."
    ),
    "query_graph_relationships": (
        "query_graph_relationships — Structural graph query (Kùzu, Cypher). "
        "Use the time_window from search_semantic_events to find which entities interacted. "
        "Node labels: Vehicle, Pedestrian, Infrastructure. "
        "Example: MATCH (s)-[r:INTERACTS_WITH]->(o) "
        "WHERE r.trajectory_time_window = '10.0-15.0' RETURN s.name, r.predicate, o.name"
    ),
    "verify_physics_math": (
        "verify_physics_math — Raw kinematic statistics (DuckDB). "
        "Returns max speed and minimum signed acceleration for a window. "
        "Use for quick numeric lookups or cross-checking rule engine results."
    ),
    "evaluate_traffic_rules": (
        "evaluate_traffic_rules — Symbolic Rule Engine (deterministic, auditable). "
        "Checks for violations: speeding, hard braking, aggressive acceleration. "
        "Each result includes exact evidence values. "
        "Always cite the 'evidence' field in your final answer."
    ),
    "query_zone_flow": (
        "query_zone_flow — Zone entry/exit counts and OD (Origin-Destination) pairs. "
        "Use for flow, counts, gate entry/exit, dwell time, OD matrix."
    ),
    "search_vehicles_by_type": (
        "search_vehicles_by_type — Find all track_ids of a given vehicle class. "
        "Call FIRST when the query mentions a vehicle type: "
        "'motorcycle', 'car', 'bus', 'truck', 'bicycle', 'person'. "
        "For 'two-wheelers': call twice — 'motorcycle' then 'bicycle'. "
        "Returns track_ids to pass to verify_physics_math and evaluate_traffic_rules."
    ),
    "get_vehicle_trajectory": (
        "get_vehicle_trajectory — Full spatial path reconstruction for one vehicle. "
        "Returns sampled (timestamp, pos_x, pos_y, speed, nearest_landmark). "
        "Use when you need WHERE a vehicle was at each moment, not just peak speed."
    ),
    "get_vehicle_proximity": (
        "get_vehicle_proximity — Tail-to-head gap between two vehicles (NOT centre-to-centre). "
        "gap = centre_distance - half_length_A - half_length_B. "
        "gap ≤ 0 → collision_confirmed=true. "
        "Returns min_gap_m, collision_confirmed, timestamp_s, both vehicle positions."
    ),
    "compare_vehicle_kinematics": (
        "compare_vehicle_kinematics — Side-by-side kinematic stats for multiple vehicles. "
        "Pass track_ids_csv e.g. '4,9' to get stats for both simultaneously. "
        "Use instead of calling verify_physics_math twice for incident comparisons."
    ),
    "get_vehicle_data_at_interval": (
        "get_vehicle_data_at_interval — Fixed-rate snapshots of all vehicles at regular intervals. "
        "Use after detect_traffic_queue to find vehicle positions during a congestion episode."
    ),
    "compute_ttc": (
        "compute_ttc — Time-to-Collision (TTC) between two vehicles. "
        "TTC = gap / closing_speed. Only defined when vehicles are approaching. "
        "Conflict levels: CRITICAL < 1.5 s, SERIOUS < 3.0 s, LOW ≥ 3.0 s. "
        "Use AFTER get_vehicle_proximity confirms vehicles were close. "
        "Returns min_ttc_s, conflict_level, gap_at_min_ttc_m, closing_speed_ms."
    ),
    "get_speed_statistics": (
        "get_speed_statistics — 85th-percentile speed (V85), mean, and max speed. "
        "V85 is the standard road safety / speed limit review metric. "
        "Use for: 'what is the 85th percentile speed?', 'are vehicles speeding?' "
        "Pass track_id=-1 for all vehicles, ≥0 for a specific vehicle."
    ),
    "get_vehicle_count_report": (
        "get_vehicle_count_report — Vehicle count and flow rate per time period. "
        "Use for: 'how many vehicles per hour?', 'when was peak traffic?' "
        "Use period_secs=900 for 15-min buckets (PHF analysis). "
        "Returns vehicles_per_hour for each bucket."
    ),
    "detect_traffic_queue": (
        "detect_traffic_queue — Queue / congestion episode detection. "
        "Finds time windows where ≥ N vehicles are simultaneously near-stationary. "
        "Use for: 'was there a queue?', 'how long did congestion last?'"
    ),
    "get_turning_movement_counts": (
        "get_turning_movement_counts — Gate-level TMC matrix. "
        "Pairs entry gate with exit gate per vehicle to produce turn counts. "
        "Use for: 'how many turned left from North?', 'what is the dominant movement?' "
        "Requires zone_config.json with named gates."
    ),
    "get_data_quality_report": (
        "get_data_quality_report — ALWAYS call before citing rule violations. "
        "Returns real vs interpolated frame counts, YOLO confidence, "
        "gate crossing confidence, and a quality rating (HIGH/MEDIUM/LOW). "
        "If rating is LOW, state findings as indicative not confirmed."
    ),
    "explain_threshold": (
        "explain_threshold — Call when the user asks WHY a threshold was chosen. "
        "Returns the engineering standard / source citation for any rule. "
        "E.g. explain_threshold('HARD_BRAKING') → cites the source standard."
    ),
    "link_violation_to_conflict": (
        "link_violation_to_conflict — Call AFTER confirming both a violation and a conflict exist "
        "and you are confident one caused the other. "
        "Writes a permanent CAUSES edge to the Kùzu graph."
    ),
    "get_reasoning_trace": (
        "get_reasoning_trace — Call when user asks 'how did you reach that conclusion?' "
        "Retrieves the full tool-call audit log for a prior session."
    ),
}

_PLAN_DECISION_RULES: dict = {
    "behavioral": (
        "Call search_entity_profiles first to find the vehicle matching the description. "
        "Then call verify_physics_math to confirm kinematic evidence. "
        "Then call evaluate_traffic_rules for formal violation verdicts. "
        "Always call get_data_quality_report before citing rule violations."
    ),
    "vehicle_type": (
        "Call search_vehicles_by_type first with the requested class (e.g. 'motorcycle'). "
        "For two-wheelers call it twice: 'motorcycle' then 'bicycle'. "
        "Then call verify_physics_math and evaluate_traffic_rules per track_id. "
        "Always call get_data_quality_report before citing rule violations."
    ),
    "conflict": (
        "Call search_semantic_events first to find the conflict event and time window. "
        "Then call query_graph_relationships to identify the involved vehicles. "
        "Then call get_vehicle_proximity to confirm minimum gap. "
        "Then call compute_ttc for conflict severity. "
        "Always call get_data_quality_report before citing findings."
    ),
    "incident": (
        "Follow the Analysis Plan step by step. "
        "Start with search_semantic_events to find the incident time window. "
        "Then query_graph_relationships to find involved vehicles. "
        "Then compare_vehicle_kinematics for pre-accident and incident windows. "
        "Then get_vehicle_proximity to confirm closest approach. "
        "Then evaluate_traffic_rules for each vehicle. "
        "Always call get_data_quality_report before citing rule violations."
    ),
    "vehicle_specific": (
        "Call search_semantic_events first to find events involving this vehicle. "
        "Then verify_physics_math for kinematic statistics. "
        "Then evaluate_traffic_rules to check for violations. "
        "Then query_graph_relationships for interactions with other vehicles. "
        "Always call get_data_quality_report before citing rule violations."
    ),
    "turning_movements": (
        "Call get_turning_movement_counts to get the full TMC matrix. "
        "Then call query_zone_flow for supplementary dwell times and gate counts."
    ),
    "speed_compliance": (
        "Call get_speed_statistics with track_id=-1 to get V85, mean, max for all vehicles. "
        "Then call evaluate_traffic_rules for vehicles whose speed exceeds the limit. "
        "Always call get_data_quality_report before citing rule violations."
    ),
    "flow": (
        "Call query_zone_flow to get gate counts and OD pairs."
    ),
    "volume": (
        "Call get_vehicle_count_report with period_secs=900 for 15-min PHF buckets. "
        "Then call get_vehicle_count_report with period_secs=3600 for hourly volumes."
    ),
    "queue": (
        "Call detect_traffic_queue to find congestion episodes. "
        "Then call get_vehicle_data_at_interval at each episode start/end for vehicle positions. "
        "Then call get_vehicle_count_report to correlate queue episodes with traffic volume."
    ),
    "relational": (
        "Call search_semantic_events to find the relevant time window. "
        "Then call query_graph_relationships to find which vehicles interacted. "
        "Then call verify_physics_math for the involved vehicles."
    ),
    "default": (
        "Call search_semantic_events to find relevant events. "
        "Then search_entity_profiles to find relevant vehicle profiles. "
        "Then verify_physics_math for any identified vehicles. "
        "Then evaluate_traffic_rules if violations are suspected."
    ),
}

_OUTPUT_FORMAT = """\
REQUIRED OUTPUT FORMAT — write your final answer in plain, readable prose.

Rules:
- Do NOT paste raw JSON or tool output. Extract only the key numbers and facts.
- Do NOT use the words "FINDING:", "EVIDENCE:", "CONFIDENCE:" as headers.
  Instead write naturally: start with a direct answer, then support it.
- Use markdown: **bold** for vehicle IDs and key values, bullet lists for \
multiple items, ### for section headings if the answer is long.
- Keep it concise. One short paragraph per vehicle or topic is enough.
- Always state the speed limit when reporting speed violations.
- If no violations were found, say so in one sentence.
- End with a one-line confidence statement in italics: \
*Confidence: HIGH — based on [source].*

Example of good style:
**Vehicle 4** was speeding at **55.8 km/h** (limit: 50 km/h) for 65% of its \
observed time, and hard-braked at **−3.5 m/s²** (threshold: −3.0 m/s²) near \
t=29s. No other violations were detected.
*Confidence: HIGH — kinematic data, 1089 trusted frames.*"""


def _build_plan_system_prompt(plan_key: str) -> str:
    """
    Builds a minimal system prompt containing only the tool descriptions for
    the tools bound to the given plan.  Falls back to describing all tools
    when plan_key is unknown.

    Reduces token usage by ~70% compared to always sending all 19 descriptions.
    """
    tools_in_plan = _PLAN_TOOLS.get(plan_key)
    if tools_in_plan is None:
        tool_names = list(_TOOL_DESCRIPTIONS.keys())
    else:
        tool_names = [t.name for t in tools_in_plan]

    tool_block = "\n\n".join(
        f"{i + 1}. {_TOOL_DESCRIPTIONS[name]}"
        for i, name in enumerate(tool_names)
        if name in _TOOL_DESCRIPTIONS
    )
    decision_rule = _PLAN_DECISION_RULES.get(
        plan_key,
        "Follow the Analysis Plan step by step. "
        "Always cite the tool output that supports each claim. "
        "Base your answer strictly on what the tools returned.",
    )
    return (
        _TOOL_CALL_MANDATE
        + "You are an expert traffic safety analyst with access to a "
        "Neuro-Symbolic analysis system.\n\n"
        + f"Available tools:\n{tool_block}\n\n"
        + f"Decision rule: {decision_rule}\n\n"
        + _OUTPUT_FORMAT
    )


_SYSTEM_PROMPT_SEMANTIC = _TOOL_CALL_MANDATE + \
"""You are an expert traffic safety analyst.

You have two tools available:
1. search_semantic_events  — Searches VLM-generated event descriptions (frame-level).
                             Use to find specific events and summarise what happened.
2. search_entity_profiles  — Searches per-vehicle longitudinal behavioral summaries.
                             Use for "which vehicle was most aggressive?" type questions.

Choose the appropriate tool based on whether the query is about a specific event
or about a vehicle's overall behaviour across the video.
Base your final answer strictly on what the tools returned. Do not invent facts."""

# ---------------------------------------------------------------------------
# Planner — deterministic template-based (Symbolic).
# Maps query keywords to a fixed investigation sequence.
# No LLM involved — output is always the same for the same query type.
# ---------------------------------------------------------------------------

_PLAN_TEMPLATES = {
    "vehicle_type": {
        "keywords": [
            "motorcycle", "two-wheel", "bicycle", "bike",
            "bus", "buses", "truck", "trucks", "pedestrian", "person",
            "vehicle type", "vehicles by type",
        ],
        "plan": (
            "1. Call search_vehicles_by_type to find all track_ids of the requested class.\n"
            "   For two-wheelers call it twice: 'motorcycle' then 'bicycle'.\n"
            "2. Call verify_physics_math for each track_id.\n"
            "3. Call evaluate_traffic_rules for each track_id.\n"
            "4. Summarise behaviour patterns across all vehicles of this type."
        ),
    },
    # conflict is checked before incident so "time-to-collision" (which contains
    # "collision") routes to the conflict plan rather than the incident plan.
    "conflict": {
        "keywords": [
            "ttc", "time to collision", "time-to-collision", "close call",
            "how close", "dangerous gap", "separation", "closest",
            "conflict severity", "minimum gap",
            "tailgat", "following distance", "headway",
        ],
        "plan": (
            "1. Call search_semantic_events to find the conflict event and time window.\n"
            "2. Call query_graph_relationships to identify the two involved vehicles.\n"
            "3. Call get_vehicle_proximity to confirm minimum gap and whether collision occurred.\n"
            "4. Call compute_ttc for the same vehicles and window to get conflict severity level.\n"
            "5. Synthesise: gap + TTC together give the full conflict picture."
        ),
    },
    "incident": {
        # "t=" and "why" removed — too broad, captured trajectory and threshold queries.
        # "cause" removed — too broad, now covered by more specific phrases.
        "keywords": [
            "incident", "collision", "crash", "near-miss", "near miss",
            "accident", "happened at", "what happened", "reason",
            "led to", "before the",
        ],
        "plan": (
            "1. Call search_semantic_events to find the incident and its time_window_pointer (e.g. '10.0-15.0').\n"
            "2. Call query_graph_relationships to find which vehicles interacted at that window:\n"
            "   MATCH (s)-[r:INTERACTS_WITH]->(o) WHERE r.trajectory_time_window = 'WINDOW_VALUE' RETURN s.name, r.predicate, o.name, r.motion_state, r.phase\n"
            "3. For each involved vehicle, traverse the PRECEDES chain to find pre-accident behaviour (up to 3 windows back):\n"
            "   MATCH (v:Vehicle {name:'VEHICLE_NAME'})-[r:PRECEDES*1..3]->(v2) WHERE r[-1].to_window = 'WINDOW_VALUE' RETURN r[*].from_window, r[*].to_window\n"
            "4. Call compare_vehicle_kinematics with all involved vehicle IDs for the PRE-ACCIDENT window (start_time = incident_start - 15, end_time = incident_start) to compare speeds side-by-side.\n"
            "5. Call get_vehicle_proximity with the two closest vehicles for the INCIDENT window to find the minimum distance and time of closest approach.\n"
            "6. Call compare_vehicle_kinematics again for the INCIDENT window itself to confirm impact kinematics for all parties simultaneously.\n"
            "7. Call evaluate_traffic_rules for each involved vehicle over the full range (pre-accident + incident) to detect violations that preceded the crash.\n"
            "8. Synthesise: combine pre-accident behaviour, proximity data, rule violations, and impact kinematics to state the cause."
        ),
    },
    "behavioral": {
        "keywords": [
            "aggressive", "dangerous", "worst", "most",
            "behaviour", "behavior", "profile", "erratic",
        ],
        "plan": (
            "1. Call search_entity_profiles to find vehicles matching the description.\n"
            "2. Call verify_physics_math to confirm kinematic evidence.\n"
            "3. Call evaluate_traffic_rules for formal violation verdicts."
        ),
    },
    # vehicle_specific is checked before relational so "Vehicle 4 interact with"
    # routes to vehicle_specific rather than relational.
    "vehicle_specific": {
        "keywords": ["vehicle ", "track id", "track_id"],
        "plan": (
            "1. Call search_semantic_events to find events involving this vehicle.\n"
            "2. Call verify_physics_math to get kinematic statistics.\n"
            "3. Call evaluate_traffic_rules to check for violations.\n"
            "4. Call query_graph_relationships to find interactions with other vehicles."
        ),
    },
    # turning_movements is checked before flow to prevent "count" in flow from
    # capturing turning movement count queries (TMC matrix).
    "turning_movements": {
        "keywords": [
            "turning", "turn", "left turn", "right turn", "u-turn",
            "tmc", "approach", "departure", "intersection movement",
            "dominant movement", "turning count", "movement count",
        ],
        "plan": (
            "1. Call get_turning_movement_counts to get the full TMC matrix.\n"
            "2. Identify dominant movements and any unexpected movements (e.g. U-turns).\n"
            "3. Call query_zone_flow for dwell times and gate counts to supplement TMC."
        ),
    },
    "speed_compliance": {
        "keywords": [
            "85th", "v85", "percentile", "speed limit", "compliance",
            "average speed", "mean speed", "typical speed",
            "speeding", "over the limit", "exceed", "over limit",
            "speed violation", "were speeding", "is speeding",
        ],
        "plan": (
            "1. Call get_speed_statistics (track_id=-1) to get V85, mean, max for all vehicles.\n"
            "2. Call evaluate_traffic_rules for vehicles whose max speed exceeds the posted limit.\n"
            "3. Summarise compliance: percentage of vehicles within limit, V85 vs posted limit."
        ),
    },
    "flow": {
        # "count" and "how many vehicles" removed — too broad.
        # "count" captured turning movement count queries; "how many vehicles"
        # captured volume queries.  Zone/gate-specific keywords are sufficient.
        "keywords": [
            "flow", "zone", "gate", "entry", "exit",
            "od matrix", "origin", "destination", "dwell",
        ],
        "plan": (
            "1. Call query_zone_flow to get gate counts and OD pairs."
        ),
    },
    "volume": {
        "keywords": [
            "volume", "how many", "count per", "per hour", "per minute",
            "peak hour", "peak traffic", "traffic count", "traffic volume",
            "phf", "aadt",
        ],
        "plan": (
            "1. Call get_vehicle_count_report with period_secs=900 (15-min buckets) for PHF.\n"
            "2. Call get_vehicle_count_report with period_secs=3600 for hourly volumes.\n"
            "3. Identify peak period and compute PHF = total_peak_hour / (4 × peak_15min)."
        ),
    },
    "queue": {
        "keywords": [
            "queue", "congestion", "traffic jam", "backup", "shockwave",
            "stationary", "slow traffic", "gridlock",
        ],
        "plan": (
            "1. Call detect_traffic_queue to find congestion episodes.\n"
            "2. For each episode, call get_vehicle_data_at_interval at the episode start/end\n"
            "   to get vehicle positions and identify the queue's spatial extent.\n"
            "3. Call get_vehicle_count_report to correlate queue episodes with traffic volume."
        ),
    },
    # relational is last — keywords like "interact" and "following" are common
    # words that appear in many query types; checking more-specific templates first
    # prevents false matches.
    "relational": {
        # "conflict" removed — belongs to the conflict template.
        # "between" removed — too broad, appears in trajectory and conflict queries.
        "keywords": [
            "interact", "relationship", "following", "tailgat", "together",
        ],
        "plan": (
            "1. Call search_semantic_events to find the relevant time window.\n"
            "2. Call query_graph_relationships to find which vehicles interacted.\n"
            "3. Call verify_physics_math for the involved vehicles."
        ),
    },
}

_DEFAULT_PLAN = (
    "1. Call search_semantic_events to find relevant events.\n"
    "2. Call search_entity_profiles to find relevant vehicle profiles.\n"
    "3. Call verify_physics_math for any identified vehicles.\n"
    "4. Call evaluate_traffic_rules if violations are suspected."
)


def _select_plan(query: str) -> tuple:
    """
    Deterministic keyword matcher.
    Returns (plan: str, plan_key: str).
    Falls back to (_DEFAULT_PLAN, 'default') if no keywords match.
    """
    q = query.lower()
    for key, template in _PLAN_TEMPLATES.items():
        if any(kw in q for kw in template["keywords"]):
            return template["plan"], key
    return _DEFAULT_PLAN, "default"


# ---------------------------------------------------------------------------
# Plan-scoped tool subsets.
# Binding only 3–6 tools relevant to the selected plan dramatically reduces
# the LLM's decision space compared to exposing all 20 tools at once.
# The ToolNode still registers all tools so it can execute any call the LLM
# makes regardless of which subset was bound.
# ---------------------------------------------------------------------------
_PLAN_TOOLS: dict = {
    "vehicle_type": [
        search_vehicles_by_type, verify_physics_math,
        evaluate_traffic_rules, get_data_quality_report, get_speed_statistics,
    ],
    "conflict": [
        search_semantic_events, query_graph_relationships,
        get_vehicle_proximity, compute_ttc,
        link_violation_to_conflict, get_data_quality_report,
    ],
    "incident": [
        search_semantic_events, query_graph_relationships,
        compare_vehicle_kinematics, get_vehicle_proximity,
        evaluate_traffic_rules, link_violation_to_conflict, get_data_quality_report,
    ],
    "behavioral": [
        search_entity_profiles, verify_physics_math,
        evaluate_traffic_rules, get_data_quality_report,
    ],
    "vehicle_specific": [
        search_semantic_events, verify_physics_math,
        evaluate_traffic_rules, query_graph_relationships,
        get_vehicle_trajectory, get_data_quality_report,
    ],
    "turning_movements": [
        get_turning_movement_counts, query_zone_flow,
    ],
    "speed_compliance": [
        get_speed_statistics, evaluate_traffic_rules,
        explain_threshold, get_data_quality_report,
    ],
    "flow": [
        query_zone_flow, get_turning_movement_counts,
    ],
    "volume": [
        get_vehicle_count_report, query_zone_flow,
    ],
    "queue": [
        detect_traffic_queue, get_vehicle_data_at_interval,
        get_vehicle_count_report,
    ],
    "relational": [
        search_semantic_events, query_graph_relationships,
        verify_physics_math, get_vehicle_proximity, get_data_quality_report,
    ],
    "default": [
        search_semantic_events, search_entity_profiles,
        verify_physics_math, evaluate_traffic_rules,
    ],
}

# ---------------------------------------------------------------------------
# LLM — provider selected by AGENT_LLM_PROVIDER env var.
#
#   AGENT_LLM_PROVIDER=ollama  (local Ollama, no API key required)
#     AGENT_MODEL=gemma4:e2b   (or any model pulled in Ollama)
#
#   AGENT_LLM_PROVIDER=openai  (default — any OpenAI-compatible API)
#     AGENT_API_KEY=<key>
#     AGENT_API_BASE_URL=https://api.groq.com/openai/v1
#     AGENT_MODEL=llama-3.3-70b-versatile
# ---------------------------------------------------------------------------
_LLM_PROVIDER = os.environ.get("AGENT_LLM_PROVIDER", "openai")
_AGENT_MODEL  = os.environ.get("AGENT_MODEL", "llama-3.3-70b-versatile")

if _LLM_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=_AGENT_MODEL, temperature=0.0)
else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=_AGENT_MODEL,
        api_key=os.environ.get("AGENT_API_KEY", ""),
        base_url=os.environ.get("AGENT_API_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0.0,
        max_retries=3,
        request_timeout=60,
    )
llm_full = llm.bind_tools(TOOLS_FULL)
llm_semantic = llm.bind_tools(TOOLS_SEMANTIC)

_plan_llm_cache: dict = {}


def _get_plan_llm(plan_key: str):
    """Returns a cached tool-bound LLM for the given plan key."""
    if plan_key not in _plan_llm_cache:
        tools = _PLAN_TOOLS.get(plan_key)
        _plan_llm_cache[plan_key] = (
            llm.bind_tools(tools) if tools else llm_full
        )
    return _plan_llm_cache[plan_key]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> AgentState:
    """
    Router node: classifies the query intent using embedding cosine-similarity.

    If `route` is already set in the incoming state (pre-seeded by the caller),
    the embedding model is not loaded — this avoids a SIGSEGV caused by loading
    sentence-transformers after ChatOpenAI's C-extensions are initialised in the
    same process.  Used by the eval harness which pre-computes routes via L1.
    """
    if state.get("route"):
        # Route pre-seeded — skip cosine similarity, just stamp session metadata.
        return {
            "session_id": state.get("session_id") or str(uuid.uuid4()),
            "routing_explanation": f"Pre-seeded route: {state['route']}",
            "reasoning_steps": [],
            "contradictions": [],
        }

    from .hierarchical_router import _embed, _max_cos_sim, _get_proto_embeddings

    query  = state["query"]
    protos = _get_proto_embeddings()
    q_vec  = _embed([query])[0]
    full_score = _max_cos_sim(q_vec, protos["full_analysis"])
    sem_score  = _max_cos_sim(q_vec, protos["semantic_lookup"])
    route      = "full_analysis" if full_score >= sem_score else "semantic_lookup"
    routing_explanation = (
        f"Routed to {route} "
        f"(full_analysis score={full_score:.3f}, semantic_lookup score={sem_score:.3f}). "
        + ("Full physics + rule engine activated." if route == "full_analysis"
           else "VLM semantic search only — no kinematic tools available.")
    )

    return {
        "route": route,
        "session_id": str(uuid.uuid4()),
        "routing_explanation": routing_explanation,
        "reasoning_steps": [],
        "contradictions": [],
        "plan_key": "",
    }


def planner_node(state: AgentState) -> AgentState:
    """
    Planner node: selects a deterministic investigation plan from
    _PLAN_TEMPLATES using keyword matching on the query.

    If `plan_key` is already set in the incoming state (pre-seeded), the
    keyword matching is skipped and the stored plan_key is used directly.

    Only activated for 'full_analysis' queries.
    """
    if state.get("route") != "full_analysis":
        return {"plan": "", "plan_key": ""}

    if state.get("plan_key"):
        # Plan pre-seeded by caller — reconstruct plan text from key.
        plan = _PLAN_TEMPLATES.get(state["plan_key"], {}).get("plan", _DEFAULT_PLAN)
        return {"plan": plan}

    plan, plan_key = _select_plan(state["query"])
    print(f"\nAnalysis Plan (symbolic) [{plan_key}]:\n{plan}\n")
    return {"plan": plan, "plan_key": plan_key}


def initialize(state: AgentState) -> AgentState:
    """
    Entry node: seeds the message history with the route-appropriate system
    prompt and the user's query as a HumanMessage.

    If a plan was produced by the planner, it is appended to the system prompt
    so the agent knows what steps to follow.
    """
    is_full = state.get("route") == "full_analysis"
    if is_full:
        plan_key = state.get("plan_key") or "default"
        system_prompt = _build_plan_system_prompt(plan_key)
    else:
        system_prompt = _SYSTEM_PROMPT_SEMANTIC

    plan = state.get("plan", "")
    if plan:
        system_prompt = (
            system_prompt
            + f"\n\n[Analysis Plan]\n{plan}\n\nFollow this plan step by step."
        )

    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["query"]),
        ]
    }


def agent_node(state: AgentState) -> AgentState:
    """
    Core reasoning node.

    For semantic_lookup queries: binds only search_semantic_events + search_entity_profiles.
    For full_analysis queries: binds only the 3–6 tools relevant to the plan_key
      selected by the symbolic planner.  Falling back to the full 20-tool binding
      only when no plan_key is available.

    Narrowing the tool set makes correct tool selection tractable for smaller
    LLMs — the model chooses from 3–6 options instead of 20.
    """
    route = state.get("route")
    if route != "full_analysis":
        llm_to_use = llm_semantic
    else:
        plan_key = state.get("plan_key", "")
        llm_to_use = _get_plan_llm(plan_key) if plan_key else llm_full
    response = llm_to_use.invoke(state["messages"])
    return {"messages": [response]}


def finalize(state: AgentState) -> AgentState:
    """
    Extracts the last AIMessage, reconstructs the reasoning trace from the
    message history, persists it to DuckDB, and appends metadata to the summary.
    """
    last = state["messages"][-1]
    session_id = state.get("session_id", "unknown")

    # --- Extract reasoning trace from message history ---
    # Match AIMessage tool_calls to their corresponding ToolMessages by tool_call_id.
    steps: list = []
    step_num = 0
    tool_call_id_map: dict = {}   # tool_call_id → step dict

    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                step_num += 1
                step = {
                    "step": step_num,
                    "tool": tc["name"],
                    "args": tc.get("args", {}),
                    "output_excerpt": "",
                }
                tc_id = tc.get("id", "")
                if tc_id:
                    tool_call_id_map[tc_id] = step
                steps.append(step)
        elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
            matched = tool_call_id_map.get(msg.tool_call_id)
            if matched is not None:
                matched["output_excerpt"] = str(msg.content)[:300]

    # Persist trace to DuckDB
    try:
        from .tools import _get_duckdb
        _get_duckdb().insert_reasoning_trace(session_id, steps)
    except Exception:
        pass  # non-fatal — trace persistence is best-effort

    summary = last.content

    return {
        "final_summary": summary,
        "reasoning_steps": steps,
        "route": state.get("route", "unknown"),
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# Contradiction check node — compares VLM (neural) output with rule engine
# (symbolic) output and flags disagreements in state.
# ---------------------------------------------------------------------------

_SMOOTH_KEYWORDS = frozenset({
    "normal", "smooth", "clear", "flowing", "steady",
    "calm", "typical", "regular", "undisturbed",
})

# If any of these appear in a semantic result chunk, that chunk describes an
# incident — smooth-keyword hits inside it are not "normal traffic" signals.
_INCIDENT_KEYWORDS = frozenset({
    "collision", "accident", "crash", "rear-end", "rear_end",
    "emergency", "tailgating", "speeding", "violation", "hard_braking",
    "wrong_way", "near-miss", "near_miss", "wreck", "impact",
})


def _strip_incident_sentences(text: str) -> str:
    """
    Remove sentences that contain incident keywords so smooth-keyword matching
    only fires on genuinely calm descriptions, not on incident summaries that
    happen to use words like 'normal speed' or 'steady approach'.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = [
        s for s in sentences
        if not any(kw in s.lower() for kw in _INCIDENT_KEYWORDS)
    ]
    return " ".join(clean)


def contradiction_check(state: AgentState) -> AgentState:
    """
    Post-finalize node: scans the message history for neural-symbolic contradictions.

    A contradiction is flagged when:
      - A semantic tool (search_semantic_events / search_entity_profiles) returned
        descriptions containing 'normal', 'smooth', 'clear' etc. in non-incident
        sentences, AND
      - The rule engine (evaluate_traffic_rules) returned actual violations in
        the same session.

    Incident descriptions (containing 'collision', 'accident', 'speeding' etc.)
    are excluded from the smooth-keyword scan to prevent false positives where
    an entity profile describes a crash vehicle as having "caused normal traffic
    disruption" or similar phrasing.

    If contradictions are found they are appended to final_summary so the
    user is explicitly warned that the neural and symbolic layers disagreed.
    """
    # Map tool_call_id → tool_name from AIMessages
    tool_call_id_to_name: dict = {}
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_id_to_name[tc.get("id", "")] = tc["name"]

    # Collect tool results by tool name
    tool_results: dict = {}
    for msg in state["messages"]:
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            name = tool_call_id_to_name.get(msg.tool_call_id, "")
            if name:
                tool_results.setdefault(name, []).append(str(msg.content))

    # Only flag smooth keywords in non-incident context.
    raw_semantic = " ".join(
        tool_results.get("search_semantic_events", []) +
        tool_results.get("search_entity_profiles", [])
    )
    semantic_text = _strip_incident_sentences(raw_semantic).lower()

    rule_text = " ".join(tool_results.get("evaluate_traffic_rules", [])).lower()

    smooth_hits = [w for w in _SMOOTH_KEYWORDS if w in semantic_text]
    has_violation = (
        '"violations"' in rule_text
        and '"violation_count": 0' not in rule_text
        and "NO_VIOLATIONS" not in rule_text
    )

    contradictions: list = list(state.get("contradictions") or [])
    if smooth_hits and has_violation:
        contradictions.append(
            f"NEURAL-SYMBOLIC CONTRADICTION: VLM description contains "
            f"'{', '.join(smooth_hits)}' (suggesting normal traffic) but the "
            f"symbolic rule engine detected violations. "
            f"Trust the rule engine — it uses raw kinematics, not visual interpretation. "
            f"The VLM may have missed the event or described a different time window."
        )

    return {
        "contradictions": contradictions,
        "final_summary": state.get("final_summary", ""),
    }


# ---------------------------------------------------------------------------
# ToolNode must register all tools across both paths so it can execute
# whichever tool the active LLM binding emits.
# ---------------------------------------------------------------------------
tools_node = ToolNode(TOOLS_FULL)


# ---------------------------------------------------------------------------
# Graph compilation
#
#   route_query ──► planner ──► initialize ──► agent ──► tools ──► agent (loop)
#                                                  │
#                                                  └── finalize ──► END
#
# tools_condition inspects the last AIMessage:
#   - has tool_calls  → "tools"
#   - plain text      → "__end__" (mapped to "finalize")
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("route_query", route_query)
workflow.add_node("planner", planner_node)
workflow.add_node("initialize", initialize)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.add_node("finalize", finalize)
workflow.add_node("contradiction_check", contradiction_check)

workflow.set_entry_point("route_query")
workflow.add_edge("route_query", "planner")
workflow.add_edge("planner", "initialize")
workflow.add_edge("initialize", "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": "finalize"},
)

workflow.add_edge("tools", "agent")              # observation → next reasoning step
workflow.add_edge("finalize", "contradiction_check")
workflow.add_edge("contradiction_check", END)

# recursion_limit caps the agent ↔ tools loop.
# Each tool call = 2 LangGraph steps (agent → tool → agent).
# Hard/incident plans call up to 11 tools = 22+ steps; 50 gives headroom
# for retries and finalize/contradiction_check nodes.
#
# IMPORTANT: pass the config dict at invoke time, not via .config attribute.
# The .config attribute approach is not supported in all LangGraph versions.
# Callers must use: agent_app.invoke(state, config=AGENT_INVOKE_CONFIG)
AGENT_INVOKE_CONFIG: dict = {"recursion_limit": 50}
agent_app = workflow.compile()
