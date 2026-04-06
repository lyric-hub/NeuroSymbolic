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

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage  # HumanMessage used in initialize

from .langgraph_state import AgentState
from .hierarchical_router import _classify_intent
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
]
TOOLS_SEMANTIC = [search_semantic_events, search_entity_profiles]

# ---------------------------------------------------------------------------
# System prompts — tailored per intent class.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_FULL = """You are an expert traffic safety analyst with access to a \
Neuro-Symbolic analysis system.

Available tools:
1. search_semantic_events      — Semantic search over VLM-generated EVENT descriptions (Milvus).
                                 Finds specific events in specific time windows.
                                 Returns time_window_pointer (e.g. '10.0-15.0') for further queries.
                                 Use for: "what happened at t=10s?", "find a near-miss event".

2. search_entity_profiles      — Behavioral profile search over per-VEHICLE longitudinal summaries.
                                 Searches summaries accumulated over the full video.
                                 Use for: "which vehicle was most aggressive?", "find the speeder",
                                 "which vehicle hard-braked the most throughout the video?".
                                 Returns: track_id, summary, first_seen, last_seen.

3. query_graph_relationships   — Structural graph query (Kùzu, Cypher).
                                 Use the time_window from tool 1 to find which entities interacted.
                                 Node labels: Vehicle, Pedestrian, Infrastructure.
                                 Example: MATCH (s)-[r:INTERACTS_WITH]->(o)
                                          WHERE r.trajectory_time_window = '10.0-15.0'
                                          RETURN s.name, r.predicate, o.name

4. verify_physics_math         — Raw kinematic statistics (DuckDB).
                                 Returns max speed and minimum signed acceleration for a window.
                                 Use for quick numeric lookups or cross-checking rule engine results.

5. evaluate_traffic_rules      — Symbolic Rule Engine (deterministic, auditable).
                                 Use this to CHECK FOR VIOLATIONS: speeding, hard braking,
                                 aggressive acceleration. Each result includes exact evidence values.
                                 Always cite the 'evidence' field in your final answer.

6. query_zone_flow             — Zone entry/exit counts and OD (Origin-Destination) pairs.
                                 Use for flow, counts, gate entry/exit, dwell time, OD matrix.

7. search_vehicles_by_type     — Find all track_ids of a given vehicle class.
                                 Use FIRST when the query mentions a vehicle type:
                                 "motorcycle", "car", "bus", "truck", "bicycle", "person".
                                 For "two-wheelers": call twice — "motorcycle" + "bicycle".
                                 Returns track_ids to use with tools 4 and 5.

8. get_vehicle_trajectory      — Full spatial path reconstruction for one vehicle.
                                 Returns sampled (timestamp, pos_x, pos_y, speed, nearest_landmark).
                                 Use when you need to know WHERE a vehicle was at each moment,
                                 not just peak speed. E.g. "was it in the intersection when it braked?"

9. get_vehicle_proximity       — Tail-to-head gap between two vehicles (NOT centre-to-centre).
                                 gap = centre_distance - half_length_A - half_length_B.
                                 gap ≤ 0 → collision_confirmed=true.
                                 Use to answer "did they actually collide?" after identifying involved vehicles.
                                 Returns min_gap_m, collision_confirmed, timestamp_s, both vehicle positions.

10. compare_vehicle_kinematics — Side-by-side kinematic stats for multiple vehicles.
                                 Pass track_ids_csv e.g. "4,9" to get stats for both simultaneously.
                                 Use instead of calling verify_physics_math twice for incident comparisons.

11. compute_ttc               — Time-to-Collision (TTC) between two vehicles.
                                 TTC = gap / closing_speed. Only defined when vehicles are approaching.
                                 Conflict levels: CRITICAL < 1.5 s, SERIOUS < 3.0 s, LOW ≥ 3.0 s.
                                 Use AFTER get_vehicle_proximity confirms vehicles were close.
                                 Returns min_ttc_s, conflict_level, gap_at_min_ttc_m, closing_speed_ms.

12. get_speed_statistics      — 85th-percentile speed (V85), mean, and max speed.
                                 V85 is the standard road safety / speed limit review metric.
                                 Use for: "what is the 85th percentile speed?", "are vehicles speeding?"
                                 Pass track_id=-1 for all vehicles, ≥0 for a specific vehicle.

13. get_vehicle_count_report  — Vehicle count and flow rate per time period.
                                 Use for: "how many vehicles per hour?", "when was peak traffic?"
                                 Use period_secs=900 for 15-min buckets (PHF analysis).
                                 Returns vehicles_per_hour for each bucket.

14. detect_traffic_queue      — Queue / congestion episode detection.
                                 Finds time windows where ≥ N vehicles are simultaneously near-stationary.
                                 Use for: "was there a queue?", "how long did congestion last?"
                                 After finding episodes, use get_vehicle_data_at_interval for positions.

15. get_turning_movement_counts — Gate-level TMC matrix.
                                 Pairs entry gate with exit gate per vehicle to produce turn counts.
                                 Use for: "how many vehicles turned left from North?", "what is the
                                 dominant movement?" Requires zone_config.json with named gates.

Decision rules:
- Global behavioral questions ("most aggressive", "which vehicle sped the most"): tool 2.
- Safety/violation questions ("did vehicle 4 brake hard?"): tools 1 → 5.
- Relational questions ("which vehicles interacted?"): tools 1 → 3.
- Combined safety + relationships: tools 1 → 3 → 5.
- Full incident reconstruction: tools 1 → 3 → 5 → 4 (raw stats for extra context).
- Flow/count/OD questions: tool 6 directly.
- Vehicle-type questions ("behaviour of motorcycles"): tool 7 → tools 4 + 5 per track_id.
- TTC / conflict severity: tool 8 (proximity) → tool 11 (TTC) for full conflict picture.
- Speed compliance / V85: tool 12 for all vehicles, then tool 5 per flagged vehicle.
- Traffic volume / peak hour: tool 13 directly.
- Queue / congestion: tool 14 to find episodes, then tool 10 for vehicle positions inside.
- Turning movements / intersection flow: tool 15 directly.
- Always cite the tool output that supports each claim in your final answer.
- Base your final answer strictly on what the tools returned. Do not invent facts."""

_SYSTEM_PROMPT_SEMANTIC = """You are an expert traffic safety analyst.

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
            "buses", "trucks", "pedestrian", "person",
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
    "incident": {
        "keywords": [
            "incident", "collision", "crash", "near-miss", "near miss",
            "accident", "happened at", "t=", "what happened", "reason",
            "cause", "why", "led to", "before the",
        ],
        "plan": (
            "1. Call search_semantic_events to find the incident and its time_window_pointer (e.g. '10.0-15.0').\n"
            "2. Call query_graph_relationships to find which vehicles interacted at that window:\n"
            "   MATCH (s)-[r:INTERACTS_WITH]->(o) WHERE r.trajectory_time_window = '<window>' RETURN s.name, r.predicate, o.name, r.motion_state, r.phase\n"
            "3. For each involved vehicle, traverse the PRECEDES chain to find pre-accident behaviour (up to 3 windows back):\n"
            "   MATCH (v:Vehicle {name:'<name>'})-[r:PRECEDES*1..3]->(v2) WHERE r[-1].to_window = '<window>' RETURN r[*].from_window, r[*].to_window\n"
            "4. Call compare_vehicle_kinematics with all involved vehicle IDs for the PRE-ACCIDENT window (start_time = incident_start - 15, end_time = incident_start) to compare speeds side-by-side.\n"
            "5. Call get_vehicle_proximity with the two closest vehicles for the INCIDENT window to find the minimum distance and time of closest approach.\n"
            "6. Call compare_vehicle_kinematics again for the INCIDENT window itself to confirm impact kinematics for all parties simultaneously.\n"
            "7. Call evaluate_traffic_rules for each involved vehicle over the full range (pre-accident + incident) to detect violations that preceded the crash.\n"
            "8. Synthesise: combine pre-accident behaviour, proximity data, rule violations, and impact kinematics to state the cause."
        ),
    },
    "relational": {
        "keywords": [
            "interact", "relationship", "between", "conflict",
            "following", "tailgat", "together",
        ],
        "plan": (
            "1. Call search_semantic_events to find the relevant time window.\n"
            "2. Call query_graph_relationships to find which vehicles interacted.\n"
            "3. Call verify_physics_math for the involved vehicles."
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
    "vehicle_specific": {
        "keywords": ["vehicle ", "track id", "track_id"],
        "plan": (
            "1. Call search_semantic_events to find events involving this vehicle.\n"
            "2. Call verify_physics_math to get kinematic statistics.\n"
            "3. Call evaluate_traffic_rules to check for violations.\n"
            "4. Call query_graph_relationships to find interactions with other vehicles."
        ),
    },
    "flow": {
        "keywords": [
            "flow", "count", "zone", "gate", "entry", "exit",
            "od matrix", "origin", "destination", "how many vehicles", "dwell",
        ],
        "plan": (
            "1. Call query_zone_flow to get gate counts and OD pairs."
        ),
    },
    "speed_compliance": {
        "keywords": [
            "85th", "v85", "percentile", "speed limit", "compliance",
            "average speed", "mean speed", "typical speed",
        ],
        "plan": (
            "1. Call get_speed_statistics (track_id=-1) to get V85, mean, max for all vehicles.\n"
            "2. Call evaluate_traffic_rules for vehicles whose max speed exceeds the posted limit.\n"
            "3. Summarise compliance: percentage of vehicles within limit, V85 vs posted limit."
        ),
    },
    "volume": {
        "keywords": [
            "volume", "how many", "count per", "per hour", "per minute",
            "peak hour", "traffic count", "phf", "aadt",
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
    "turning_movements": {
        "keywords": [
            "turning", "turn", "left turn", "right turn", "through", "u-turn",
            "tmc", "approach", "departure", "intersection movement",
        ],
        "plan": (
            "1. Call get_turning_movement_counts to get the full TMC matrix.\n"
            "2. Identify dominant movements and any unexpected movements (e.g. U-turns).\n"
            "3. Call query_zone_flow for dwell times and gate counts to supplement TMC."
        ),
    },
    "conflict": {
        "keywords": [
            "ttc", "time to collision", "conflict", "close call",
            "how close", "dangerous gap", "separation",
        ],
        "plan": (
            "1. Call search_semantic_events to find the conflict event and time window.\n"
            "2. Call query_graph_relationships to identify the two involved vehicles.\n"
            "3. Call get_vehicle_proximity to confirm minimum gap and whether collision occurred.\n"
            "4. Call compute_ttc for the same vehicles and window to get conflict severity level.\n"
            "5. Synthesise: gap + TTC together give the full conflict picture."
        ),
    },
}

_DEFAULT_PLAN = (
    "1. Call search_semantic_events to find relevant events.\n"
    "2. Call search_entity_profiles to find relevant vehicle profiles.\n"
    "3. Call verify_physics_math for any identified vehicles.\n"
    "4. Call evaluate_traffic_rules if violations are suspected."
)


def _select_plan(query: str) -> str:
    """
    Deterministic keyword matcher — returns the first matching plan template.
    Falls back to _DEFAULT_PLAN if no keywords match.
    """
    q = query.lower()
    for template in _PLAN_TEMPLATES.values():
        if any(kw in q for kw in template["keywords"]):
            return template["plan"]
    return _DEFAULT_PLAN

# ---------------------------------------------------------------------------
# LLM — two bindings: full (all 5 tools) and semantic (search only).
# ChatOllama is required for bind_tools(); OllamaLLM is text-only.
# ---------------------------------------------------------------------------
llm = ChatOllama(model="qwen2.5:72b", temperature=0.0)
llm_full = llm.bind_tools(TOOLS_FULL)
llm_semantic = llm.bind_tools(TOOLS_SEMANTIC)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> AgentState:
    """
    Router node: classifies the query intent using embedding cosine-similarity
    and stores the result in state so all downstream nodes can branch on it.
    """
    return {"route": _classify_intent(state["query"])}


def planner_node(state: AgentState) -> AgentState:
    """
    Planner node: selects a deterministic investigation plan from
    _PLAN_TEMPLATES using keyword matching on the query.

    This is the symbolic planning layer — no LLM is called here.
    The plan is always the same for the same query type, making it
    fully auditable and reproducible.

    Only activated for 'full_analysis' queries.
    """
    if state.get("route") != "full_analysis":
        return {"plan": ""}

    plan = _select_plan(state["query"])
    print(f"\n📋 Analysis Plan (symbolic):\n{plan}\n")
    return {"plan": plan}


def initialize(state: AgentState) -> AgentState:
    """
    Entry node: seeds the message history with the route-appropriate system
    prompt and the user's query as a HumanMessage.

    If a plan was produced by the planner, it is appended to the system prompt
    so the agent knows what steps to follow.
    """
    is_full = state.get("route") == "full_analysis"
    system_prompt = _SYSTEM_PROMPT_FULL if is_full else _SYSTEM_PROMPT_SEMANTIC

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
    Selects the tool-bound LLM that matches the pre-computed route, then
    invokes it against the full message history.  The LLM either:
      (a) emits a tool_call  → LangGraph routes to the tools node, loops back
      (b) emits plain text   → tools_condition routes to 'finalize'
    """
    llm_to_use = llm_full if state.get("route") == "full_analysis" else llm_semantic
    response = llm_to_use.invoke(state["messages"])
    return {"messages": [response]}


def finalize(state: AgentState) -> AgentState:
    """
    Extracts the last AIMessage content as the final answer.
    Kept separate so api.py / main.py can still read state['final_summary'].
    """
    last = state["messages"][-1]
    return {"final_summary": last.content}


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

workflow.set_entry_point("route_query")
workflow.add_edge("route_query", "planner")
workflow.add_edge("planner", "initialize")
workflow.add_edge("initialize", "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": "finalize"},
)

workflow.add_edge("tools", "agent")   # observation → next reasoning step
workflow.add_edge("finalize", END)

# recursion_limit caps the agent ↔ tools loop.
# 5 tools × worst-case 5 retries = 25; 30 gives a small safety margin.
#
# IMPORTANT: pass the config dict at invoke time, not via .config attribute.
# The .config attribute approach is not supported in all LangGraph versions.
# Callers must use: agent_app.invoke(state, config=AGENT_INVOKE_CONFIG)
AGENT_INVOKE_CONFIG: dict = {"recursion_limit": 30}
agent_app = workflow.compile()
