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
                 search_semantic_events    → Milvus ANN (event-level)    (neural)
                 search_entity_profiles    → Milvus ANN (vehicle-level)  (neural)
                 query_graph_relationships → Kùzu Cypher                 (symbolic)
                 verify_physics_math       → DuckDB raw stats            (symbolic)
                 evaluate_traffic_rules    → Rule engine                 (symbolic)
                 query_zone_flow           → DuckDB OD analysis          (symbolic)

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
from langchain_core.messages import SystemMessage, HumanMessage

from .langgraph_state import AgentState
from .hierarchical_router import _classify_intent
from .tools import (
    search_semantic_events,
    search_entity_profiles,
    query_graph_relationships,
    verify_physics_math,
    query_zone_flow,
    evaluate_traffic_rules,
)

# ---------------------------------------------------------------------------
# Tool sets — the router selects which LLM binding to use.
# ToolNode must register all tools so it can execute whichever the LLM calls.
# ---------------------------------------------------------------------------
TOOLS_FULL = [
    search_semantic_events,
    search_entity_profiles,   # Vehicle-level longitudinal behavioral profiles
    query_graph_relationships,
    verify_physics_math,
    evaluate_traffic_rules,   # Symbolic Rule Engine
    query_zone_flow,
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

Decision rules:
- Global behavioral questions ("most aggressive", "which vehicle sped the most"): tool 2.
- Safety/violation questions ("did vehicle 4 brake hard?"): tools 1 → 5.
- Relational questions ("which vehicles interacted?"): tools 1 → 3.
- Combined safety + relationships: tools 1 → 3 → 5.
- Full incident reconstruction: tools 1 → 3 → 5 → 4 (raw stats for extra context).
- Flow/count/OD questions: tool 6 directly.
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
# Planner prompt — produces an explicit investigation plan before ReAct.
# ---------------------------------------------------------------------------
_PLANNER_PROMPT = (
    "You are a traffic analysis planning system. "
    "Decompose the following query into a concrete, ordered investigation plan (3-5 steps max).\n\n"
    "For each step specify which tool to use and what specific data to look for.\n\n"
    "Available tools: search_semantic_events, search_entity_profiles, "
    "query_graph_relationships, verify_physics_math, evaluate_traffic_rules, query_zone_flow\n\n"
    "Query: {query}\n\n"
    "Output a numbered list only. Be specific and concise."
)

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
    Planner node: decomposes the user's query into a structured investigation
    plan BEFORE the ReAct loop begins.

    This is the meta-reasoning / symbolic planning layer.  By committing to a
    plan upfront, the system's reasoning becomes transparent (the plan is stored
    in state['plan'] and visible to callers) and the agent LLM is less likely
    to skip steps or call tools redundantly.

    Only activated for 'full_analysis' queries.  Simple semantic lookups do
    not need a plan.
    """
    if state.get("route") != "full_analysis":
        return {"plan": ""}

    response = llm.invoke(
        [HumanMessage(content=_PLANNER_PROMPT.format(query=state["query"]))]
    )
    plan = response.content
    print(f"\n📋 Analysis Plan:\n{plan}\n")
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
