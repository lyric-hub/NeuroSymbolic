from typing import Annotated, List, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The original user query — used to seed the first HumanMessage.
    query: str

    # Intent class set by the router node before the agent runs.
    # 'full_analysis'   → all tools available
    # 'semantic_lookup' → only search_semantic_events available (Milvus only)
    route: str

    # Human-readable explanation of why the router chose this route.
    # Included in final_summary so users know what analysis depth was applied.
    routing_explanation: str

    # Structured analysis plan produced by the PlannerNode.
    # Populated only for 'full_analysis' queries; empty string otherwise.
    # Storing it here makes the system's reasoning transparent and inspectable.
    plan: str

    # Key of the matched plan template (e.g. "incident", "vehicle_type").
    # Used by agent_node to bind only the tools relevant to this plan
    # instead of all 20 — reduces the LLM's decision space to ≤6 tools.
    plan_key: str

    # Growing message history: HumanMessage → AIMessage (with tool_calls) →
    # ToolMessage (tool result) → AIMessage → ...
    # The add_messages reducer appends each new message rather than overwriting.
    messages: Annotated[list[AnyMessage], add_messages]

    # Final answer extracted from the last AIMessage — kept for API compatibility.
    final_summary: str

    # Ordered list of tool calls made during this session — extracted from the
    # message history in finalize() and written to DuckDB analysis_sessions.
    # Each step: {step, tool, args, output_excerpt}.
    reasoning_steps: List[dict]

    # UUID identifying this query session — links final_summary to its
    # reasoning trace in the analysis_sessions DuckDB table.
    session_id: str

    # Neural-symbolic contradiction flags produced by the contradiction_check
    # node.  Non-empty list means the VLM and rule engine disagreed.
    contradictions: List[str]
