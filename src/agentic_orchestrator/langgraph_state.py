from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The original user query — used to seed the first HumanMessage.
    query: str

    # Intent class set by the router node before the agent runs.
    # 'full_analysis'   → all 5 tools available
    # 'semantic_lookup' → only search_semantic_events available (Milvus only)
    route: str

    # Structured analysis plan produced by the PlannerNode.
    # Populated only for 'full_analysis' queries; empty string otherwise.
    # Storing it here makes the system's reasoning transparent and inspectable.
    plan: str

    # Growing message history: HumanMessage → AIMessage (with tool_calls) →
    # ToolMessage (tool result) → AIMessage → ...
    # The add_messages reducer appends each new message rather than overwriting.
    messages: Annotated[list[AnyMessage], add_messages]

    # Final answer extracted from the last AIMessage — kept for API compatibility.
    final_summary: str
