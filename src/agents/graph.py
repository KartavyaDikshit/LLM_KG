from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import planner_node, extractor_node, validator_node, deduplicator_node, query_node

def create_agentic_workflow(fast_mode=False):
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("deduplicator", deduplicator_node)

    if fast_mode:
        # Turbo Path: Just extract and clean
        workflow.set_entry_point("extractor")
        workflow.add_edge("extractor", "deduplicator")
    else:
        # High-Precision Path
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "extractor")
        workflow.add_edge("extractor", "validator")

        # Conditional edge: if valid, proceed to deduplicator; if not, retry extractor (max 3 iterations)
        def should_continue(state: AgentState):
            if state["is_valid"] or state.get("iterations", 0) >= 3:
                return "deduplicator"
            return "extractor"

        workflow.add_conditional_edges(
            "validator",
            should_continue,
            {
                "deduplicator": "deduplicator",
                "extractor": "extractor"
            }
        )

    workflow.add_edge("deduplicator", END)

    return workflow.compile()

def create_query_workflow():
    """Simple workflow for graph querying."""
    workflow = StateGraph(AgentState)
    workflow.add_node("query_engine", query_node)
    workflow.set_entry_point("query_engine")
    workflow.add_edge("query_engine", END)
    return workflow.compile()
