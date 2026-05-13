from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import planner_node, extractor_node, validator_node

def create_agentic_workflow():
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("validator", validator_node)

    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "extractor")
    workflow.add_edge("extractor", "validator")

    # Conditional edge: if valid, end; if not, retry extractor (max 3 iterations)
    def should_continue(state: AgentState):
        if state["is_valid"] or state.get("iterations", 0) >= 3:
            return END
        return "extractor"

    workflow.add_conditional_edges(
        "validator",
        should_continue,
        {
            END: END,
            "extractor": "extractor"
        }
    )

    return workflow.compile()
