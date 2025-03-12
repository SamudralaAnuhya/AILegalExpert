from langgraph.graph import StateGraph, END
from modules.nodes import (
    process_uploaded_file,
    query_refinement,
    generate_hyde_document,
    retrieve_legal_context,
    generate_draft_response,
    human_in_the_loop,
    verify_response
)
from modules.utils import AgentState  # Importing the AgentState type definition

def create_workflow():
    """Defines and compiles the query processing workflow."""
    workflow = StateGraph(AgentState)  # Ensure AgentState is defined

    # Add all processing nodes
    workflow.add_node("process_uploaded_file", process_uploaded_file)
    workflow.add_node("query_refinement", query_refinement)
    workflow.add_node("generate_hyde_document", generate_hyde_document)
    workflow.add_node("retrieve_legal_context", retrieve_legal_context)
    workflow.add_node("generate_draft_response", generate_draft_response)
    workflow.add_node("human_in_the_loop", human_in_the_loop)
    workflow.add_node("verify_response", verify_response)

    # Define the workflow path
    workflow.set_entry_point("process_uploaded_file")
    workflow.add_edge("process_uploaded_file", "query_refinement")
    workflow.add_edge("query_refinement", "generate_hyde_document")
    workflow.add_edge("generate_hyde_document", "retrieve_legal_context")
    workflow.add_edge("retrieve_legal_context", "generate_draft_response")

    # Add conditional edge based on confidence
    workflow.add_conditional_edges(
        "generate_draft_response",
        lambda s: "human_in_the_loop" if s["confidence_score"] < 0.7 else "verify_response"
    )

    workflow.add_edge("human_in_the_loop", "verify_response")
    workflow.add_edge("verify_response", END)

    return workflow.compile()
