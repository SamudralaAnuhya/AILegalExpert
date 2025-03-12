from typing import TypedDict, Optional

class AgentState(TypedDict):
    """Defines the state structure for the legal RAG workflow"""
    user_query: str
    uploaded_file_content: Optional[str]
    restructured_query: Optional[str]
    hypothetical_document: Optional[str]
    retrieved_context: Optional[str]
    draft: Optional[str]
    confidence_score: float
    requires_feedback: bool
    feedback: Optional[str]
    final_response: Optional[str]
