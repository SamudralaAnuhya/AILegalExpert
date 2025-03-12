import os
import logging
import streamlit as st
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import PyPDF2
import docx
import pytesseract
from PIL import Image
from groq import Groq

# --------------------------------------------------
# Configuration
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Legal RAG Assistant", layout="wide")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Please set the GROQ_API_KEY environment variable.")
client = Groq(api_key=api_key) if api_key else None

# Define models
# MAIN_MODEL = "llama-3.3-70b-specdec"  # Best for legal reasoning and analysis
# DRAFT_MODEL = "qwen-qwq-32b"  # Good balance of quality and speed
DRAFT_MODEL = "gemma2-9b-it"  
MAIN_MODEL = "llama3-70b-8192"


try:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to load embeddings: {e}")
    embedding_model = None

# Global vector DB
vector_db = None

# --------------------------------------------------
# AgentState Definition
# --------------------------------------------------
class AgentState(TypedDict):
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

# --------------------------------------------------
# File Extraction & Vector DB Setup
# --------------------------------------------------
def extract_text(file) -> str:
    if not file:
        return ""
    try:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text)
        elif file.type == "text/plain":
            text = file.read().decode("utf-8", errors="ignore")
        elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
        else:
            st.warning(f"Unsupported file format: {file.type}")
            return ""
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""


def setup_vector_db(content: str):
    global vector_db
    if not content.strip() or embedding_model is None:
        logger.warning("No content or no embedding model.")
        return
    # Use SemanticChunker from LangChain Experimental for semantic chunking
    
    semantic_chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=90
    )
    # Create semantic documents (chunks) from the full content
    documents = semantic_chunker.create_documents([content])
    vector_db = FAISS.from_documents(documents, embedding_model)
    logger.info("Vector DB created with %d chunks.", len(documents))



# --------------------------------------------------
# Workflow Nodes
# --------------------------------------------------
def process_uploaded_file(state: AgentState) -> AgentState:
    global vector_db
    file_text = state.get("uploaded_file_content", "")
    if file_text and embedding_model:
        setup_vector_db(file_text)
    return state

def query_refinement(state: AgentState) -> AgentState:
    query = state["user_query"]
    refined = query
    if "what happens" in query.lower():
        refined = f"What are the legal consequences of {query.lower().replace('what happens', '').strip()} in the U.S.?"
    elif query.lower().startswith("can i"):
        refined = f"Legal analysis regarding: {query}"
    elif "how to" in query.lower():
        refined = f"Step-by-step legal guide for {query.lower().replace('how to', '').strip()}"
    else:
        if not (query.endswith(".") or query.endswith("?") or query.endswith("!")):
            refined += "?"
    state["restructured_query"] = refined
    return state

def generate_hyde_document(state: AgentState) -> AgentState:
    if client is None:
        state["hypothetical_document"] = ""
        return state
    
    query = state["restructured_query"] or state["user_query"]
    prompt = f"""You are a legal expert. Generate a hypothetical legal document paragraph that would be directly answers this query The document should be detailed and in-depth.:

    Query: {query}

    Your hypothetical document should:
    1. Contain factual legal information that might appear in legal texts
    2. Be written in formal legal language
    3. Contain specific details, precedents, or statutes that would help answer the query
    4. Be 2-3 paragraphs in length

    Hypothetical Legal Document:"""

    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=350
        )
        state["hypothetical_document"] = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"HyDE generation error: {e}")
        state["hypothetical_document"] = ""
    return state

def retrieve_legal_context(state: AgentState) -> AgentState:
    global vector_db
    
    if not vector_db:
        state["retrieved_context"] = ""
        return state
    
    # Use HyDE approach for better retrieval
    hyde_text = state.get("hypothetical_document", "")
    original_query = state["restructured_query"] or state["user_query"]
    
    # If HyDE failed, fall back to original query
    query_text = hyde_text if hyde_text else original_query
    
    try:
        # Get relevant document chunks
        results = vector_db.similarity_search(query_text, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        state["retrieved_context"] = context
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        state["retrieved_context"] = ""
    return state


def generate_draft_response(state: AgentState) -> AgentState:
    if client is None:
        state["draft"] = "LLM client not initialized."
        return state
    
    query = state["restructured_query"] or state["user_query"]
    context = state.get("retrieved_context", "")
    
    # If no context was retrieved
    if not context.strip():
        prompt = f"""Answer the following legal question based on general legal knowledge:

Question: {query}

Important instructions:
- Start with a direct answer to the question
- Explain that no specific information was found in the uploaded document
- Provide general legal information that might be helpful
- Request more specific documents if appropriate

Your response should be professional and concise, focused directly on answering the user's question."""
        
        try:
            resp = client.chat.completions.create(
                model=DRAFT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250
            )
            state["draft"] = resp.choices[0].message.content.strip()
            state["confidence_score"] = 0.4  # Low confidence when no context found
        except Exception as e:
            logger.error(f"Draft generation error: {e}")
            state["draft"] = "I couldn't find relevant information in your document. Could you please clarify your question or upload a more relevant document?"
            state["confidence_score"] = 0.3
        return state

    # Normal RAG response with context
    prompt = f"""You are a legal assistant answering a question about a legal document.

Question: {query}

Relevant excerpts from the legal document:
```
{context}
```

Instructions:
- Start with a direct answer to the question
- Only use information from the provided document excerpts
- Use clear, professional legal language
- Organize information logically with clear sections if needed
- If the document doesn't fully address the question, acknowledge this limitation

Your response should be professional and focused directly on answering the user's question without meta-commentary about your process."""

    try:
        resp = client.chat.completions.create(
            model=DRAFT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        state["draft"] = resp.choices[0].message.content.strip()
        
        # Use a dynamic confidence score based on the availability of context
        state["confidence_score"] = 0.65 if len(context) > 500 else 0.55
    except Exception as e:
        logger.error(f"Draft generation error: {e}")
        state["draft"] = f"I encountered an issue generating a response. Please try again or rephrase your question."
        state["confidence_score"] = 0.3
    
    return state


def verify_response(state: AgentState) -> AgentState:
    if client is None:
        state["final_response"] = state["draft"]
        return state
    
    draft = state["draft"]
    query = state["restructured_query"] or state["user_query"]
    context = state.get("retrieved_context", "")
    
    prompt = f"""Review and improve this legal response.

Original Question: {query}

Draft Response:
```
{draft}
```

Relevant document excerpts:
```
{context}
```

Instructions:
- Ensure the response directly answers the original question first
- Verify all legal information is accurate based on the document
- Keep a professional, concise tone appropriate for legal communication
- Remove any meta-commentary about the response crafting process
- Make sure the response is well-structured and easy to understand
- Ensure the response only discusses information from the document or clearly marks general legal information

Your final response:"""

    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        verification_result = resp.choices[0].message.content.strip()
        
        # Format the final response with proper markdown
        if not "disclaimer" in verification_result.lower():
            verification_result += "\n\n**Disclaimer:** This is AI-assisted information based on document analysis, not legal advice. Consult an attorney."
            
        state["final_response"] = verification_result
    except Exception as e:
        logger.error(f"Verification error: {e}")
        
        # Fall back to draft with disclaimer
        if not "disclaimer" in draft.lower():
            state["final_response"] = draft + "\n\n**Disclaimer:** This is AI-assisted information based on document analysis, not legal advice. Consult an attorney."
        else:
            state["final_response"] = draft
    
    return state



def human_in_the_loop(state: AgentState) -> AgentState:
    state["requires_feedback"] = True
    return state

def refine_with_feedback(state: AgentState, feedback: str) -> AgentState:
    if client is None:
        state["draft"] = state["draft"] + f"\n\nFeedback incorporated: {feedback}"
        state["requires_feedback"] = False
        return state
        
    draft = state["draft"]
    query = state["restructured_query"] or state["user_query"]
    context = state.get("retrieved_context", "")
    
    prompt = f"""Refine this legal response based on user feedback.

Original Question: {query}

Current Response:
```
{draft}
```

User Feedback:
{feedback}

Document Context:
```
{context}
```

Instructions:
- Address the user's feedback directly
- Keep the focus on answering the original question
- Incorporate any corrections or additional information from the feedback
- Maintain a professional, concise legal communication style with bullet points
- Ensure the response is factually accurate based on the document

"""
    
    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        state["draft"] = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Feedback incorporation error: {e}")
        state["draft"] = f"{draft}\n\n**Feedback Incorporated:** {feedback}"
    
    state["requires_feedback"] = False
    return state


# --------------------------------------------------
# Build Workflow Using LangGraph
# --------------------------------------------------
def create_workflow():
    workflow = StateGraph(AgentState)
    
    # Add all nodes
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

# --------------------------------------------------
# Streamlit UI (Chat-Style)
# --------------------------------------------------
def main():
    # Custom CSS for nicer fonts
    st.markdown(
    """
    <style>
    .stChatMessage, .stChatMessage p {
        font-family: "Arial", sans-serif;
        font-size: 16px;
        line-height: 1.5;
    }
    
    /* Styled box for draft responses */
    .draft-response-box {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        background-color: #f9f9f9;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Confidence indicator styles */
    .confidence-meter {
        margin-top: 10px;
        margin-bottom: 15px;
    }
    
    .confidence-bar {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        margin-top: 5px;
    }
    
    .confidence-level {
        height: 100%;
        border-radius: 4px;
    }
    
    .confidence-high {
        background-color: #4CAF50;
    }
    
    .confidence-medium {
        background-color: #FFC107;
    }
    
    .confidence-low {
        background-color: #F44336;
    }
    
    .legal-disclaimer {
        font-size: 12px;
        color: #666;
        border-top: 1px solid #eee;
        padding-top: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    
    # App header
    st.title("📜 Legal RAG Assistant")
    st.markdown("Upload legal documents and ask questions to get context-aware answers")
    
    # Explanation expander
    with st.expander("How it works"):
        st.write("""
        This Legal RAG (Retrieval-Augmented Generation) Assistant helps you get answers from your legal documents:
        
        1. **Upload a document** using the sidebar uploader (PDF, DOCX, TXT or images)
        2. **Ask legal questions** specific to the document content
        3. The system will:
           - Refine your query for legal context
           - Generate a hypothetical document to improve retrieval (HyDE technique)
           - Search for relevant sections in your uploaded document
           - Generate a response based on the retrieved content
           - Request your feedback if confidence is low
           
        **Note:** This is a demonstration tool and does not provide actual legal advice.
        """)

    # Sidebar: File upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a legal document", 
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"]
        )
        
        if uploaded_file and "uploaded_content" not in st.session_state:
            with st.spinner("Extracting document text..."):
                file_text = extract_text(uploaded_file)
                if file_text:
                    st.session_state["uploaded_content"] = file_text
                    st.success(f"Document extracted: {len(file_text)} characters")
                    if len(file_text) < 100:
                        st.warning("The extracted text is very short. This might affect the quality of responses.")
                else:
                    st.error("Could not extract text from the document. Please try another file.")
        
        # Show document info if available
        if "uploaded_content" in st.session_state:
            st.success("✓ Document loaded")
            if st.button("Clear document"):
                if "uploaded_content" in st.session_state:
                    del st.session_state["uploaded_content"]
                st.rerun()

    # Initialize conversation messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Handle feedback workflow if in progress
    if "awaiting_feedback" in st.session_state and st.session_state.awaiting_feedback:
        st.info("The system has low confidence in its answer. Please provide feedback or approve.")
    
        # Display draft in a scrollable box with confidence indicator
        confidence = st.session_state.current_state["confidence_score"]
        confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.5 else "confidence-low"
        
        st.markdown("<h3>Draft Response</h3>", unsafe_allow_html=True)
        # Confidence meter
        st.markdown(
        f"""
        <div class="confidence-meter">
            <span>Confidence: {confidence:.2f}</span>
            <div class="confidence-bar">
                <div class="confidence-level {confidence_class}" style="width: {confidence * 100}%;"></div>
            </div>
        </div>
        """, 
        unsafe_allow_html=True)

        st.markdown(
        f"""
        <div class="draft-response-box">
            {st.session_state.current_draft}
        </div>
        """, 
        unsafe_allow_html=True
    )


        # st.markdown(st.session_state.current_draft)
        
        # Create columns for feedback input and buttons
        feedback_col1, feedback_col2 = st.columns([3, 1])
        
        with feedback_col1:
            feedback = st.text_area("Your feedback or corrections:", key="feedback_input", height=100)
        
        with feedback_col2:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                approve_button = st.button("Approve As Is")
            with col2_2:
                submit_button = st.button("Submit Feedback")
        
        # Handle approve button
        if approve_button:
            # Use the stored state to complete the workflow
            stored_state = st.session_state.current_state
            refined = verify_response(stored_state)
            final_answer = refined["final_response"]
            
            # Add final answer to conversation
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            # Clear the feedback session state
            del st.session_state.awaiting_feedback
            del st.session_state.current_draft
            del st.session_state.current_state
            
            st.rerun()
            
        # Handle feedback submission
        if submit_button and feedback:
            # Use the stored state to refine with feedback
            stored_state = st.session_state.current_state
            refined = refine_with_feedback(stored_state, feedback)
            refined = verify_response(refined)
            final_answer = refined["final_response"]
            
            # Add final answer to conversation
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            # Clear the feedback session state
            del st.session_state.awaiting_feedback
            del st.session_state.current_draft
            del st.session_state.current_state
            
            st.rerun()
    
    else:
        # Normal input flow - use chat input
        user_input = st.chat_input("Ask a legal question about your document...")
        
        if user_input:
            if "uploaded_content" not in st.session_state:
                st.warning("Please upload a document first.")
            else:
                # Add user message to conversation
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                # Show typing indicator
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document..."):
                        # Build initial state for workflow
                        state: AgentState = {
                            "user_query": user_input,
                            "uploaded_file_content": st.session_state["uploaded_content"],
                            "restructured_query": None,
                            "hypothetical_document": None,
                            "retrieved_context": None,
                            "draft": None,
                            "confidence_score": 1.0,
                            "requires_feedback": False,
                            "feedback": None,
                            "final_response": None
                        }

                        workflow = create_workflow()
                        result = workflow.invoke(state)

                        # If human feedback is needed, enter feedback collection mode
                        if result["requires_feedback"]:
                            # Store state and draft for feedback handling
                            st.session_state.awaiting_feedback = True
                            st.session_state.current_draft = result["draft"]
                            st.session_state.current_state = result
                            st.rerun()
                        else:
                            # No feedback required, display final response
                            final_answer = result["final_response"]
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                            st.markdown(final_answer)

    # Add a footer
    st.markdown(
        """
        <div class="legal-disclaimer">
        This tool is for demonstration purposes only. Not a substitute for professional legal advice.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
