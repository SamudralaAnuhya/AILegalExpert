import streamlit as st  # ✅ Streamlit must be imported first
st.set_page_config(page_title="Legal RAG Assistant", layout="wide")
from config import setup_logging
from modules.file_processing import extract_text
from modules.workflow import create_workflow
from modules.nodes import verify_response, refine_with_feedback
from modules.utils import AgentState

# Initialize logging
logger = setup_logging()

# Function to Load External CSS
def load_css():
    """Loads an external CSS file for styling"""
    try:
        with open("static/styles.css") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ CSS file not found. Make sure `static/styles.css` exists.")

# Load Custom CSS
load_css()

# Streamlit UI Configuration


def main():
    """Main Streamlit UI Logic"""
    
    # App header
    st.title("📜 Legal RAG Assistant")
    st.markdown("Upload legal documents and ask questions to get context-aware answers.")

    # Explanation expander
    with st.expander("ℹ️ How it works"):
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
        st.header("📂 Document Upload")
        uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])
        
        if uploaded_file and "uploaded_content" not in st.session_state:
            with st.spinner("🔍 Extracting document text..."):
                file_text = extract_text(uploaded_file)
                if file_text:
                    st.session_state["uploaded_content"] = file_text
                    st.success(f"✅ Document extracted: {len(file_text)} characters")
                    if len(file_text) < 100:
                        st.warning("⚠️ Extracted text is very short. This might affect the quality of responses.")
                else:
                    st.error("❌ Could not extract text from the document. Please try another file.")
        
        # Show document info if available
        if "uploaded_content" in st.session_state:
            st.success("📄 Document loaded successfully")
            if st.button("🗑️ Clear document"):
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
        st.info("⚠️ The system has low confidence in its answer. Please provide feedback or approve.")
    
        # Display draft in a scrollable box with confidence indicator
        confidence = st.session_state.current_state["confidence_score"]
        confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.5 else "confidence-low"
        
        st.markdown("<h3>✍️ Draft Response</h3>", unsafe_allow_html=True)
        
        # Confidence meter
        st.markdown(
        f"""
        <div class="confidence-meter">
            <span>Confidence: {confidence:.2f}</span>
            <div class="confidence-bar">
                <div class="confidence-level {confidence_class}" style="width: {confidence * 100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
        f"""
        <div class="draft-response-box">
            {st.session_state.current_draft}
        </div>
        """, unsafe_allow_html=True)

        # Feedback input
        feedback = st.text_area("✏️ Provide feedback or corrections:", key="feedback_input", height=100)

        col1, col2 = st.columns([1, 1])
        with col1:
            approve_button = st.button("✅ Approve As Is")
        with col2:
            submit_button = st.button("✏️ Submit Feedback")

        # Handle feedback approval
        if approve_button:
            stored_state = st.session_state.current_state
            refined = verify_response(stored_state)
            final_answer = refined["final_response"]
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            del st.session_state.awaiting_feedback
            del st.session_state.current_draft
            del st.session_state.current_state
            st.rerun()
        
        # Handle feedback submission
        if submit_button and feedback:
            stored_state = st.session_state.current_state
            refined = refine_with_feedback(stored_state, feedback)
            refined = verify_response(refined)
            final_answer = refined["final_response"]
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            del st.session_state.awaiting_feedback
            del st.session_state.current_draft
            del st.session_state.current_state
            st.rerun()
    
    else:
        # Chat input for user queries
        user_input = st.chat_input("💬 Ask a legal question about your document...")
        
        if user_input:
            if "uploaded_content" not in st.session_state:
                st.warning("⚠️ Please upload a document first.")
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("🤖 Analyzing document..."):
                        state = AgentState(
                            user_query=user_input,
                            uploaded_file_content=st.session_state["uploaded_content"],
                            restructured_query=None,
                            hypothetical_document=None,
                            retrieved_context=None,
                            draft=None,
                            confidence_score=1.0,
                            requires_feedback=False,
                            feedback=None,
                            final_response=None
                        )

                        workflow = create_workflow()
                        result = workflow.invoke(state)

                        if result["requires_feedback"]:
                            st.session_state.awaiting_feedback = True
                            st.session_state.current_draft = result["draft"]
                            st.session_state.current_state = result
                            st.rerun()
                        else:
                            final_answer = result["final_response"]
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                            st.markdown(final_answer)

    # Disclaimer
    st.markdown(
        """
        <div class="legal-disclaimer">
        ⚖️ This tool is for demonstration purposes only. Not a substitute for professional legal advice.
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
