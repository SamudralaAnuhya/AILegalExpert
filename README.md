Overview
The Legal RAG Assistant is an advanced document analysis tool designed specifically for legal documents. It combines state-of-the-art Retrieval-Augmented Generation techniques with an agentic workflow to provide accurate, context-aware answers to legal questions. The assistant can process various document formats (PDF, DOCX, TXT, and images) and uses semantic understanding to generate reliable responses based on document content.
Features

Advanced RAG Techniques:

Semantic chunking for contextually-aware document segmentation
Hypothetical Document Embedding (HyDE) for improved retrieval
Query refinement tailored for legal context
Two-model approach (speculative RAG) for balanced speed and accuracy


Agentic Workflow:

LangGraph-based workflow orchestration
Confidence-based human feedback loop
Conditional response paths based on context availability


User Experience:

Chat-style interface built with Streamlit
Document upload and extraction
Confidence visualization
Human-in-the-loop feedback for low-confidence responses



Installation
bashCopy# Clone the repository
git clone https://github.com/SamudralaAnuhya/AILegalExpert.git
cd legal-rag-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Requirements
Copystreamlit==1.27.0
langchain==0.0.267
langchain-experimental==0.0.10
langchain_huggingface==0.0.5
langchain_community==0.0.10
langgraph==0.0.14
faiss-cpu==1.7.4
groq==0.4.0
PyPDF2==3.0.1
python-docx==0.8.11
Pillow==9.5.0
pytesseract==0.3.10
Environment Variables
Create a .env file in the project root with the following:
CopyGROQ_API_KEY=your_groq_api_key
Usage

Run the Streamlit app:
bashCopystreamlit run app.py

Upload a legal document (PDF, DOCX, TXT, or an image)
Ask questions about the document in the chat interface
Review responses and provide feedback when prompted

Workflow Explanation

Document Processing: The system extracts text from uploaded documents and creates a semantic vector database.
Query Processing: User questions are refined to match legal terminology before retrieval.
Retrieval: Relevant document sections are retrieved using semantic search, optionally enhanced with HyDE.
Response Generation: A draft response is created with a smaller model, then verified by a larger model.
Feedback Loop: If confidence is low, users can review and provide feedback before generating the final answer.

Models Used

Draft Model: gemma2-9b-it - Efficient for initial response generation
Main Model: llama3-70b-8192 - Powerful for verification and refinement
