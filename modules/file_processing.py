import PyPDF2
import docx
import pytesseract
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from config import setup_logging  # Ensure config.py has setup_logging()
from config import logger  # Ensure config.py has logger initialized

logger = setup_logging()


# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extract text from uploaded files
def extract_text(file) -> str:
    try:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return "\n\n".join(p.text for p in docx.Document(file).paragraphs if p.text)
        elif file.type == "text/plain":
            return file.read().decode("utf-8", errors="ignore")
        elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
            return pytesseract.image_to_string(Image.open(file))
    except Exception as e:
        return f"Error extracting text: {e}"

def setup_vector_db(content: str):
    if not content.strip() or embedding_model is None:
        logger.warning("No content or no embedding model.")
        return None  # Return None if no valid content

    semantic_chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )
    documents = semantic_chunker.create_documents([content])
    for i, doc in enumerate(documents):
        print(f"DEBUG: Chunk {i} => {doc.page_content[:300]}...")

    vector_db = FAISS.from_documents(documents, embedding_model)

    logger.info(f"✅ Vector DB created with {len(documents)} chunks.")
    return vector_db  # RETURN the vector DB instead of modifying a global variable
