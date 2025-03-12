from modules.file_processing import setup_vector_db
from modules.prompts import (
    query_refinement_prompt,
    hyde_generation_prompt,
    document_retrieval_prompt,
    verify_response_prompt,
    refine_response_prompt
)
from config import MAIN_MODEL , DRAFT_MODEL ,logger
from groq import Groq



from config import GROQ_API_KEY  
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None  # Ensure it’s valid



# def process_uploaded_file(state):
#     """Processes uploaded file and creates vector database."""
#     state["vector_db"] = setup_vector_db(state["uploaded_file_content"])
#     return state

def process_uploaded_file(state):
    """Processes uploaded file and creates vector database."""
    content = state.get("uploaded_file_content", "")
    if not content:
        state["vector_db"] = None
        return state
    vector_db = setup_vector_db(content)  # Ensure this function RETURNS a valid vector DB
    state["vector_db"] = vector_db  # Assign the vector_db properly
    return state




def query_refinement(state):
    """Refines the user query for better retrieval."""
    state["restructured_query"] = query_refinement_prompt(state["user_query"])
    return state

def generate_hyde_document(state):
    """Generates a hypothetical document paragraph to improve retrieval."""
    query = state["restructured_query"] or state["user_query"]
    prompt = hyde_generation_prompt(query)

    try:
        response = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=350
        )
        state["hypothetical_document"] = response.choices[0].message.content.strip()
    except Exception as e:
        state["hypothetical_document"] = f"Error generating document: {e}"

    return state

def retrieve_legal_context(state):
    vector_db = state.get("vector_db")
    if not vector_db:
        state["retrieved_context"] = "⚠️ No vector database found..."
        return state

    # If you have HyDE text, use it as the query
    hyde_text = state.get("hypothetical_document", "")
    query_text = hyde_text.strip() if hyde_text.strip() else (
        state.get("restructured_query") or state.get("user_query")
    )

    try:
        results = vector_db.similarity_search(query_text, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        state["retrieved_context"] = context if context else "⚠️ No relevant excerpts found."
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        state["retrieved_context"] = "⚠️ Error retrieving legal context."
    return state




def generate_draft_response(state):
    context = state.get("retrieved_context", "")
    query = state.get("user_query")
    
    if not context.strip():
        logger.warning("No document excerpts found. Using general legal knowledge.")
    
    prompt = f"""
    You are a legal assistant answering a legal question based on a provided document.
    
    **Question:** {query}
    
    **Relevant Document Excerpts:** 
    {context if context else "⚠️ No relevant excerpts found in document."}
    
    **Instructions:**
    - If document excerpts are provided, use them **exclusively**.
    - If no excerpts exist, state clearly: "The document does not explicitly mention this topic."
    - Keep responses legally accurate and professional.
    - Ensure clarity and structure.
    
    Your response:
    """

    response = client.chat.completions.create(
        model=MAIN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )
    
    state["draft"] = response.choices[0].message.content.strip()
    # Set confidence_score based on context length or some heuristic:
    state["confidence_score"] = 0.65 if len(context) > 500 else 0.55
    return state


def verify_response(state):
    """Verifies and improves AI response."""
    draft = state["draft"]
    query = state["restructured_query"] or state["user_query"]
    context = state.get("retrieved_context", "")

    prompt = verify_response_prompt(query, draft, context)

    try:
        response = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        state["final_response"] = response.choices[0].message.content.strip()
    except Exception as e:
        state["final_response"] = f"Error verifying response: {e}"

    return state

def human_in_the_loop(state):
    """Triggers human review if AI confidence is low."""
    state["requires_feedback"] = True
    return state

def refine_with_feedback(state, feedback):
    """Refines AI response based on user feedback using a structured prompt."""
    query = state.get("user_query")
    context = state.get("retrieved_context", "")
    draft = state.get("draft", "")

    prompt = refine_response_prompt(query, draft, feedback, context)

    try:
        response = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        state["draft"] = response.choices[0].message.content.strip()
    except Exception as e:
        state["draft"] = f"{draft}\n\n**Feedback Incorporated:** {feedback}"

    state["requires_feedback"] = False
    return state


