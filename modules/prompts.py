def query_refinement_prompt(query: str) -> str:
    """Refine user query for better retrieval."""
    if "what happens" in query.lower():
        return f"What are the legal consequences of {query.lower().replace('what happens', '').strip()}?"
    elif query.lower().startswith("can i"):
        return f"Legal analysis regarding: {query}"
    elif "how to" in query.lower():
        return f"Step-by-step legal guide for {query.lower().replace('how to', '').strip()}"
    else:
        return query if query.endswith((".", "?", "!")) else query + "?"

def hyde_generation_prompt(query: str) -> str:
    """Generates a hypothetical legal document paragraph to improve retrieval."""
    return f"""You are a legal expert. Generate a hypothetical legal document paragraph that directly answers this query.

    **Query:** {query}

    Your document should:
    - Contain factual legal information appearing in real legal texts.
    - Use formal legal language.
    - Include specific details, precedents, or statutes where relevant.
    - Be 2-3 paragraphs long.

    **Hypothetical Legal Document:**
    """

def document_retrieval_prompt(query: str, context: str) -> str:
    """Prompt for RAG retrieval-based answering."""
    return f"""You are a legal assistant answering a question about a legal document.

    **Question:** {query}

    **Relevant excerpts from the document:**
    ```
    {context}
    ```

    **Instructions:**
    - Start with a direct answer to the question.
    - Only use information from the document.
    - Use professional, clear legal language.
    - Organize information logically.
    - If the document lacks information, acknowledge this limitation.

    **Your response:**
    """

def verify_response_prompt(query: str, draft: str, context: str) -> str:
    """Prompt for improving and verifying AI response."""
    return f"""Review and improve this legal response.

    **Original Question:** {query}

    **Draft Response:**
    ```
    {draft}
    ```

    **Relevant document excerpts:**
    ```
    {context}
    ```

    **Instructions:**
    - Ensure the response directly answers the question.
    - Verify accuracy against the document.
    - Use a professional, concise legal tone.
    - Remove any unnecessary commentary.
    - Ensure clarity and structure.

    **Your final response:**
    """

def refine_response_prompt(query, draft, feedback, context):
    """Generates a prompt to refine the AI response based on user feedback."""
    return f"""Refine this legal response based on user feedback.

    **Original Question:**
    {query}

    **Current Response:**
    ```
    {draft}
    ```

    **User Feedback:**
    {feedback}

    **Document Context:**
    ```
    {context}
    ```

    **Your revised response (concise, accurate, professional):**"""
