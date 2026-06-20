"""
Query pipeline: embed the user's question, retrieve the closest chunks from
ChromaDB, build a strictly-grounded prompt, and ask Gemini to answer.
"""

import os
from dotenv import load_dotenv
from google import genai

from . import config
from .vector_store import get_collection

load_dotenv()


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found")
    return genai.Client(api_key=api_key)


SYSTEM_PROMPT = (
    "You are a professional, accurate document Q&A assistant. "
    "Answer the user's question using ONLY the provided document context below. "
    "Cite the sources (filenames and pages) inline next to facts you mention. "
    "If the answer cannot be found in the context, clearly state that the "
    "documents do not contain the answer."
)


def query_rag_pipeline(
    user_query: str,
    db_path: str = config.DB_DIR,
    k: int = config.TOP_K,
) -> dict:

    collection = get_collection(
        db_path=db_path,
        create=False,
        for_query=True,
    )

    results = collection.query(
        query_texts=[user_query],
        n_results=k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_blocks = []
    citations = []

    for doc, meta in zip(documents, metadatas):
        source_name = meta.get("source", "unknown")
        page_num = meta.get("page", "N/A")

        citation = f"{source_name} Page {page_num}"

        context_blocks.append(
            f"[{citation}]\n{doc}"
        )

        citations.append(citation)

    if not context_blocks:
        return {
            "answer": "I am sorry, but the provided documents do not contain the answer to your question.",
            "citations": [],
            "raw_context": [],
        }

    context_payload = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
{SYSTEM_PROMPT}

CONTEXT:
{context_payload}

QUESTION:
{user_query}

ANSWER:
"""

    client = get_client()

    response = client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=prompt,
    )

    answer = response.text if hasattr(response, "text") else str(response)

    return {
        "answer": answer,
        "citations": citations,
        "raw_context": documents,
    }
