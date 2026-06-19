"""
ChromaDB persistence helpers shared by ingest.py, query.py, and app.py.

IMPORTANT — why this file does NOT use
`chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction`:

That built-in wrapper internally depends on the legacy `google.generativeai`
package, which Google deprecated in May 2026 in favor of the unified
`google-genai` SDK. Depending on exactly which chromadb version and which
google-generativeai version pip resolves on a given deploy, that wrapper's
internal client construction can pass options (e.g. a `headers` kwarg) that
the resolved SDK version doesn't accept — producing errors like:
    ValueError: ClientOptions does not accept an option 'headers'
This is a version-compatibility issue *inside chromadb's bundled wrapper*,
not something fixable by changing your own code's call signature.

The fix: don't depend on that wrapper at all. We implement our own tiny,
stable embedding function that calls the modern `google-genai` SDK directly.
This has a much smaller, more stable surface area and won't break again
just because chromadb or the Google SDK bump a minor version.
"""

import os
import time

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from . import config

load_dotenv()


def _get_api_key() -> str:
    # Re-read on every call (not just at import time) so this also picks up
    # a key set later via Streamlit secrets in app.py.
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to a .env file, your environment, "
            "or Streamlit secrets."
        )
    return key


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Minimal, dependency-light embedding function using the unified
    `google-genai` SDK's `client.models.embed_content` call.

    document_mode=True  -> task_type="RETRIEVAL_DOCUMENT" (used when indexing)
    document_mode=False -> task_type="RETRIEVAL_QUERY"    (used when searching)
    Using the correct task_type for each side measurably improves retrieval
    relevance versus using one task_type for both.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, document_mode: bool = True):
        self.model_name = model_name
        self.document_mode = document_mode
        self._client = genai.Client(api_key=_get_api_key())

    def __call__(self, input: Documents) -> Embeddings:
        task_type = "RETRIEVAL_DOCUMENT" if self.document_mode else "RETRIEVAL_QUERY"

        last_error = None
        for attempt in range(config.EMBED_MAX_RETRIES):
            try:
                response = self._client.models.embed_content(
                    model=self.model_name,
                    contents=input,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                return [e.values for e in response.embeddings]
            except genai_errors.APIError as e:
                last_error = e
                # Retry on rate limits / transient server errors only.
                status = getattr(e, "code", None)
                if status in (429, 500, 503) and attempt < config.EMBED_MAX_RETRIES - 1:
                    delay = config.EMBED_RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                    print(f"[embedding] {status} error, retrying in {delay}s "
                          f"(attempt {attempt + 1}/{config.EMBED_MAX_RETRIES})...")
                    time.sleep(delay)
                    continue
                raise
        raise last_error  # pragma: no cover — defensive, loop always returns or raises


def get_embedding_function(document_mode: bool = True) -> GeminiEmbeddingFunction:
    return GeminiEmbeddingFunction(model_name=config.EMBEDDING_MODEL, document_mode=document_mode)


def get_collection(db_path: str = config.DB_DIR, create: bool = False, for_query: bool = False):
    """
    for_query=False (default) -> document-mode embedding function (for ingest/add)
    for_query=True            -> query-mode embedding function (for search)
    """
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = get_embedding_function(document_mode=not for_query)

    if create:
        return client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def save_chunks(chunks: list[dict], db_path: str = config.DB_DIR, batch_size: int = 100) -> int:
    """Embed and persist chunks into ChromaDB, in batches to keep requests small
    (the Gemini embedding API caps batch size, and small batches keep memory low)."""
    collection = get_collection(db_path=db_path, create=True, for_query=False)

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        ids = [f"id_{batch_start + i}" for i in range(len(batch))]
        documents = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return len(chunks)
