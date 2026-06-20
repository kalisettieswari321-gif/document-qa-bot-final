"""
Central configuration for the Document Q&A RAG bot.
"""

import os

# ---- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

# ---- Chunking ----------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---- Retrieval ---------------------------------------------------------
TOP_K = 4

# ---- Models ------------------------------------------------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

# ---- ChromaDB ----------------------------------------------------------
COLLECTION_NAME = "document_knowledge_base"

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

# ---- Embedding Retry Settings -----------------------------------------
EMBED_MAX_RETRIES = 3
EMBED_RETRY_BASE_DELAY_SECONDS = 2