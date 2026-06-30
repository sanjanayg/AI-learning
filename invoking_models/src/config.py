import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv(
        "GROQ_MODEL",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    )
    GROQ_MODEL_VERSATILE = os.getenv(
        "GROQ_MODEL_VERSATILE",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    )
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "5"))
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
    
    # PDF OCR Settings
    PDF_OCR_CONCURRENCY = int(os.getenv("PDF_OCR_CONCURRENCY", "5"))
    PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "150"))

    # Qdrant Vector DB Settings
    QDRANT_URL = os.getenv("QDRANT_URL")  # e.g., "http://localhost:6333"
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant")  # persistent local directory
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "multi_tenant_rag")
    
    # Embedding Settings
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # PostgreSQL / Async SQLAlchemy Settings
    DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql+asyncpg://postgres:pass@localhost:5432/postgres
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    # ── ID Extraction Config ─────────────────────────────────────────────────
    # Each entry: {"name": str, "pattern": str (raw regex)}
    # Retune patterns here — no code changes needed elsewhere.
    #
    # Pattern rationale (built from real document types seen in eval data):
    #   ALPHA_NUMERIC  — EPA/government refs: EPA-HQ-OW-2011-0049, AB-2024-1193
    #   NUMERIC_LONG   — Pure numeric identifiers ≥6 digits: 139738, 200500019
    #   PREFIXED_CODE  — Invoice/PO codes: INV-00234, PO-2024-9901
    #   CASE_REF       — Year-anchored gov/legal refs: 2015-0019-ADM, 2024-1234-XY
    #
    # To add a new pattern: append another dict to this list.
    # To disable a pattern: remove its entry (no regex flags needed — re.IGNORECASE applied globally).
    ID_PATTERNS: List[Dict[str, Any]] = [
        {
            "name": "ALPHA_NUMERIC",
            # Hyphenated codes: 2–6 uppercase letters, dash, 2–4 digit year, dash, 2–6 digits
            # e.g. EPA-HQ-OW-2011-0049, AB-2024-1193, WQ-2023-001
            "pattern": r"\b[A-Z]{2,6}(?:-[A-Z]{2,6})*-\d{2,4}-\d{2,6}\b",
        },
        {
            "name": "NUMERIC_LONG",
            # Standalone numeric sequences 6–15 digits (invoice #, case #, tracking #)
            # e.g. 139738, 2005000190, 123456789012
            # Excludes years (exactly 4 digits) and versions (e.g. 3.14)
            "pattern": r"(?<!\.)\b\d{6,15}\b(?!\.)",
        },
        {
            "name": "PREFIXED_CODE",
            # Short prefix + 4–8 digit number: INV-00234, PO-9901, REF-20240001
            "pattern": r"\b[A-Z]{2,5}-\d{4,8}\b",
        },
        {
            "name": "CASE_REF",
            # Year-anchored administrative refs: 2015-0019-ADM, 2024-1234-XY
            "pattern": r"\b20\d{2}-\d{4}-[A-Z]{2,4}\b",
        },
        {
            "name": "REF_WITH_DOTS",
            # UN Consolidated List style refs: QI.E.175.04 / TI.B.24.01 / QE.R.128.08
            # Format: 1-2 letters, dot, 1 letter, dot, 1-3 digits, dot, 2 digits, optional trailing dot
            "pattern": r"\b[A-Z]{2}\.[A-Z]\.\d{1,3}\.\d{2}?",
        }
    ]

    # Maximum number of exact-ID match results included in merged output.
    # Capping prevents a false-positive-heavy query from crowding out hybrid results.
    # Exact matches are always placed first in the result ordering.
    MAX_EXACT_MATCHES: int = int(os.getenv("MAX_EXACT_MATCHES", "3"))


settings = Settings()
