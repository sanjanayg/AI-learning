import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv(
        "GROQ_MODEL",
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


settings = Settings()