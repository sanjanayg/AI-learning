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
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "multi_tenant_rag")

    # Embedding Settings
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # PostgreSQL / Async SQLAlchemy Settings
    DATABASE_URL = os.getenv("DATABASE_URL")
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    # AWS SQS Settings
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")  # full queue URL from AWS console

    # Local file storage for raw uploaded files (worker reads from here)
    FILE_STORAGE_PATH = os.getenv("FILE_STORAGE_PATH", "./data/uploads")


settings = Settings()
