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
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    
    # FIXED: Replaced colon (type hint) with standard assignment for runtime execution
    PDF_OCR_CONCURRENCY = int(os.getenv("PDF_OCR_CONCURRENCY", "5"))
    
    # FIXED: Added the missing DPI configuration property
    PDF_OCR_DPI = int(os.getenv("PDF_OCR_DPI", "150"))

settings = Settings()
