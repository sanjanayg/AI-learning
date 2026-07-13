import os

class CacheSettings:
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() in ("true", "1", "yes")
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))
    CACHE_DUPLICATE_THRESHOLD: float = float(os.getenv("CACHE_DUPLICATE_THRESHOLD", "0.98"))
    CACHE_TOP_K: int = int(os.getenv("CACHE_TOP_K", "5"))
    CACHE_TTL_DAYS: int = int(os.getenv("CACHE_TTL_DAYS", "30"))
    CACHE_CLEANUP_INTERVAL_HOURS: int = int(os.getenv("CACHE_CLEANUP_INTERVAL_HOURS", "24"))
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "10000"))
    CACHE_MIN_ANSWER_LENGTH: int = int(os.getenv("CACHE_MIN_ANSWER_LENGTH", "20"))
    CACHE_METRICS_ENABLED: bool = os.getenv("CACHE_METRICS_ENABLED", "True").lower() in ("true", "1", "yes")
    CACHE_COLLECTION_NAME: str = os.getenv("CACHE_COLLECTION_NAME", "semantic_cache")
    
    # Static metadata to match
    PROMPT_VERSION: str = os.getenv("CACHE_PROMPT_VERSION", "v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Expiry settings
    CACHE_UNUSED_EXPIRY_DAYS: int = int(os.getenv("CACHE_UNUSED_EXPIRY_DAYS", "90"))

cache_settings = CacheSettings()
