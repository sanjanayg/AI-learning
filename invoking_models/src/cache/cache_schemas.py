from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CacheEntryPayload(BaseModel):
    id: str
    query: str
    query_embedding: Optional[List[float]] = None
    answer: str
    tenant_id: str
    intent: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    kb_version: int
    embedding_model: str
    llm_model: str
    prompt_version: str
    created_at: str  # ISO-formatted string
    expires_at: str  # ISO-formatted string
    last_accessed: str  # ISO-formatted string
    access_count: int = 1
    source_type: str = "LLM"  # "CACHE" or "LLM"

class CacheLookupResult(BaseModel):
    hit: bool
    entry: Optional[CacheEntryPayload] = None
    similarity_score: Optional[float] = None
    reason: Optional[str] = None
