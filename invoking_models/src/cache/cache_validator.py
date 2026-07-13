import json
import logging
import asyncio
from typing import Dict, Any, Tuple
from services.llm_service import LLMService
from cache.cache_config import cache_settings
from cache.cache_schemas import CacheEntryPayload

logger = logging.getLogger(__name__)

INTENT_ENTITY_PROMPT = """You are an intent classifier and entity extractor.
Analyze the user query and extract:
1. "intent": A brief label for the user intent (e.g., "query_documentation", "compare_data", "general_qa", "unclear").
2. "entities": A JSON object containing key entities. Extract numeric counts, years, metrics, names, etc. (e.g., {"year": 2024, "count": 5}). Only include key specific entities that differentiate questions (like "top 5 customer" -> {"count": 5, "type": "customer"}, "sales in 2025" -> {"year": 2025}).

Respond with ONLY a JSON object, no other text, no markdown fences:
{"intent": "intent_label", "entities": {"key": "value"}}
"""

class CacheValidator:
    def __init__(self):
        self.llm_service = LLMService()

    async def extract_intent_and_entities(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Calls a fast, cheap model (openai/gpt-oss-20b) to classify intent and extract entities.
        """
        try:
            # We use LLMService's provider client directly to make the call
            provider = self.llm_service.provider
            
            response = await asyncio.to_thread(
                provider.client.chat.completions.create,
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": INTENT_ENTITY_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=150,
            )

            raw = response.choices[0].message.content.strip()
            # Clean up markdown code block wrappers if any
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            parsed = json.loads(raw)
            intent = parsed.get("intent", "general_qa")
            entities = parsed.get("entities", {})
            if not isinstance(entities, dict):
                entities = {}
            return intent, entities

        except Exception as e:
            logger.warning("Failed to extract intent and entities via lightweight LLM: %s. Falling back to default.", e)
            return "general_qa", {}

    @staticmethod
    def validate(
        entry: CacheEntryPayload,
        tenant_id: str,
        kb_version: int,
        prompt_version: str,
        embedding_model: str,
        llm_model: str,
        intent: str,
        entities: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validates if the cached entry matches the current query context.
        All parameters must match:
        - tenant_id
        - kb_version
        - prompt_version
        - embedding_model
        - llm_model
        - intent
        - entities (must have exact matching values for common keys)
        """
        if entry.tenant_id != tenant_id:
            return False, "Tenant ID mismatch"
        
        if entry.kb_version != kb_version:
            return False, f"KB version mismatch: cache has {entry.kb_version}, query requires {kb_version}"
            
        if entry.prompt_version != prompt_version:
            return False, f"Prompt version mismatch: cache has {entry.prompt_version}, query requires {prompt_version}"
            
        if entry.embedding_model != embedding_model:
            return False, f"Embedding model mismatch: cache has {entry.embedding_model}, query requires {embedding_model}"
            
        if entry.llm_model != llm_model:
            return False, f"LLM model mismatch: cache has {entry.llm_model}, query requires {llm_model}"
            
        if entry.intent != intent:
            return False, f"Intent mismatch: cache has '{entry.intent}', query has '{intent}'"
            
        # Entities matching
        # All entities extracted in the current query must match exactly with what is in the cache entry.
        # Check both ways or key-by-key for important keys.
        cached_entities = entry.entities or {}
        
        # If the set of keys is different, check if any values mismatch
        all_keys = set(cached_entities.keys()).union(set(entities.keys()))
        for key in all_keys:
            val_cached = cached_entities.get(key)
            val_query = entities.get(key)
            # Standardize type comparison (e.g. integer vs string representation)
            if val_cached != val_query:
                # Try comparing as strings if both are not None
                if val_cached is not None and val_query is not None:
                    if str(val_cached) != str(val_query):
                        return False, f"Entity mismatch for key '{key}': cache has {val_cached}, query has {val_query}"
                else:
                    return False, f"Entity mismatch for key '{key}': cache has {val_cached}, query has {val_query}"

        return True, "Valid"
