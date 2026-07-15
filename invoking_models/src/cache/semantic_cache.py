import uuid
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
)

from cache.cache_config import cache_settings
from cache.cache_schemas import CacheEntryPayload, CacheLookupResult
from cache.cache_validator import CacheValidator
from cache.cache_metrics import metrics_service
from rag.vector_store import QdrantStore

logger = logging.getLogger(__name__)

class SemanticCacheService:
    def __init__(self):
        self.collection_name = cache_settings.CACHE_COLLECTION_NAME
        # Reuse singleton Qdrant Client from QdrantStore
        self.client = QdrantStore._get_client()
        self._ensure_collection_exists()
        
        # Concurrency: Single-flight request coalescing state
        self._inflight_events: Dict[Tuple[str, str], asyncio.Event] = {}
        self._inflight_lock = asyncio.Lock()

    def _ensure_collection_exists(self):
        """
        Checks for collection existence. Initializes it if missing, configuring Cosine similarity
        and programmatically building payload indexes for tenant isolation and cleanups.
        """
        try:
            exists = self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info("Creating semantic cache collection '%s' in Qdrant...", self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Matches SentenceTransformer all-MiniLM-L6-v2 vector dimension
                        distance=Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=16,
                        ef_construct=100,
                    )
                )
                
                # Indexes for tenant isolation and cleanups
                logger.info("Creating KEYWORD index on 'tenant_id'")
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="tenant_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="expires_at",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="kb_version",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
        except Exception as exc:
            logger.exception("Failed to ensure/create Qdrant cache collection: %s", self.collection_name)
            raise RuntimeError(f"Qdrant cache collection setup failed: {str(exc)}") from exc

    async def lookup(
        self,
        query: str,
        query_embedding: list[float],
        tenant_id: str,
        kb_version: int,
        prompt_version: str,
        embedding_model: str,
        llm_model: str,
        intent: str,
        entities: Dict[str, Any]
    ) -> CacheLookupResult:
        """
        Searches the semantic cache for a match.
        Applies tenant isolation filter in Qdrant, retrieves top K, and validates candidates in python.
        """
        if not cache_settings.CACHE_ENABLED:
            return CacheLookupResult(hit=False, reason="Cache disabled")

        start_time = asyncio.get_event_loop().time()

        try:
            # Query Qdrant with tenant filter
            tenant_filter = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            )

            # Retrieve top matches
            search_response = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=tenant_filter,
                limit=cache_settings.CACHE_TOP_K,
                with_payload=True
            )

            if not search_response.points:
                lookup_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
                metrics_service.record_miss("below_threshold", lookup_time_ms)
                return CacheLookupResult(hit=False, reason="No cache entries for tenant")

            # Evaluate candidates
            for hit in search_response.points:
                score = hit.score
                payload = hit.payload
                # Check similarity threshold
                if score < cache_settings.CACHE_SIMILARITY_THRESHOLD:
                    continue

                try:
                    entry = CacheEntryPayload(**payload)
                except Exception as e:
                    logger.warning("Failed to deserialize cache payload: %s", e)
                    continue

                # Check expiration
                expires_at = datetime.fromisoformat(entry.expires_at)
                if expires_at < datetime.now(timezone.utc):
                    # Record expired miss
                    lookup_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
                    metrics_service.record_miss("expired", lookup_time_ms)
                    # Let's delete this expired entry asynchronously
                    asyncio.create_task(self.delete_entry(hit.id))
                    continue

                # Validate metadata fields
                valid, mismatch_reason = CacheValidator.validate(
                    entry=entry,
                    tenant_id=tenant_id,
                    kb_version=kb_version,
                    prompt_version=prompt_version,
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                    intent=intent,
                    entities=entities
                )

                if valid:
                    # Valid cache hit!
                    lookup_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
                    metrics_service.record_hit(lookup_time_ms)
                    
                    # Update access statistics asynchronously
                    asyncio.create_task(self.update_access(hit.id, entry))
                    
                    # Change source_type to CACHE when returning
                    entry.source_type = "CACHE"
                    
                    logger.info(
                        "Cache Hit: Query='%s', Similarity=%.4f, Cache ID=%s, Lookup Time=%.2fms",
                        query, score, hit.id, lookup_time_ms
                    )
                    return CacheLookupResult(hit=True, entry=entry, similarity_score=score)
                else:
                    logger.info("Cache candidate metadata mismatch: %s", mismatch_reason)

            # If we reached here, no candidate passed
            lookup_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000.0
            metrics_service.record_miss("metadata_mismatch", lookup_time_ms)
            return CacheLookupResult(hit=False, reason="Metadata mismatch or below similarity threshold")

        except Exception as e:
            logger.error("Error performing cache lookup: %s", e)
            return CacheLookupResult(hit=False, reason=f"Lookup error: {str(e)}")

    async def store(
        self,
        query: str,
        query_embedding: list[float],
        answer: str,
        tenant_id: str,
        intent: str,
        entities: Dict[str, Any],
        kb_version: int,
        llm_model: str,
        ui_answer: str | None = None
    ) -> Optional[str]:
        """
        Stores a new cache entry or updates an existing duplicate cache entry.
        """
        if not cache_settings.CACHE_ENABLED:
            return None

        # Guard: Minimum answer length
        if len(answer) < cache_settings.CACHE_MIN_ANSWER_LENGTH:
            return None

        try:
            # 1. Check for duplicates (semantically identical entries) to prevent multiple copies
            tenant_filter = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            )

            search_response = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=tenant_filter,
                limit=1,
                with_payload=True
            )

            if search_response.points:
                best_hit = search_response.points[0]
                if best_hit.score >= cache_settings.CACHE_DUPLICATE_THRESHOLD:
                    # Update the existing entry instead of creating a new duplicate
                    logger.info(
                        "Duplicate cache entry found (score=%.4f >= %.4f). Updating access metadata for point %s.",
                        best_hit.score, cache_settings.CACHE_DUPLICATE_THRESHOLD, best_hit.id
                    )
                    try:
                        entry = CacheEntryPayload(**best_hit.payload)
                        await self.update_access(best_hit.id, entry)
                        return best_hit.id
                    except Exception as e:
                        logger.warning("Failed to update duplicate cache entry, writing new point: %s", e)

            # 2. Write new entry
            point_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(days=cache_settings.CACHE_TTL_DAYS)

            payload = CacheEntryPayload(
                id=point_id,
                query=query,
                query_embedding=query_embedding,
                answer=answer,
                ui_answer=ui_answer or "",
                tenant_id=tenant_id,
                intent=intent,
                entities=entities,
                kb_version=kb_version,
                embedding_model=cache_settings.EMBEDDING_MODEL,
                llm_model=llm_model,
                prompt_version=cache_settings.PROMPT_VERSION,
                created_at=now.isoformat(),
                expires_at=expires_at.isoformat(),
                last_accessed=now.isoformat(),
                access_count=1,
                source_type="LLM"
            )

            point = PointStruct(
                id=point_id,
                vector=query_embedding,
                payload=payload.model_dump()
            )

            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.collection_name,
                points=[point]
            )

            metrics_service.record_insertion()
            logger.info("Stored new semantic cache entry with ID: %s for tenant: %s", point_id, tenant_id)
            return point_id

        except Exception as e:
            logger.error("Error storing cache entry: %s", e)
            return None

    async def update_access(self, point_id: str, entry: CacheEntryPayload):
        """
        Updates the last_accessed timestamp and access_count of a cache entry.
        """
        try:
            entry.last_accessed = datetime.now(timezone.utc).isoformat()
            entry.access_count += 1
            
            await asyncio.to_thread(
                self.client.set_payload,
                collection_name=self.collection_name,
                payload={
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count
                },
                points=[point_id]
            )
        except Exception as e:
            logger.warning("Failed to update access metadata for cache entry %s: %s", point_id, e)

    async def delete_entry(self, point_id: str):
        """
        Deletes a cache entry by its Qdrant point ID.
        """
        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[point_id])
            )
            metrics_service.record_deletion()
        except Exception as e:
            logger.error("Failed to delete cache entry %s: %s", point_id, e)

    # Concurrency: Request Coalescing helper methods
    async def get_or_create_inflight_event(self, tenant_id: str, normalized_query: str) -> Tuple[asyncio.Event, bool]:
        """
        Returns (event, is_creator). If is_creator is True, the caller is responsible for invoking RAG
        and later setting the event. If False, the caller should wait on the event.
        """
        key = (tenant_id, normalized_query)
        async with self._inflight_lock:
            if key in self._inflight_events:
                return self._inflight_events[key], False
            else:
                event = asyncio.Event()
                self._inflight_events[key] = event
                return event, True

    async def release_inflight_event(self, tenant_id: str, normalized_query: str):
        key = (tenant_id, normalized_query)
        async with self._inflight_lock:
            if key in self._inflight_events:
                event = self._inflight_events.pop(key)
                event.set()
