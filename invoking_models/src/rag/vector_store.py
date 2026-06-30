import uuid
import logging
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny,
)
from config import settings
from schemas import DocumentChunk

logger = logging.getLogger(__name__)


class QdrantStore:
    _client = None

    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.client = self._get_client()
        self._ensure_collection_exists()

    @classmethod
    def _get_client(cls) -> QdrantClient:
        """
        Singleton pattern for the Qdrant client.
        Connects to a remote Qdrant service if QDRANT_URL is set, 
        otherwise falls back to a persistent local disk path (qdrant_storage).
        """
        if cls._client is None:
            if settings.QDRANT_URL:
                logger.info("Connecting to remote Qdrant instance: %s", settings.QDRANT_URL)
                cls._client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                )
            else:
                logger.info("Initializing persistent local Qdrant storage at: %s", settings.QDRANT_PATH)
                cls._client = QdrantClient(
                    path=settings.QDRANT_PATH
                )
        return cls._client

    def _ensure_collection_exists(self):
        """
        Checks for collection existence. Initializes it if missing, configuring Cosine similarity 
        and programmatically building payload indexes to guarantee fast and secure multi-tenant isolation.
        """
        try:
            # check if collection exists
            exists = self.client.collection_exists(self.collection_name)
            if not exists:
                logger.info("Creating collection '%s' in Qdrant...", self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # matches SentenceTransformer all-MiniLM-L6-v2 vector dimension
                        distance=Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=16,
                        ef_construct=100,
                    )
                )
                
                # Programmatically build index on chat_id payload key (critical for isolation performance)
                logger.info("Creating KEYWORD index on 'chat_id' for multi-tenant isolation")
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="chat_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                # Programmatically build index on file_id payload key (for file-level tracking/deletion)
                logger.info("Creating KEYWORD index on 'file_id'")
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="file_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                # KEYWORD index on extracted_ids (array field) — enables MatchAny
                # scroll queries for exact-ID lookup without a full collection scan.
                logger.info("Creating KEYWORD index on 'extracted_ids' for exact-ID lookup")
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="extracted_ids",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                # ── Smoke-test: verify extracted_ids index is queryable in file-mode Qdrant ──
                # File-mode SQLite backend can behave differently from server mode.
                # This explicit count confirms the index accepts MatchAny filters
                # before any real data is written.
                try:
                    _test_filter = Filter(
                        must=[
                            FieldCondition(
                                key="extracted_ids",
                                match=MatchAny(any=["__smoke_test__"])
                            )
                        ]
                    )
                    _count = self.client.count(
                        collection_name=self.collection_name,
                        count_filter=_test_filter,
                        exact=True,
                    )
                    logger.info(
                        "extracted_ids index smoke-test passed (count=%d, exact=True)",
                        _count.count
                    )
                except Exception as smoke_exc:
                    logger.warning(
                        "extracted_ids index smoke-test failed — file-mode Qdrant may not "
                        "support MatchAny on this version. Exact-ID lookup will fall back "
                        "to hybrid-only. Error: %s",
                        smoke_exc
                    )
        except Exception as exc:
            logger.exception("Failed to ensure/create Qdrant collection: %s", self.collection_name)
            raise RuntimeError(f"Qdrant collection setup failed: {str(exc)}") from exc

    async def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]):
        """
        Upserts a batch of document chunks and their dense embeddings.
        Uses deterministic UUIDs derived from chat_id, file_id, and chunk_id 
        to prevent duplicate entries if the same file is uploaded multiple times.
        """
        if not chunks:
            return
        
        points = []
        namespace = uuid.uuid5(uuid.NAMESPACE_DNS, "multitenantrag.pipeline")

        for chunk, embedding in zip(chunks, embeddings):
            # Deterministic UUID generation to handle duplicates safely (upsert/idempotency)
            point_id = str(uuid.uuid5(namespace, f"{chunk.chat_id}_{chunk.file_id}_{chunk.chunk_id}"))
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=chunk.model_dump()
                )
            )

        # Sync Qdrant client call; wrapping it in asyncio.to_thread is possible, but QdrantClient
        # performs lightweight networking/IPC so direct call is fine, or we can use thread pool
        import asyncio
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points
        )
        logger.info("Upserted %d chunks into Qdrant collection '%s'", len(points), self.collection_name)

    def search_chunks(self, chat_id: str, query_vector: list[float], limit: int = 5) -> list[DocumentChunk]:
        """
        Performs vector similarity search.
        Enforces a strict mathematical partition wall using Qdrant payload filtering on 'chat_id'.
        Uses the modern Qdrant query_points API.
        """
        tenant_filter = Filter(
            must=[
                FieldCondition(
                    key="chat_id",
                    match=MatchValue(value=chat_id)
                )
            ]
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=tenant_filter,
            limit=limit
        )

        chunks = []
        for hit in response.points:
            payload = hit.payload
            chunks.append(DocumentChunk(**payload))

        return chunks

    def exact_id_lookup(
        self,
        chat_id: str,
        ids: list[str],
        limit: int | None = None,
    ) -> list[DocumentChunk]:
        """
        Scroll-based exact match on extracted_ids using MatchAny, always
        combined with a strict chat_id filter for multi-tenant isolation.

        Uses client.scroll() (not query_points) because we want deterministic
        payload retrieval, not vector-scored ranking.  Scroll is the correct
        Qdrant primitive for payload-filter-only lookups.

        Cross-tenant isolation guarantee: the chat_id FieldCondition in `must`
        ensures a point from another tenant can never satisfy the filter even
        if its extracted_ids happen to contain the queried ID.

        Args:
            chat_id: Partition key — only points belonging to this tenant are returned.
            ids:     List of ID strings to match against extracted_ids (MatchAny semantics).
            limit:   Max points to return. Defaults to settings.MAX_EXACT_MATCHES.

        Returns:
            List of DocumentChunk objects, empty list if no matches.
        """
        if not ids:
            return []

        effective_limit = limit if limit is not None else settings.MAX_EXACT_MATCHES

        exact_filter = Filter(
            must=[
                FieldCondition(
                    key="chat_id",
                    match=MatchValue(value=chat_id),
                ),
                FieldCondition(
                    key="extracted_ids",
                    match=MatchAny(any=ids),
                ),
            ]
        )

        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=exact_filter,
            limit=effective_limit,
            with_payload=True,
            with_vectors=False,
        )

        points = scroll_result[0]  # (points_list, next_page_offset)
        chunks: list[DocumentChunk] = []
        for pt in points:
            if pt.payload:
                try:
                    chunks.append(DocumentChunk(**pt.payload))
                except Exception as exc:
                    logger.warning(
                        "exact_id_lookup: skipped malformed payload for point %s: %s",
                        pt.id, exc
                    )

        logger.info(
            "exact_id_lookup: chat_id=%s ids=%r → %d match(es)",
            chat_id, ids, len(chunks)
        )
        return chunks
