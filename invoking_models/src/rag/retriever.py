"""
RAG retrieval orchestration with exact-ID lookup + hybrid search merge.

Flow (always runs both paths):
  1. extract_query_ids(raw_query)           → list[str]  (sync, regex, microseconds)
  2. exact_id_lookup(chat_id, ids, MAX)     → list[DocumentChunk]  (if IDs found)
  3. dense vector search (always runs)      → list[DocumentChunk]
  4. merge: dedup by chunk_id, exact-match chunks first, capped at MAX_EXACT_MATCHES
  5. truncate merged list to overall top_k

Design invariants:
  - extract_query_ids uses the RAW user query, NOT the LLM-rewritten standalone query,
    so the literal ID the user typed is never changed before regex matching.
  - extract_query_ids is fully independent of validate_query / RAGGuardrails.
  - Hybrid search ALWAYS runs regardless of whether IDs were found.
  - Exact matches are capped at settings.MAX_EXACT_MATCHES to prevent false-positive
    flooding from crowding out hybrid results.
"""

from __future__ import annotations

import asyncio
import logging

from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantStore
from rag.id_extractor import extract_query_ids
from config import settings
from schemas import DocumentChunk

logger = logging.getLogger(__name__)


def _merge_results(
    exact_chunks: list[DocumentChunk],
    hybrid_chunks: list[DocumentChunk],
    top_k: int,
    max_exact: int,
) -> list[DocumentChunk]:
    """
    Merge exact-ID matches with hybrid search results.

    Rules:
      1. Exact matches are capped at `max_exact` before merge (false-positive guard).
      2. Exact chunks come first in the output ordering (precision priority).
      3. Hybrid chunks fill the remainder of the top_k budget.
      4. Deduplication is by chunk_id — if a chunk appears in both result sets,
         the exact-match copy is kept (it comes first and gets absorbed into `seen`).
      5. Final list is truncated to top_k.

    Args:
        exact_chunks:  Results from exact_id_lookup, already limited at scroll time.
        hybrid_chunks: Results from dense vector search.
        top_k:         Maximum total chunks to return to the caller.
        max_exact:     Cap on exact matches added to the merged list.

    Returns:
        Merged, deduplicated, top_k-truncated list.
    """
    seen: set[str] = set()
    merged: list[DocumentChunk] = []

    # Exact matches first — capped at max_exact
    for chunk in exact_chunks[:max_exact]:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            merged.append(chunk)

    # Hybrid fills remaining budget
    for chunk in hybrid_chunks:
        if len(merged) >= top_k:
            break
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            merged.append(chunk)

    return merged[:top_k]


class RAGRetriever:
    def __init__(self):
        self.vector_store = QdrantStore()

    async def retrieve_relevant_chunks(
        self,
        chat_id: str,
        query: str,
        limit: int = 5,
        raw_query: str | None = None,
    ) -> list[DocumentChunk]:
        """
        Full retrieval orchestration: exact-ID lookup merged with dense search.

        Args:
            chat_id:   Tenant partition key. Filters both exact and hybrid search.
            query:     Typically the LLM-rewritten standalone query (used for embedding).
            limit:     Total number of chunks to return (top_k budget).
            raw_query: The original user query BEFORE rewriting. If provided, ID
                       extraction runs on this value so the literal typed IDs are
                       preserved. Falls back to `query` if not provided.

        Returns:
            Merged list of DocumentChunks, exact-ID matches first, up to `limit` total.
        """
        logger.info(
            "RAGRetriever.retrieve_relevant_chunks: chat_id=%s, limit=%d", chat_id, limit
        )

        # ── Step 1: ID detection ─────────────────────────────────────────────
        # Always run on the RAW user query (pre-rewrite) so the literal ID
        # the user typed is never altered by query rewriting.
        id_source = raw_query if raw_query is not None else query
        query_ids = extract_query_ids(id_source)

        # ── Step 2: Exact-ID lookup (only if IDs detected) ───────────────────
        # Runs in a thread pool since QdrantClient is synchronous.
        exact_chunks: list[DocumentChunk] = []
        if query_ids:
            logger.info(
                "Exact-ID path triggered: %d ID(s) detected %r", len(query_ids), query_ids
            )
            exact_chunks = await asyncio.to_thread(
                self.vector_store.exact_id_lookup,
                chat_id=chat_id,
                ids=query_ids,
                limit=settings.MAX_EXACT_MATCHES,
            )
            logger.info(
                "Exact-ID lookup returned %d chunk(s) for chat_id=%s",
                len(exact_chunks), chat_id,
            )
        else:
            logger.debug("No IDs detected in query — exact-ID path skipped.")

        # ── Step 3: Dense vector search (ALWAYS runs) ────────────────────────
        # This is the existing hybrid search path. It always runs regardless of
        # whether exact-ID matches were found, ensuring hybrid results always
        # occupy the majority of the top_k budget.
        query_vector = await EmbeddingService.embed_query(query)
        if not query_vector:
            logger.warning("Empty query embedding for query: %r", query)
            # Graceful fallback: if embedding fails, return whatever exact matches exist
            return exact_chunks[:limit]

        hybrid_chunks = await asyncio.to_thread(
            self.vector_store.search_chunks,
            chat_id=chat_id,
            query_vector=query_vector,
            limit=limit,
        )
        logger.info(
            "Dense search returned %d chunk(s) for chat_id=%s",
            len(hybrid_chunks), chat_id,
        )

        # ── Step 4: Merge — exact first, hybrid fills remainder ───────────────
        merged = _merge_results(
            exact_chunks=exact_chunks,
            hybrid_chunks=hybrid_chunks,
            top_k=limit,
            max_exact=settings.MAX_EXACT_MATCHES,
        )

        logger.info(
            "Retrieval complete: exact=%d hybrid=%d merged=%d (top_k=%d) chat_id=%s",
            len(exact_chunks), len(hybrid_chunks), len(merged), limit, chat_id,
        )
        return merged
