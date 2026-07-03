"""
RAG retrieval orchestration — hybrid search.

Flow:
  1. extract_query_ids(raw_query)           → list[str]   (regex, microseconds)
  2. exact_id_lookup(chat_id, ids)          → list[DocumentChunk]  (if IDs found)
  3. keyword_search_chunks(chat_id, query)  → list[DocumentChunk]  (BM25/TF-IDF)
  4. dense vector search(chat_id, vector)   → list[DocumentChunk]  (always runs)
  5. reciprocal_rank_fusion(exact, bm25, vector) → scored + deduplicated
  6. truncate to top_k

Design invariants:
  - ID extraction always uses the RAW user query (pre-rewrite).
  - Dense search always uses the rewritten standalone query embedding.
  - BM25 uses the rewritten query for better term matching.
  - Exact matches receive a weight multiplier (EXACT_WEIGHT) before RRF.
  - chat_id isolation is enforced at every retrieval path.
  - Fallback: embedding failure → exact + BM25; BM25 failure → exact + vector.
"""

from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantStore
from rag.id_extractor import extract_query_ids
from config import settings
from schemas import DocumentChunk

logger = logging.getLogger(__name__)


# ── RRF helpers ───────────────────────────────────────────────────────────────

class _RankedList(NamedTuple):
    chunks: list[DocumentChunk]
    weight: float  # per-path multiplier applied to RRF score


def reciprocal_rank_fusion(
    ranked_lists: list[_RankedList],
    k: int = 60,
) -> list[DocumentChunk]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.

    RRF score for chunk c:
        score(c) = sum over lists L of: weight_L / (k + rank_L(c))
    where rank is 1-based.  Chunks absent from a list contribute 0.

    Deduplication is by chunk_id — first occurrence (highest-scoring list)
    wins for the DocumentChunk object returned.

    Args:
        ranked_lists: Each entry pairs a ranked chunk list with a weight multiplier.
        k:            RRF smoothing constant (default 60, per the original paper).

    Returns:
        Deduplicated list sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, DocumentChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked.chunks, start=1):
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + ranked.weight / (k + rank)
            chunks_by_id.setdefault(cid, chunk)

    return [
        chunks_by_id[cid]
        for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


def merge_exact_and_hybrid_results(
    exact_chunks: list[DocumentChunk],
    fused_chunks: list[DocumentChunk],
    top_k: int,
    max_exact: int,
) -> list[DocumentChunk]:
    """
    Place exact-ID matches first (capped at max_exact), then fill with RRF-fused
    results, deduplicating by chunk_id throughout.
    """
    seen: set[str] = set()
    merged: list[DocumentChunk] = []

    for chunk in exact_chunks[:max_exact]:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            merged.append(chunk)

    for chunk in fused_chunks:
        if len(merged) >= top_k:
            break
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            merged.append(chunk)

    return merged[:top_k]


# ── Retriever ─────────────────────────────────────────────────────────────────

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
        Full hybrid retrieval: exact-ID + BM25 + dense vector → RRF fusion.

        Args:
            chat_id:   Tenant partition key.
            query:     LLM-rewritten standalone query (used for embedding + BM25).
            limit:     Total chunks to return (top_k budget).
            raw_query: Original user query before rewriting. ID extraction runs on
                       this value so literal typed IDs are never altered. Falls back
                       to `query` if not provided.

        Returns:
            Merged, deduplicated list of DocumentChunks up to `limit`.
        """
        logger.info(
            "RAGRetriever.retrieve_relevant_chunks: chat_id=%s limit=%d", chat_id, limit
        )

        # ── Step 1: ID detection (always on raw query) ────────────────────────
        id_source = raw_query if raw_query is not None else query
        query_ids = extract_query_ids(id_source)

        # ── Step 2: Exact-ID lookup ───────────────────────────────────────────
        exact_chunks: list[DocumentChunk] = []
        if query_ids:
            logger.info("Exact-ID path: %d ID(s) detected %r", len(query_ids), query_ids)
            exact_chunks = await asyncio.to_thread(
                self.vector_store.exact_id_lookup,
                chat_id=chat_id,
                ids=query_ids,
                limit=settings.MAX_EXACT_MATCHES,
            )
            logger.info("Exact-ID lookup: %d chunk(s) returned", len(exact_chunks))
        else:
            logger.debug("No IDs detected — exact-ID path skipped.")

        # ── Step 3: BM25 keyword search ───────────────────────────────────────
        bm25_chunks: list[DocumentChunk] = []
        try:
            bm25_chunks = await asyncio.to_thread(
                self.vector_store.keyword_search_chunks,
                chat_id=chat_id,
                query=query,
                limit=limit,
            )
            logger.info("BM25 search: %d chunk(s) returned", len(bm25_chunks))
        except Exception as exc:
            logger.warning("BM25 search failed, continuing without it: %s", exc)

        # ── Step 4: Dense vector search ───────────────────────────────────────
        vector_chunks: list[DocumentChunk] = []
        try:
            query_vector = await EmbeddingService.embed_query(query)
            if not query_vector:
                raise ValueError("Empty embedding returned")
            vector_chunks = await asyncio.to_thread(
                self.vector_store.search_chunks,
                chat_id=chat_id,
                query_vector=query_vector,
                limit=limit,
            )
            logger.info("Dense search: %d chunk(s) returned", len(vector_chunks))
        except Exception as exc:
            logger.warning("Dense vector search failed, continuing without it: %s", exc)
            if not bm25_chunks:
                # Both BM25 and vector failed — return whatever exact matches exist
                logger.error("All search paths failed; returning exact matches only.")
                return exact_chunks[:limit]

        # ── Step 5: RRF fusion (BM25 + vector; exact handled separately) ─────
        ranked_lists: list[_RankedList] = []
        if bm25_chunks:
            ranked_lists.append(_RankedList(bm25_chunks, settings.BM25_WEIGHT))
        if vector_chunks:
            ranked_lists.append(_RankedList(vector_chunks, settings.VECTOR_WEIGHT))
        fused = reciprocal_rank_fusion(ranked_lists, k=settings.RRF_K) if ranked_lists else []
        logger.info("RRF fusion: %d unique chunk(s) after fusion", len(fused))

        # ── Step 6: Merge exact-first, fill with fused, deduplicate ──────────
        merged = merge_exact_and_hybrid_results(
            exact_chunks=exact_chunks,
            fused_chunks=fused,
            top_k=limit,
            max_exact=settings.MAX_EXACT_MATCHES,
        )

        logger.info(
            "Retrieval complete: exact=%d bm25=%d vector=%d fused=%d merged=%d (top_k=%d) chat_id=%s",
            len(exact_chunks), len(bm25_chunks), len(vector_chunks),
            len(fused), len(merged), limit, chat_id,
        )
        return merged
