import asyncio
import logging
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantStore
from schemas import DocumentChunk

logger = logging.getLogger(__name__)


class RAGRetriever:
    def __init__(self):
        self.vector_store = QdrantStore()

    async def retrieve_relevant_chunks(
        self, 
        chat_id: str, 
        query: str, 
        limit: int = 5
    ) -> list[DocumentChunk]:
        """
        Generates an embedding for the search query, then queries Qdrant
        strictly filtering the search candidate set by the chat_id payload value.
        Runs the search in a thread pool to avoid blocking the event loop.
        """
        logger.info("Retrieving chunks for chat_id=%s, query=%r", chat_id, query)
        
        # 1. Embed the query (async, thread-pooled)
        query_vector = await EmbeddingService.embed_query(query)
        if not query_vector:
            logger.warning("Empty query embedding generated for query: %r", query)
            return []

        # 2. Query Qdrant with strict tenant filtering (async, thread-pooled)
        chunks = await asyncio.to_thread(
            self.vector_store.search_chunks,
            chat_id=chat_id,
            query_vector=query_vector,
            limit=limit
        )

        logger.info("Retrieved %d relevant chunks for chat_id=%s", len(chunks), chat_id)
        return chunks
