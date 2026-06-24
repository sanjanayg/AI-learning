import asyncio
from sentence_transformers import SentenceTransformer
from config import settings


class EmbeddingService:
    _model = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Lazy-loads the SentenceTransformer model to prevent blocking startup of other components.
        Returns a singleton instance.
        """
        if cls._model is None:
            cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return cls._model

    @classmethod
    async def embed_documents(cls, texts: list[str]) -> list[list[float]]:
        """
        Generates dense embeddings for a list of document chunks.
        Runs in a separate thread pool to prevent blocking the async event loop.
        """
        if not texts:
            return []
        model = cls.get_model()
        # model.encode is CPU-heavy; run in thread pool
        embeddings = await asyncio.to_thread(
            model.encode, 
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        return embeddings.tolist()

    @classmethod
    async def embed_query(cls, text: str) -> list[float]:
        """
        Generates a dense embedding for a single search query.
        """
        embeddings = await cls.embed_documents([text])
        return embeddings[0] if embeddings else []
