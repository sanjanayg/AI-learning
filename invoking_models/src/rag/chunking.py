from langchain_text_splitters import TokenTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class TokenChunkingService:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)


class SemanticChunker:
    def __init__(self, similarity_threshold: float = 0.65, min_chunk_sentences: int = 2):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_text(self, text: str) -> list[str]:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return [text]

        embeddings = self.model.encode(sentences)
        chunks, current = [], [sentences[0]]

        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i - 1], embeddings[i]) / (
                np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8
            )
            if sim < self.similarity_threshold and len(current) >= self.min_chunk_sentences:
                chunks.append(". ".join(current) + ".")
                current = []
            current.append(sentences[i])

        if current:
            chunks.append(". ".join(current) + ".")

        return chunks