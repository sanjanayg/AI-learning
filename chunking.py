import re
from typing import List
import tiktoken

import numpy as np
from sentence_transformers import SentenceTransformer

encoding = tiktoken.get_encoding("cl100k_base")

class SemanticChunker:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.65,
        min_chunk_sentences: int = 2,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences

    def split_into_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    
    def chunk_text(self, text: str) -> List[str]:
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        embeddings = self.model.encode(sentences)

        print("the embedding are",len(embeddings))
        chunks = []
        current_chunk = [sentences[0]]

        for index in range(1, len(sentences)):
            previous_embedding = embeddings[index - 1]
            current_embedding = embeddings[index]

            similarity = self.cosine_similarity(
                previous_embedding,
                current_embedding
            )

            should_split = (
                similarity < self.similarity_threshold
                and len(current_chunk) >= self.min_chunk_sentences
            )

            if should_split:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

            current_chunk.append(sentences[index])

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


if __name__ == "__main__":
    sample_text = """
    Artificial Intelligence is changing many industries. 
    Machine learning is a major branch of AI. 
    Deep learning uses neural networks to learn patterns from data.

    Pizza is a popular Italian food. 
    It is made with dough, cheese, and toppings. 
    Many people like pizza for dinner.

    Vector databases are useful in RAG applications. 
    They store embeddings and help retrieve similar text. 
    FAISS is one popular vector search library.
    """

    chunker = SemanticChunker(
        similarity_threshold=0.65,
        min_chunk_sentences=2
    )

    chunks = chunker.chunk_text(sample_text)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        token_count = len(encoding.encode(chunk))
        print(f"Tokens: {token_count}")

        print(chunk)