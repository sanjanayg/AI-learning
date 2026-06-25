from qdrant_client import QdrantClient
# adjust import path to match your QdrantStore setup
from src.rag.vector_store import QdrantStore

store = QdrantStore()
response = store.client.scroll(
    collection_name=store.collection_name,
    limit=10,
    with_payload=True,
    with_vectors=False
)
points = response[0]
for p in points:
    print(p.id, p.payload)