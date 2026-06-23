from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "schema_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5  # how many most-relevant tables to return

_model = None
_qdrant = None


def _get_clients():
    global _model, _qdrant
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _qdrant is None:
        _qdrant = QdrantClient(path="./qdrant_data")
    return _model, _qdrant


def retrieve_relevant_schema(question: str) -> str:
    """
    Embeds the question, retrieves the top-k most relevant table schemas
    from Qdrant, and returns them as a compact schema string for the prompt.
    """
    model, qdrant = _get_clients()

    vector = model.encode(question).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=TOP_K,
    ).points

    if not results:
        return "No relevant schema found."

    schema_parts = [r.payload["document"] for r in results]
    return "\n\n".join(schema_parts)
