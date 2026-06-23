import os
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION_NAME = "schema_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
# Columns whose name suggests they hold enum-like values worth sampling
ENUM_HINTS = {"status", "active", "type", "flag", "state", "enabled", "is_", "kind", "deleted"}
MAX_DISTINCT = 10  # max sample values to fetch per enum column


def get_conn():
    return psycopg2.connect(os.getenv("DB_URL"), connect_timeout=5)


def _is_enum_column(col_name: str) -> bool:
    lower = col_name.lower()
    return any(hint in lower for hint in ENUM_HINTS)


def fetch_schema_with_samples() -> list[dict]:
    """
    Returns one dict per table:
    {
      "table": "users",
      "columns": [{"name": "status", "type": "character varying", "sample_values": ["Y","N"]}, ...]
    }
    """
    conn = get_conn()
    tables = []

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # All columns across all public tables
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
        rows = cur.fetchall()

        # Group by table
        table_map: dict[str, list] = {}
        for row in rows:
            t = row["table_name"]
            table_map.setdefault(t, []).append({
                "name": row["column_name"],
                "type": row["data_type"],
                "sample_values": []
            })

        # For enum-like columns, fetch distinct values
        for table, columns in table_map.items():
            for col in columns:
                if _is_enum_column(col["name"]):
                    try:
                        cur.execute(
                            f'SELECT DISTINCT "{col["name"]}" FROM "{table}" '
                            f'WHERE "{col["name"]}" IS NOT NULL LIMIT %s',
                            (MAX_DISTINCT,)
                        )
                        col["sample_values"] = [str(r[col["name"]]) for r in cur.fetchall()]
                    except Exception:
                        pass  # skip if column not queryable

            tables.append({"table": table, "columns": columns})

    conn.close()
    return tables


def build_table_document(table_info: dict) -> str:
    """
    Builds a rich text description of one table for embedding.
    Example:
      Table: users
      Columns:
        - id (integer)
        - status (character varying) — sample values: Y, N
        - name (text)
    """
    lines = [f"Table: {table_info['table']}", "Columns:"]
    for col in table_info["columns"]:
        line = f"  - {col['name']} ({col['type']})"
        if col["sample_values"]:
            line += f" — sample values: {', '.join(col['sample_values'])}"
        lines.append(line)
    return "\n".join(lines)


def index_schema():
    """Fetch schema, embed each table, upsert into Qdrant."""
    model = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(path="./qdrant_data")

    # Only index if collection doesn't exist yet
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        print("[schema_indexer] Schema already indexed, skipping.")
        return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    tables = fetch_schema_with_samples()
    points = []

    for idx, table_info in enumerate(tables):
        doc = build_table_document(table_info)
        vector = model.encode(doc).tolist()

        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={
                "table": table_info["table"],
                "document": doc,
                "columns_json": json.dumps(table_info["columns"])
            }
        ))

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[schema_indexer] Indexed {len(points)} tables into Qdrant.")


if __name__ == "__main__":
    index_schema()
