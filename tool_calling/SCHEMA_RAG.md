# Schema-Oriented RAG with Qdrant

Instead of dumping the entire database schema into every LLM prompt, this system
stores each table's metadata as a vector embedding in Qdrant and retrieves only
the tables relevant to the user's question at query time.

---

## Why

A full schema prompt has two problems:

- **Token waste** — every table gets sent even if the question only touches one or two
- **Enum ambiguity** — the LLM doesn't know if `status` holds `Y/N`, `0/1`, `active/inactive`, etc., so it guesses wrong

This RAG approach solves both by storing rich per-table documents (with sampled enum values)
and retrieving only what's needed per question.

---

## How It Works

```
Server Startup
└── schema_indexer.index_schema()
      ├── Connect to PostgreSQL
      ├── Fetch all tables + columns + data types from information_schema
      ├── For enum-like columns (status, active, type, flag, state, enabled, kind)
      │     └── SELECT DISTINCT → captures actual values: "Y","N" / "0","1" / "active","inactive"
      ├── Build one text document per table
      ├── Embed each document with all-MiniLM-L6-v2 (384-dim)
      └── Store in Qdrant local collection "schema_index"
            └── Skips if collection already exists (no re-indexing on every restart)

Per Request (/mcp-questions)
└── schema_retriever.retrieve_relevant_schema(question)
      ├── Embed the user question with the same model
      ├── Cosine similarity search in Qdrant → top 5 matching tables
      └── Return compact schema string → injected into LLM system prompt
```

---

## Files

| File | Responsibility |
|---|---|
| `schema_indexer.py` | Connects to Postgres, builds table documents, embeds and stores in Qdrant |
| `schema_retriever.py` | Embeds user question, queries Qdrant, returns relevant schema string |
| `qdrant_data/` | Local Qdrant storage folder (auto-created, git-ignored) |

---

## Table Document Format

Each table is stored as a text document like this:

```
Table: users
Columns:
  - id (integer)
  - name (character varying)
  - status (character varying) — sample values: Y, N
  - created_at (timestamp without time zone)
```

The `— sample values` part is what lets the LLM know that `status = 'Y'` means active,
not `status = 'active'` or `status = 1`.

---

## Enum Column Detection

Columns are automatically sampled if their name contains any of these hints:

```
status, active, type, flag, state, enabled, is_, kind
```

Up to 10 distinct values are fetched per column. To add more hints, update
`ENUM_HINTS` in `schema_indexer.py`.

---

## Configuration

All constants are at the top of each file:

| Constant | File | Default | Description |
|---|---|---|---|
| `COLLECTION_NAME` | both | `schema_index` | Qdrant collection name |
| `EMBED_MODEL` | both | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K` | `schema_retriever.py` | `5` | Number of tables returned per query |
| `MAX_DISTINCT` | `schema_indexer.py` | `10` | Max enum sample values per column |
| `ENUM_HINTS` | `schema_indexer.py` | see above | Column name fragments that trigger sampling |

---

## Re-indexing After Schema Changes

The indexer skips if the collection already exists. If your database schema changes
(new tables, renamed columns, changed enum values), delete the local store and restart:

```powershell
# Windows
rmdir /s /q qdrant_data

# Then restart the server — it will re-index automatically
uvicorn main:app --reload
```

---

## Dependencies

```
qdrant-client
sentence-transformers
psycopg2-binary
```

All are included in `requirements.txt`.
