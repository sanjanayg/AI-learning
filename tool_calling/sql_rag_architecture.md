# Architecture: Text-to-SQL & Multi-Domain Agent

This document outlines the professional,architecture to solve the challenges of **multi-domain query routing**, **schema retrieval (Schema RAG)**, and **column-value mapping** (e.g., matching "active" to `'Y'`/`'N'`).

---

## 1. High-Level Architectural Flow

```text
               [ User Input ]
                      │
                      ▼
        [ Unified MCP Orchestrator ]
         (Groq / Llama 3.3 / GPT-4)
                      │
      ┌───────────────┼───────────────┐
      ▼               ▼               ▼
[ Weather Tool ]  [ Email Tool ]  [ DB Query Agent ]
                                      │
                                      ▼
                        [ Entity & Value Resolution ]
                        - Match "active" -> active = 'Y'
                        - Match "BSOL" -> client_name = 'BSOL Shipping'
                                      │
                                      ▼
                           [ Dynamic Schema RAG ]
                           - Fetch relevant tables from Vector DB
                                      │
                                      ▼
                           [ SQL Gen & Validation ]
                           - Generate PostgreSQL query
                           - Parse SQL to block non-SELECT
                                      │
                                      ▼
                              [ execute query ]
```

---

## 2. Multi-Domain Routing (Weather, Email, SQL)

### The Production Approach: Unified Tool Selection
In production, a hardcoded "intent classifier" step is fragile and does not allow for agentic reasoning (e.g., retrieving SQL data and then emailing it). 

Instead, rely on the **Model Context Protocol (MCP)** and **Native LLM Tool Calling** to act as a dynamic routing system:
1. Expose each capability as a distinct tool schema to the LLM (e.g., `get_weather`, `run_query`, `send_email`).
2. Give the orchestrator LLM a system prompt that enforces strict guidelines on when and how to call these tools.
3. The LLM acts as the router naturally, determining if a question is about weather, database records, or sending an email.

---

## 3. Schema RAG (Using Qdrant/Vector DB)

For databases with **more than 20 tables**, sending the full schema in the prompt exceeds the context window and reduces SQL generation accuracy.

### Vector DB Schema Design
Instead of raw SQL statements, store structured **Table Metadata Blocks** in Qdrant.

#### Vector Payload structure:
```json
{
  "table_name": "vessel",
  "description": "Contains metadata for active/inactive marine vessels, vessel classification, and registry details.",
  "columns": [
    {"name": "vessel_id", "type": "INTEGER", "description": "Unique identifier of the vessel"},
    {"name": "vessel_name", "type": "VARCHAR", "description": "Official name of the vessel"},
    {"name": "vessel_type", "type": "VARCHAR", "description": "Type of cargo or design: e.g. 'oil tanker', 'container ship'"},
    {"name": "active", "type": "CHAR(1)", "description": "Status code: 'Y' for active, 'N' for inactive"}
  ],
  "relationships": [
    "vessel.vessel_id maps to client_vessel_mapping.vessel_id"
  ]
}
```

### Retrieval & Synthesis Algorithm
1. **Search**: Embedded User Query is compared against the vector database (cosine similarity on the table description + columns metadata).
2. **Filtering**: Retrieve the top $K$ tables (typically 3 to 5 tables).
3. **Prompt Injection**: Construct the subset of DDL and column descriptions matching only those $K$ tables and pass them to the LLM.

---

## 4. Solving the Column Value Mapping Problem
*(How does the LLM know "active" means `'Y'`/`'N'` or `'active'`/`'inactive'`?)*

To achieve production-grade SQL generation (accuracy > 95%), you must use a layered approach:

### Layer A: Rich Schema Metadata (The PostgreSQL way)
Rather than maintaining external spreadsheets, store column definitions directly in PostgreSQL comments. 
```sql
COMMENT ON COLUMN vessel.active IS 'Vessel active status. Valid values: Y = Active, N = Inactive';
COMMENT ON COLUMN client.client_country IS 'Origin country of the client company (e.g. India, USA, Singapore)';
```

Modify the schema discovery query in `db.py` to retrieve these comments:
```python
def get_database_schema(db: Session):
    query = """
        SELECT 
            c.table_name, 
            c.column_name, 
            c.data_type,
            pg_catalog.col_description(t.oid, c.ordinal_position) AS column_description
        FROM information_schema.columns c
        JOIN pg_class t ON t.relname = c.table_name
        WHERE c.table_schema = 'public'
        ORDER BY c.table_name, c.ordinal_position;
    """
    result = db.execute(text(query))
    # Build schema format text including comments for the LLM
```

### Layer B: Entity Resolution / Fuzzy Matching (For dynamic search values)
If the user asks: *"List clients from Inda"* or *"vessels owned by BSOL"*
1. **Entity Extraction**: Use a lightweight regex, NER model, or semantic lookup to pull key nouns ("Inda", "BSOL").
2. **Value Search**: Perform a fast lookup in a pre-computed dictionary or a vector index containing distinct column values.
3. **Resolution**:
   - "Inda" maps to `client_country = 'India'`
   - "BSOL" maps to `client_name = 'BSOL Shipping'`
4. **Context Injection**: Provide the resolved mapping parameters to the SQL generator prompt:
   ```text
   Resolved entities from user question:
   - client_country should match 'India'
   - client_name should match 'BSOL Shipping'
   ```

### Layer C: Semantic Few-Shot Retrieval (RAG for SQL patterns)
Store a historical database of **(User Question, Gold Standard SQL)** inside your Vector DB (e.g., Qdrant).
- When a user asks: *"Show inactive container ships"*
- Vector search retrieves similar past queries:
  - *Question*: "List all inactive cargo ships" 
  - *SQL*: `SELECT * FROM vessel WHERE active = 'N' AND vessel_type ILIKE 'cargo%'`
- Injecting this as a 1-shot example in the generator prompt guarantees the LLM copies the `active = 'N'` syntax.

---

## 5. Summary of Recommended Production Architecture

| Challenge | Quick Fix (POC level) | Production Level Solution (Recommended) |
| :--- | :--- | :--- |
| **Query Routing** | Prompt classification (`question_intent`) | **MCP Tool Calling & Agentic ReAct Loops** |
| **Schema Ingestion** | Full database dump to prompt | **Schema RAG (Qdrant)** - indexing table-level descriptions & relations |
| **Value Matching** | Inline prompt rules (hardcoded) | **DB Column Comments + Distinct Value Fuzzy Index + SQL Few-Shot RAG** |
| **Security** | Simple string keyword block | **SQL Parsing (e.g., using `sqlglot`), Read-Only User Roles, DB Row-Level Security (RLS)** |
