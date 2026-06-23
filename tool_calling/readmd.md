# SQL Tool Calling Agent POC

## Overview

This project demonstrates how an LLM can interact with a PostgreSQL database using a SQL Tool Calling approach.

The application receives a natural language question, generates a PostgreSQL query using an LLM, executes the query against the database, and returns a human-readable answer.

### Architecture

```text
User Question
      ↓
FastAPI
      ↓
Groq LLM
      ↓
Generate SQL Query
      ↓
SQL Validation Layer
      ↓
PostgreSQL
      ↓
Query Result
      ↓
Groq LLM
      ↓
Natural Language Response
```

---

## Features

* FastAPI REST API
* PostgreSQL Integration
* Dynamic Schema Discovery
* SQL Query Generation using LLM
* SQL Validation
* Natural Language Response Generation
* Production-Oriented Architecture
* Extensible for Agentic AI Workflows

---

## Project Structure

```text
sql_tool_project/
│
├── .env
├── main.py
├── db.py
├── sql_tool.py
├── llm_service.py
├── setup_db.py
├── requirements.txt
└── README.md
```

---

## Technology Stack

### Backend

* FastAPI
* Python 3.11+

### Database

* PostgreSQL

### LLM

* Groq
* Model: llama-3.3-70b-versatile

### Database Driver

* psycopg2-binary

---

## Installation

### Clone Repository

```bash
git clone <repository-url>

cd sql_tool_project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_groq_api_key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
```

---

## Database Setup

### Create Sample Tables

Run:

```bash
python setup_db.py
```

This creates:

* client
* vessel
* client_vessel_mapping

---

## Dynamic Schema Discovery

Instead of hardcoding table definitions, the application retrieves schema information dynamically from PostgreSQL.

Example query:

```sql
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public';
```

Benefits:

* No hardcoded schema
* Automatically adapts to schema changes
* Production-friendly

---

## Running the Application

Start FastAPI:

```bash
uvicorn main:app --reload
```

Application URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### Ask Question

Endpoint:

```http
POST /ask
```

Request:

```json
{
  "question": "List all oil tankers currently active."
}
```

Example Response:

```json
{
  "question": "List all oil tankers currently active.",
  "generated_sql": "SELECT vessel_id, vessel_name FROM vessel WHERE vessel_type ILIKE 'oil tanker' AND active = 'Y'",
  "sql_result": [
    {
      "vessel_id": 3,
      "vessel_name": "MV Horizon"
    }
  ],
  "answer": "There is currently one active oil tanker: MV Horizon."
}
```

---

## SQL Safety

The system blocks dangerous operations.

Blocked Statements:

```sql
DROP
DELETE
UPDATE
INSERT
ALTER
TRUNCATE
CREATE
```

Only SELECT statements are allowed.

Example validation:

```python
if not query.upper().startswith("SELECT"):
    raise ValueError("Only SELECT queries are allowed")
```

---

## Sample Questions

### Vessel Questions

```text
List all active vessels.

Show all oil tankers currently active.

List all container ships.

Show vessels with GRT greater than 70000.
```

### Client Questions

```text
Show all clients.

How many clients are active?

List clients from India.
```

### Relationship Questions

```text
Show all vessels assigned to BSOL Shipping.

How many active vessels does each client have?

Which client owns the most active vessels?

List inactive vessel-client mappings.
```

---

## Production Improvements

Potential future enhancements:

### Agentic AI

```text
FastAPI
   ↓
Agent
   ↓
SQL Tool
   ↓
PostgreSQL
```

### RAG Integration

```text
FastAPI
   ↓
Agent
   ↓
SQL Tool
   ↓
PostgreSQL

Vector Tool
   ↓
FAISS / pgvector
```

### Additional Security

* Row-Level Security
* Query Cost Analysis
* Query Timeout Controls
* Rate Limiting
* Role-Based Access Control (RBAC)
* Human Approval for Sensitive Queries

---

## Learning Objectives

This project demonstrates:

* FastAPI Development
* PostgreSQL Integration
* SQL Tool Calling
* LLM Integration
* Dynamic Schema Discovery
* Prompt Engineering
* Query Validation
* Production-Oriented AI Architecture

---

## Future Scope

* Multi-table Query Support
* Agentic Workflows
* RAG Integration
* pgvector Support
* LangGraph Integration
* Streaming Responses
* Dashboard UI

---

## Author

Sanjana YG

BE - Artificial Intelligence & Machine Learning

Python Developer | FastAPI | PostgreSQL | LLM Applications

TO RUN- uvicorn main:app --reload

Frontend= chainlit run app.py --port 8001