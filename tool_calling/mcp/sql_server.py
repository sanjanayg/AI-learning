from mcp.server.fastmcp import FastMCP
import psycopg2
import psycopg2.extras
import sqlglot
import os
import json
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("PostgreSQL MCP Server")



MAX_ROWS = 500
STATEMENT_TIMEOUT_MS = 5000


def get_conn():
    # Pass the DSN URL string directly as a positional argument
    return psycopg2.connect(os.environ["DB_URL"], connect_timeout=5)


def is_select_only(sql: str) -> bool:
    """Allow exactly one SELECT statement, reject everything else."""
    try:
        parsed = sqlglot.parse(sql, read="postgres")
        if len(parsed) != 1:
            return False
        return parsed[0].key == "select"
    except Exception:
        return False


@mcp.tool()
def run_query(sql: str) -> str:
    """
    Execute a read-only SQL SELECT query against PostgreSQL and return the
    results as a JSON string. Only a single SELECT statement is permitted —
    INSERT, UPDATE, DELETE, DDL, and multi-statement queries are rejected.
    Results are capped at 500 rows and the query is aborted if it runs
    longer than 5 seconds.
    """
    if not sql or not sql.strip():
        return json.dumps({"error": "Empty query."})

    if not is_select_only(sql):
        return json.dumps({"error": "Only a single SELECT statement is allowed."})

    conn = None
    try:
        conn = get_conn()
        conn.set_session(readonly=True)  # second layer of defense beyond the DB role
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")
            cur.execute(sql)
            rows = cur.fetchmany(MAX_ROWS)
            return json.dumps(rows, default=str)
    except psycopg2.Error as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"})
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    mcp.run(transport="stdio")