from sqlalchemy import text
from sqlalchemy.orm import Session


BLOCKED_KEYWORDS = [
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "TRUNCATE"
]


def validate_sql(query: str):
    upper_query = query.upper().strip()

    if not upper_query.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    for keyword in BLOCKED_KEYWORDS:
        if keyword in upper_query:
            raise ValueError(f"Unsafe SQL blocked: {keyword}")


def run_sql_tool(query: str, db: Session):
    validate_sql(query)

    result = db.execute(text(query))

    rows = result.mappings().all()

    return [dict(row) for row in rows]