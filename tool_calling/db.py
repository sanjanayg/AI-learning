import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker,Session

load_dotenv()
DATABASE_URL = os.getenv("DB_URL")


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_database_schema(db: Session):
    result = db.execute(
        text("""
        SELECT
            table_name,
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
        """)
    )
    rows = result.fetchall()
    schema = {}

    for table_name, column_name, data_type in rows:
        if table_name not in schema:
            schema[table_name] = []

        schema[table_name].append(
            f"{column_name} {data_type}"
        )

    schema_text = ""

    for table, columns in schema.items():
        schema_text += f"\n{table}(\n"
        schema_text += ",\n".join(
            [f"  {col}" for col in columns]
        )
        schema_text += "\n)\n"
    return schema_text