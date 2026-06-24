from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.endpoints import extraction_router, chunking_router, chat_router, chats_router
from db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    On startup: initialises the PostgreSQL schema (CREATE TABLE IF NOT EXISTS).
    On shutdown: nothing extra needed — SQLAlchemy pool disposes automatically.
    """
    await init_db()
    yield


app = FastAPI(
    title="Image and PDF Text Extraction API",
    description="Production-grade document text parsing and LLM OCR orchestrator.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(extraction_router)
app.include_router(chunking_router)
app.include_router(chat_router)
app.include_router(chats_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)