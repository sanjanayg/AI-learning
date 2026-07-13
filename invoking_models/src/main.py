from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import (
    extraction_router,
    chunking_router,
    chat_router,
    chats_router,
    cache_router,
    get_cache_service,
)
from db.database import init_db
from cache.cache_cleanup import CacheCleanupService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    On startup: initialises the PostgreSQL schema (CREATE TABLE IF NOT EXISTS) and starts the cache cleanup service.
    On shutdown: stops the cache cleanup service.
    """
    await init_db()
    
    cleanup_service = CacheCleanupService(get_cache_service())
    cleanup_service.start()
    
    yield
    
    cleanup_service.stop()


app = FastAPI(
    title="Image and PDF Text Extraction API",
    description="Production-grade document text parsing and LLM OCR orchestrator.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extraction_router)
app.include_router(chunking_router)
app.include_router(chat_router)
app.include_router(chats_router)
app.include_router(cache_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)