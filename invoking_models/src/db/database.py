"""
Async SQLAlchemy engine, session factory, and FastAPI dependency.

Usage in endpoints:
    from db.database import get_db
    ...
    async def my_endpoint(db: AsyncSession = Depends(get_db)):
        ...

Startup:
    Call `await init_db()` from the FastAPI lifespan or startup event to
    auto-create all tables (CREATE TABLE IF NOT EXISTS).
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import settings
from db.models import Base

logger = logging.getLogger(__name__)

# ── Engine ───────────────────────────────────────────────────────────────────

def _build_engine():
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Add it to your .env file: "
            "postgresql+asyncpg://postgres:<password>@localhost:5432/postgres"
        )
    return create_async_engine(
        settings.DATABASE_URL,
        echo=False,          # Set True locally for SQL debug logging
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=True,  # Recycles stale connections automatically
    )


async_engine = _build_engine()

# ── Session factory ───────────────────────────────────────────────────────────

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevents lazy-load errors after commit in async context
    autocommit=False,
    autoflush=False,
)

# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields a scoped AsyncSession per request and guarantees cleanup.
    Use as a FastAPI Depends() parameter on any endpoint that needs DB access.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# ── Startup initialiser ───────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Auto-creates all tables defined in models.py if they don't already exist.
    Safe to call on every startup (CREATE TABLE IF NOT EXISTS semantics).
    """
    logger.info("Running DB schema initialisation (CREATE TABLE IF NOT EXISTS)...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("DB schema ready.")
