"""
Async CRUD helpers for the three persistence tables.

All functions accept an AsyncSession injected via FastAPI's get_db() dependency.
None of these functions call session.commit() — that is handled by the get_db()
context manager so the entire request is atomic.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Chat, ChatFile, ChatMessage

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Chats ─────────────────────────────────────────────────────────────────────

async def list_chats(db: AsyncSession) -> Sequence[Chat]:
    """Return all chat sessions ordered by most recently active first."""
    result = await db.execute(
        select(Chat).order_by(Chat.last_active_at.desc())
    )
    return result.scalars().all()


async def create_chat(db: AsyncSession, chat_id: str) -> Chat:
    """Insert a brand-new chat row and return it."""
    now = _utcnow()
    chat = Chat(id=chat_id, created_at=now, last_active_at=now)
    db.add(chat)
    await db.flush()  # flush to get DB-generated defaults without committing
    logger.info("Created new chat session: %s", chat_id)
    return chat


async def upsert_chat(db: AsyncSession, chat_id: str) -> None:
    """
    Insert a chat row if it doesn't exist, otherwise leave it unchanged.
    Used by the upload endpoint so that uploading into a brand-new chat_id
    (never explicitly created via POST /chats) still registers it in the DB.
    """
    now = _utcnow()
    stmt = (
        pg_insert(Chat)
        .values(id=chat_id, created_at=now, last_active_at=now)
        .on_conflict_do_nothing(index_elements=["id"])
    )
    await db.execute(stmt)


async def bump_last_active(db: AsyncSession, chat_id: str) -> None:
    """Update last_active_at for the given chat to now."""
    await db.execute(
        update(Chat)
        .where(Chat.id == chat_id)
        .values(last_active_at=_utcnow())
    )


# ── Chat Files ────────────────────────────────────────────────────────────────

async def list_files(db: AsyncSession, chat_id: str) -> Sequence[ChatFile]:
    """Return all files uploaded under a given chat_id, oldest first."""
    result = await db.execute(
        select(ChatFile)
        .where(ChatFile.chat_id == chat_id)
        .order_by(ChatFile.uploaded_at.asc())
    )
    return result.scalars().all()


async def create_file(
    db: AsyncSession,
    *,
    chat_id: str,
    file_id: str,
    file_name: str,
    chunk_count: int,
) -> ChatFile:
    """
    Record a successfully indexed file.
    Only called after Qdrant upsert succeeds — never before.
    """
    chat_file = ChatFile(
        id=uuid.uuid4(),
        chat_id=chat_id,
        file_id=file_id,
        file_name=file_name,
        uploaded_at=_utcnow(),
        chunk_count=chunk_count,
    )
    db.add(chat_file)
    await db.flush()
    logger.info("Recorded file '%s' for chat '%s' (%d chunks)", file_name, chat_id, chunk_count)
    return chat_file


# ── Chat Messages ─────────────────────────────────────────────────────────────

async def list_messages(db: AsyncSession, chat_id: str) -> Sequence[ChatMessage]:
    """Return full conversation history for a chat, chronological order."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.created_at.asc())
    )
    return result.scalars().all()


async def append_message(
    db: AsyncSession,
    *,
    chat_id: str,
    role: str,
    content: str,
    citations: list[dict] | None = None,
) -> ChatMessage:
    """
    Persist a new chat message and bump the parent chat's last_active_at.
    citations defaults to an empty list if not provided (e.g. for user messages).
    """
    message = ChatMessage(
        id=uuid.uuid4(),
        chat_id=chat_id,
        role=role,
        content=content,
        citations=citations or [],
        created_at=_utcnow(),
    )
    db.add(message)
    await db.flush()
    # Bump parent chat activity timestamp
    await bump_last_active(db, chat_id)
    logger.info("Appended %s message for chat '%s'", role, chat_id)
    return message
