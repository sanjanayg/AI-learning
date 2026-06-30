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

from sqlalchemy import select, update, func
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


async def get_next_chat_name(db: AsyncSession) -> str:
    """Generate the next available name like Chat 1, Chat 2, etc."""
    result = await db.execute(
        select(func.count()).select_from(Chat)
    )
    count = result.scalar() or 0
    # Find a name that doesn't already exist
    candidate = count + 1
    while True:
        name = f"Chat {candidate}"
        exists = await db.execute(select(Chat).where(Chat.chat_name == name))
        if not exists.scalar_one_or_none():
            return name
        candidate += 1


async def chat_name_exists(db: AsyncSession, chat_name: str) -> bool:
    """Return True if a chat with this name already exists."""
    result = await db.execute(select(Chat).where(Chat.chat_name == chat_name))
    return result.scalar_one_or_none() is not None


async def create_chat(db: AsyncSession, chat_id: str, chat_name: str) -> Chat:
    """Insert a brand-new chat row and return it."""
    now = _utcnow()
    chat = Chat(id=chat_id, chat_name=chat_name, created_at=now, last_active_at=now)
    db.add(chat)
    await db.flush()
    logger.info("Created new chat session: %s (%s)", chat_id, chat_name)
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
    if "Security guardrail violation" in content:
        content = "Security guardrail violation: Potential prompt injection or system override detected."
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


async def get_recent_history(db: AsyncSession, chat_id: str, limit: int =3):
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    return list(reversed(messages))  