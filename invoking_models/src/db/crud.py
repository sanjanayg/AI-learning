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

from db.models import Chat, ChatFile, ChatMessage, ChatSummary

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Chats ─────────────────────────────────────────────────────────────────────

async def list_chats(db: AsyncSession) -> Sequence[Chat]:
    """Return all chat sessions ordered by most recently active first."""
    result = await db.execute(
        select(Chat).where(Chat.status == "Y").order_by(Chat.last_active_at.desc())
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
    chat = Chat(id=chat_id, chat_name=chat_name, created_at=now, last_active_at=now,status='Y')
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


async def bump_last_active(db: AsyncSession, chat_id: str, tokens_used: int = 0) -> None:
    """Update last_active_at and increment total_tokens_used for the given chat."""
    values: dict = {"last_active_at": _utcnow()}
    if tokens_used > 0:
        # Use SQL-level increment to avoid race conditions on concurrent requests
        values["total_tokens_used"] = Chat.total_tokens_used + tokens_used
    await db.execute(
        update(Chat)
        .where(Chat.id == chat_id)
        .values(**values)
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
    ui_content: str = "",
    citations: list[dict] | None = None,
    tokens_used: int = 0,
    model_used: str | None = None,
    is_cached: bool = False,
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
        ui_content=ui_content,
        citations=citations or [],
        created_at=_utcnow(),
        tokens_used=tokens_used,
        model_used=model_used,
        is_cached=is_cached,
    )
    db.add(message)
    await db.flush()
    await bump_last_active(db, chat_id, tokens_used=tokens_used)
    logger.info("Appended %s message for chat '%s' (%d tokens)", role, chat_id, tokens_used)
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


# ── Chat Summaries ────────────────────────────────────────────────────────────

async def get_chat_summary(db: AsyncSession, chat_id: str) -> ChatSummary | None:
    """
    Fetch the rolling summary cache row for the given chat_id, or None if it
    doesn't exist yet (first-time summary request).
    """
    result = await db.execute(
        select(ChatSummary).where(ChatSummary.chat_id == chat_id)
    )
    return result.scalar_one_or_none()


async def upsert_chat_summary(
    db: AsyncSession,
    *,
    chat_id: str,
    summary: str,
    last_message_id: str,
) -> None:
    """
    Insert or update the rolling summary cache for a chat.
    Uses PostgreSQL ON CONFLICT DO UPDATE so the first call inserts and all
    subsequent calls update in place — never duplicates a row.
    """
    stmt = (
        pg_insert(ChatSummary)
        .values(
            chat_id=chat_id,
            summary=summary,
            last_message_id=last_message_id,
        )
        .on_conflict_do_update(
            index_elements=["chat_id"],
            set_={
                "summary": summary,
                "last_message_id": last_message_id,
            },
        )
    )
    await db.execute(stmt)
    logger.info(
        "Upserted chat summary for chat '%s' (last_message_id=%s)",
        chat_id,
        last_message_id,
    )


async def get_messages_after(
    db: AsyncSession,
    *,
    chat_id: str,
    after_message_id: str,
) -> list[ChatMessage]:
    """
    Return all messages for `chat_id` that were created AFTER the message
    identified by `after_message_id`, in chronological order.

    Used by the rolling-summary logic to fetch only the *new* messages since
    the last summary was generated — avoids re-sending the full history to
    the LLM on every report request.
    """
    # First resolve the created_at timestamp of the anchor message
    anchor_result = await db.execute(
        select(ChatMessage.created_at).where(
            ChatMessage.id == after_message_id,
            ChatMessage.chat_id == chat_id,
        )
    )
    anchor_ts = anchor_result.scalar_one_or_none()

    if anchor_ts is None:
        # Anchor message not found — fall back to returning all messages
        # (safe degradation; summary_service will treat this as a full rebuild)
        logger.warning(
            "anchor message_id=%s not found for chat=%s; returning all messages",
            after_message_id,
            chat_id,
        )
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(ChatMessage.created_at.asc())
        )
        return list(result.scalars().all())

    result = await db.execute(
        select(ChatMessage)
        .where(
            ChatMessage.chat_id == chat_id,
            ChatMessage.created_at > anchor_ts,
        )
        .order_by(ChatMessage.created_at.asc())
    )
    return list(result.scalars().all())