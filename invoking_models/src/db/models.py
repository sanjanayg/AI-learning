"""
SQLAlchemy 2.x ORM models for persistent chat session tracking.

Tables:
  chats          — one row per chat session (partition key)
  chat_files     — one row per successfully indexed file per chat
  chat_messages  — full conversation history per chat (role/content/citations)
"""

import uuid
from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── Base ────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Helper ──────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Models ───────────────────────────────────────────────────────────────────

class Chat(Base):
    """
    Registry of every chat session ever created.
    The id is the chat_id used as the Qdrant partition key.
    """
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    chat_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships (lazy="select" default; used for cascaded deletes if needed later)
    files: Mapped[list["ChatFile"]] = relationship(
        "ChatFile", back_populates="chat", cascade="all, delete-orphan"
    )
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="chat", cascade="all, delete-orphan"
    )
    total_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("chat_name", name="uq_chat_sessions_chat_name"),
    )

    def __repr__(self) -> str:
        return f"<Chat id={self.id!r} chat_name={self.chat_name!r} last_active={self.last_active_at}>"


class ChatFile(Base):
    """
    Tracks every file successfully indexed into Qdrant for a given chat.
    Written only after a successful Qdrant upsert — never before.
    """
    __tablename__ = "chat_files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chat_id: Mapped[str] = mapped_column(
        String, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(String, nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    chat: Mapped["Chat"] = relationship("Chat", back_populates="files")

    __table_args__ = (
        # Fast lookup of all files for a given chat_id
        Index("ix_chat_files_chat_id", "chat_id"),
    )

    def __repr__(self) -> str:
        return f"<ChatFile chat_id={self.chat_id!r} file_name={self.file_name!r}>"


class ChatMessage(Base):
    """
    Full conversation history per chat session.
    citations stored as JSONB — list of Citation dicts (file_name, page_number, etc.).
    """
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    chat_id: Mapped[str] = mapped_column(
        String, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)   # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    ui_content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_used: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    is_cached: Mapped[bool] = mapped_column(nullable=False, default=False)
    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")

    __table_args__ = (
        # Fast ordered retrieval of a chat's full history
        Index("ix_chat_messages_chat_id_created_at", "chat_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<ChatMessage chat_id={self.chat_id!r} role={self.role!r}>"


class ChatSummary(Base):
    """
    Rolling summary cache per chat session.
    Maps to the existing `chat_summaries` table (created outside SQLAlchemy migrations).

    - summary         : The latest accumulated LLM-generated structured summary (JSON text).
    - last_message_id : UUID string of the most recent ChatMessage included in the summary.
                        Used to fetch only *new* messages on the next summary request.
    - updated_at      : Server-side timestamp of the last summary update.
    """
    __tablename__ = "chat_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_message_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return f"<ChatSummary chat_id={self.chat_id!r} last_message_id={self.last_message_id!r}>"
