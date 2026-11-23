"""Conversations and messages for AI Tutor Chat."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from saarthi_backend.model.context_model import Base


class Conversation(Base):
    """One chat conversation (thread) per user."""

    __tablename__ = "saarthi_conversations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("saarthi_users.id", ondelete="CASCADE"), nullable=False, index=True)
    course_id: Mapped[int | None] = mapped_column(ForeignKey("saarthi_courses.id", ondelete="SET NULL"))
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ChatMessage(Base):
    """One message in a conversation (user or assistant)."""

    __tablename__ = "saarthi_chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_course_time", "course_id", "created_at"),
        Index("ix_chat_messages_reply_to", "reply_to_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("saarthi_conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    rating: Mapped[str | None] = mapped_column(String(8), nullable=True)  # "up" / "down" on an answer
    course_id: Mapped[int | None] = mapped_column(ForeignKey("saarthi_courses.id", ondelete="SET NULL"))
    topic: Mapped[str | None] = mapped_column(String(128))
    shared_query: Mapped[str | None] = mapped_column(Text)
    reply_to_id: Mapped[int | None] = mapped_column(ForeignKey("saarthi_chat_messages.id", ondelete="SET NULL"))
    answer_metadata: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))
    instructor_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
