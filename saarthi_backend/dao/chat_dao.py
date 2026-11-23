"""DAO for tutor chat conversations and messages."""

from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from saarthi_backend.model import ChatMessage, Conversation


class ConversationDAO:
    @staticmethod
    async def create(db: AsyncSession, user_id: int, title: str = "New Chat", course_id: int | None = None) -> Conversation:
        c = Conversation(user_id=user_id, title=title, course_id=course_id)
        db.add(c)
        await db.flush()
        await db.refresh(c)
        return c

    @staticmethod
    async def get_by_id(db: AsyncSession, conversation_id: int, user_id: int | None = None) -> Conversation | None:
        q = select(Conversation).where(Conversation.id == conversation_id)
        if user_id is not None:
            q = q.where(Conversation.user_id == user_id)
        r = await db.execute(q)
        return r.scalar_one_or_none()

    @staticmethod
    async def list_by_user(
        db: AsyncSession,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        *,
        topic: str | None = None,
        on_date: date | None = None,
    ) -> tuple[list[Conversation], int]:
        filters = [Conversation.user_id == user_id]
        if topic or on_date:
            message = select(ChatMessage.id).where(ChatMessage.conversation_id == Conversation.id, ChatMessage.role == "user")
            if topic:
                label = topic.lower()
                message = message.where(or_(
                    func.lower(ChatMessage.topic).contains(label, autoescape=True),
                    ChatMessage.topic.is_(None) & func.lower(ChatMessage.content).contains(label, autoescape=True),
                ))
            if on_date:
                start = datetime.combine(on_date, time.min, tzinfo=timezone.utc)
                message = message.where(ChatMessage.created_at >= start)
                if on_date != date.max:
                    message = message.where(ChatMessage.created_at < start + timedelta(days=1))
            filters.append(message.exists())
        q = (
            select(Conversation)
            .where(*filters)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        r = await db.execute(q)
        total = await db.scalar(select(func.count()).select_from(Conversation).where(*filters))
        return list(r.scalars().all()), total or 0

    @staticmethod
    async def update_title(db: AsyncSession, conversation_id: int, user_id: int, title: str) -> Conversation | None:
        c = await ConversationDAO.get_by_id(db, conversation_id, user_id)
        if not c:
            return None
        c.title = title
        await db.flush()
        await db.refresh(c)
        return c

    @staticmethod
    async def delete(db: AsyncSession, conversation_id: int, user_id: int) -> bool:
        c = await ConversationDAO.get_by_id(db, conversation_id, user_id)
        if not c:
            return False
        await db.delete(c)
        await db.flush()
        return True

    @staticmethod
    async def touch(db: AsyncSession, conversation_id: int) -> None:
        """Update updated_at (e.g. after adding a message)."""
        await db.execute(
            update(Conversation).where(Conversation.id == conversation_id).values(updated_at=datetime.now(timezone.utc))
        )
        await db.flush()


class ChatMessageDAO:
    @staticmethod
    async def create(
        db: AsyncSession, conversation_id: int, role: str, content: str, *,
        course_id: int | None = None, topic: str | None = None, shared_query: str | None = None,
        reply_to_id: int | None = None, answer_metadata: dict | None = None,
    ) -> ChatMessage:
        m = ChatMessage(conversation_id=conversation_id, role=role, content=content, course_id=course_id,
                        topic=topic, shared_query=shared_query, reply_to_id=reply_to_id, answer_metadata=answer_metadata)
        db.add(m)
        await db.flush()
        await db.refresh(m)
        return m

    @staticmethod
    async def list_by_conversation(db: AsyncSession, conversation_id: int) -> list[ChatMessage]:
        q = (
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        )
        r = await db.execute(q)
        return list(r.scalars().all())
