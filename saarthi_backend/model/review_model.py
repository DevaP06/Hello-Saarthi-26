"""Instructor review queue for AI answers (advisor spec 8.5 / 8.6, AI WBS 3.3.4)."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from saarthi_backend.model.context_model import Base


class AnswerReview(Base):
    """A course question the AI could not answer well (not covered, low confidence, or a
    student thumbs-down). An instructor can write the correct answer; it is then indexed into
    the course so retrieval uses it for the next student who asks."""

    __tablename__ = "saarthi_answer_reviews"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("saarthi_courses.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("saarthi_users.id", ondelete="SET NULL"), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)          # PII-scrubbed before it gets here
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reply_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reasons: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON list
    source: Mapped[str] = mapped_column(String(24), nullable=False, default="low_confidence")  # low_confidence | thumbs_down
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open", index=True)  # open | corrected | dismissed
    occurrences: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    correction: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_by: Mapped[int | None] = mapped_column(ForeignKey("saarthi_users.id", ondelete="SET NULL"), nullable=True)
    corrected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
