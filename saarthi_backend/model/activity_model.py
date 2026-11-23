"""Small activity events for student and course progress."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from saarthi_backend.model.context_model import Base


class ActivityEvent(Base):
    """An open, an active-time increment, or a successful sign-in."""

    __tablename__ = "saarthi_activity_events"
    __table_args__ = (
        CheckConstraint("active_seconds BETWEEN 0 AND 30"),
        CheckConstraint("material_id IS NULL OR video_id IS NULL", name="activity_one_resource"),
        Index("ix_activity_events_user_time", "user_id", "created_at"),
        Index("ix_activity_events_course_time", "course_id", "created_at"),
        Index("ix_activity_events_material", "material_id"),
        Index("ix_activity_events_video", "video_id"),
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("saarthi_users.id", ondelete="CASCADE"), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    course_id: Mapped[int | None] = mapped_column(ForeignKey("saarthi_courses.id", ondelete="CASCADE"))
    material_id: Mapped[int | None] = mapped_column(ForeignKey("saarthi_materials.id", ondelete="CASCADE"))
    video_id: Mapped[int | None] = mapped_column(ForeignKey("saarthi_videos.id", ondelete="CASCADE"))
    active_seconds: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
