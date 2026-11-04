
import uuid

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    JSON,
    String
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    username = Column(
        String(100),
        nullable=False
    )

    password_hash = Column(
        String(255),
        nullable=False
    )

    is_active = Column(
        Boolean,
        default=True,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    analyses = relationship(
        "Analysis",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    stroke_type = Column(
        String(50),
        nullable=False
    )

    form_score = Column(
        Float,
        nullable=True
    )

    biomechanical_score = Column(
        Float,
        nullable=True
    )

    temporal_similarity_score = Column(
        Float,
        nullable=True
    )

    reconstruction_error = Column(
        Float,
        nullable=True
    )

    # Complete analysis response from the evaluator
    result_data = Column(
        JSON,
        nullable=False
    )

    # Optional video storage information
    video_saved = Column(
        Boolean,
        default=False,
        nullable=False
    )

    video_object_key = Column(
        String(500),
        nullable=True
    )

    video_filename = Column(
        String(255),
        nullable=True
    )

    video_expires_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    user = relationship(
        "User",
        back_populates="analyses"
    )
