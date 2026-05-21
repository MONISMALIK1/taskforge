"""
models.py — SQLAlchemy ORM models and Pydantic schemas.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Column, DateTime, String, Text
from sqlalchemy import Enum as SAEnum

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Status enum ───────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done    = "done"
    failed  = "failed"


# ── ORM model ─────────────────────────────────────────────────────────────────


class TaskRecord(Base):
    __tablename__ = "tasks"

    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt     = Column(Text,       nullable=False)
    status     = Column(SAEnum(TaskStatus), nullable=False, default=TaskStatus.pending)
    result     = Column(Text,       nullable=True)
    logs       = Column(JSON,       nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class TaskCreate(BaseModel):
    prompt: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        examples=["Fetch the top 5 GitHub trending Python repos and save them to a file"],
    )


class TaskResponse(BaseModel):
    id:         str
    prompt:     str
    status:     TaskStatus
    result:     Optional[str]  = None
    logs:       list[str]      = []
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
