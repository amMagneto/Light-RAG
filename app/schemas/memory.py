from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

MemoryType = Literal["preference", "fact", "context", "entity"]


def new_id() -> str:
    return uuid4().hex


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    session_id: str | None = None
    memory_type: MemoryType = "fact"
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime | None = None
    updated_at: datetime | None = None
