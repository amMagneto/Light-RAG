from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DocumentStatus = Literal[
    "UPLOADED",
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "READY",
    "FAILED",
]


class DocumentMeta(BaseModel):
    id: str
    filename: str
    mime_type: str | None = None
    status: DocumentStatus = "UPLOADED"
    version: int = 1
    page_count: int | None = None
    file_size_kb: float = 0.0
    chunk_size_used: int | None = None
    chunk_overlap_used: int | None = None
    content_hash: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int = 0
    page_count: int | None = None
    duplicate: bool = False
