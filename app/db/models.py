import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.engine import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str | None] = mapped_column(String(255), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str | None] = mapped_column(String(128), default=None)
    status: Mapped[str] = mapped_column(String(16), default="UPLOADED", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    file_size_kb: Mapped[float] = mapped_column(Float, default=0.0)
    chunk_size_used: Mapped[int | None] = mapped_column(Integer, default=None)
    chunk_overlap_used: Mapped[int | None] = mapped_column(Integer, default=None)
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    storage_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    collection_id: Mapped[str | None] = mapped_column(String(36), default=None, index=True)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class DocumentVersionModel(Base):
    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    supersedes_id: Mapped[str | None] = mapped_column(String(36), default=None)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    page_start: Mapped[int | None] = mapped_column(Integer, default=None)
    page_end: Mapped[int | None] = mapped_column(Integer, default=None)
    section_path_json: Mapped[list | None] = mapped_column(JSON, default=list)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str | None] = mapped_column(String(128), default=None)
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    version_id: Mapped[str | None] = mapped_column(String(36), default=None)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)

    @property
    def section_path(self) -> list[str]:
        return list(self.section_path_json or [])


Index("ix_chunks_document_index", ChunkModel.document_id, ChunkModel.chunk_index)


class CollectionModel(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApiKeyModel(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(128))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CollectionGrantModel(Base):
    __tablename__ = "collection_grants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    api_key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"), index=True)


class IngestionJobModel(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="QUEUED", index=True)
    document_id: Mapped[str | None] = mapped_column(String(36), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class RetrievalRunModel(Base):
    __tablename__ = "retrieval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str | None] = mapped_column(String(36), index=True, default=None)
    query: Mapped[str] = mapped_column(Text)
    normalized_query: Mapped[str | None] = mapped_column(Text, default=None)
    intent: Mapped[str | None] = mapped_column(String(32), default=None)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    verdict: Mapped[str | None] = mapped_column(String(32), default=None)
    stages_json: Mapped[dict | None] = mapped_column(JSON, default=dict)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    model: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MemoryRecordModel(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str | None] = mapped_column(String(36), index=True, default=None)
    memory_type: Mapped[str] = mapped_column(String(16), default="fact")
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
