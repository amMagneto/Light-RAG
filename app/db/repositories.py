from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ChunkModel,
    DocumentModel,
    MemoryRecordModel,
    MessageModel,
    RetrievalRunModel,
    SessionModel,
)


class SessionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_or_create(self, session_id: str | None) -> str:
        if session_id:
            existing = await self._session.get(SessionModel, session_id)
            if existing is not None:
                return existing.id
            row = SessionModel(id=session_id)
        else:
            row = SessionModel()
        self._session.add(row)
        await self._session.commit()
        return row.id

    async def add_message(self, session_id: str, role: str, content: str, model: str | None = None) -> None:
        self._session.add(MessageModel(session_id=session_id, role=role, content=content, model=model))
        await self._session.commit()

    async def history(self, session_id: str, limit: int = 12) -> list[tuple[str, str]]:
        result = await self._session.execute(
            select(MessageModel)
            .where(MessageModel.session_id == session_id)
            .order_by(MessageModel.created_at.desc())
            .limit(limit)
        )
        rows = list(result.scalars().all())
        rows.reverse()
        return [(row.role, row.content) for row in rows]


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, **kwargs) -> DocumentModel:
        row = DocumentModel(**kwargs)
        self._session.add(row)
        await self._session.commit()
        return row

    async def get(self, document_id: str) -> DocumentModel | None:
        return await self._session.get(DocumentModel, document_id)

    async def list_all(self) -> list[DocumentModel]:
        result = await self._session.execute(select(DocumentModel).order_by(DocumentModel.created_at.desc()))
        return list(result.scalars().all())

    async def set_status(self, document_id: str, status: str, error: str | None = None) -> None:
        row = await self.get(document_id)
        if row is not None:
            row.status = status
            row.error = error
            await self._session.commit()

    async def update(self, document_id: str, **kwargs) -> None:
        row = await self.get(document_id)
        if row is not None:
            for key, value in kwargs.items():
                setattr(row, key, value)
            await self._session.commit()

    async def delete(self, document_id: str) -> None:
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
        await self._session.execute(delete(DocumentModel).where(DocumentModel.id == document_id))
        await self._session.commit()


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_many(self, rows: list[dict]) -> list[ChunkModel]:
        models = [ChunkModel(**row) for row in rows]
        self._session.add_all(models)
        await self._session.commit()
        return models

    async def for_document(self, document_id: str) -> list[ChunkModel]:
        result = await self._session.execute(
            select(ChunkModel).where(ChunkModel.document_id == document_id).order_by(ChunkModel.chunk_index)
        )
        return list(result.scalars().all())

    async def all_with_documents(self) -> list[tuple[ChunkModel, str]]:
        result = await self._session.execute(
            select(ChunkModel, DocumentModel.filename)
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
            .order_by(ChunkModel.document_id, ChunkModel.chunk_index)
        )
        return [(chunk, filename) for chunk, filename in result.all()]

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(ChunkModel))
        return int(result.scalar_one())


class TraceRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def record(self, **kwargs) -> RetrievalRunModel:
        row = RetrievalRunModel(**kwargs)
        self._session.add(row)
        await self._session.commit()
        return row


class MemoryRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(
        self,
        memory_id: str,
        memory_type: str,
        content: str,
        importance: float,
        confidence: float,
        session_id: str | None,
    ) -> MemoryRecordModel:
        row = MemoryRecordModel(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            confidence=confidence,
            session_id=session_id,
        )
        self._session.add(row)
        await self._session.commit()
        return row

    async def all(self, limit: int = 500) -> list[MemoryRecordModel]:
        result = await self._session.execute(select(MemoryRecordModel).order_by(MemoryRecordModel.created_at.desc()).limit(limit))
        return list(result.scalars().all())
