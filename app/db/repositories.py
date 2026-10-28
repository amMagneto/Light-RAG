from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ApiKeyModel,
    ChunkModel,
    CollectionGrantModel,
    CollectionModel,
    DocumentModel,
    DocumentVersionModel,
    IngestionJobModel,
    MemoryRecordModel,
    MessageModel,
    RetrievalRunModel,
    SessionModel,
)


class CollectionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, name: str) -> CollectionModel:
        row = CollectionModel(name=name)
        self._session.add(row)
        await self._session.commit()
        return row

    async def get_by_name(self, name: str) -> CollectionModel | None:
        result = await self._session.execute(
            select(CollectionModel).where(CollectionModel.name == name).limit(1)
        )
        return result.scalars().first()

    async def get_or_create(self, name: str) -> CollectionModel:
        existing = await self.get_by_name(name)
        return existing if existing is not None else await self.create(name)

    async def ensure_default(self) -> CollectionModel:
        return await self.get_or_create("default")

    async def list_all(self) -> list[CollectionModel]:
        result = await self._session.execute(select(CollectionModel).order_by(CollectionModel.name))
        return list(result.scalars().all())


class ApiKeyRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(ApiKeyModel))
        return int(result.scalar_one())

    async def create(
        self,
        name: str,
        key_hash: str,
        is_admin: bool = False,
        collection_ids: list[str] | None = None,
    ) -> ApiKeyModel:
        row = ApiKeyModel(name=name, key_hash=key_hash, is_admin=is_admin)
        self._session.add(row)
        await self._session.flush()
        for collection_id in collection_ids or []:
            self._session.add(CollectionGrantModel(api_key_id=row.id, collection_id=collection_id))
        await self._session.commit()
        return row

    async def get_by_hash(self, key_hash: str) -> ApiKeyModel | None:
        result = await self._session.execute(
            select(ApiKeyModel).where(ApiKeyModel.key_hash == key_hash, ApiKeyModel.revoked.is_(False)).limit(1)
        )
        return result.scalars().first()

    async def grants_for_key(self, api_key_id: str) -> list[str]:
        result = await self._session.execute(
            select(CollectionGrantModel.collection_id).where(CollectionGrantModel.api_key_id == api_key_id)
        )
        return [row for row in result.scalars()]

    async def list_all(self) -> list[ApiKeyModel]:
        result = await self._session.execute(select(ApiKeyModel).order_by(ApiKeyModel.created_at))
        return list(result.scalars().all())

    async def revoke(self, api_key_id: str) -> bool:
        row = await self._session.get(ApiKeyModel, api_key_id)
        if row is None:
            return False
        row.revoked = True
        await self._session.commit()
        return True


class IngestionJobRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, filename: str) -> IngestionJobModel:
        row = IngestionJobModel(filename=filename, status="QUEUED")
        self._session.add(row)
        await self._session.commit()
        return row

    async def get(self, job_id: str) -> IngestionJobModel | None:
        return await self._session.get(IngestionJobModel, job_id)

    async def set_status(
        self,
        job_id: str,
        status: str,
        document_id: str | None = None,
        error: str | None = None,
    ) -> None:
        row = await self.get(job_id)
        if row is not None:
            row.status = status
            if document_id is not None:
                row.document_id = document_id
            if error is not None:
                row.error = error
            await self._session.commit()

    async def recent(self, limit: int = 20) -> list[IngestionJobModel]:
        result = await self._session.execute(
            select(IngestionJobModel).order_by(IngestionJobModel.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())


class DocumentVersionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(
        self,
        document_id: str,
        version: int,
        content_hash: str | None,
        supersedes_id: str | None = None,
    ) -> DocumentVersionModel:
        row = DocumentVersionModel(
            document_id=document_id,
            version=version,
            content_hash=content_hash,
            supersedes_id=supersedes_id,
        )
        self._session.add(row)
        await self._session.commit()
        return row

    async def latest_for_document(self, document_id: str) -> DocumentVersionModel | None:
        result = await self._session.execute(
            select(DocumentVersionModel)
            .where(DocumentVersionModel.document_id == document_id)
            .order_by(DocumentVersionModel.version.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def get(self, version_id: str) -> DocumentVersionModel | None:
        return await self._session.get(DocumentVersionModel, version_id)

    async def list_for_document(self, document_id: str) -> list[DocumentVersionModel]:
        result = await self._session.execute(
            select(DocumentVersionModel)
            .where(DocumentVersionModel.document_id == document_id)
            .order_by(DocumentVersionModel.version.desc())
        )
        return list(result.scalars().all())

    async def set_chunk_count(self, version_id: str, chunk_count: int) -> None:
        row = await self.get(version_id)
        if row is not None:
            row.chunk_count = chunk_count
            await self._session.commit()


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

    async def get_by_filename(self, filename: str) -> DocumentModel | None:
        result = await self._session.execute(
            select(DocumentModel).where(DocumentModel.filename == filename).limit(1)
        )
        return result.scalars().first()

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

    async def all_with_documents(self, current_only: bool = False) -> list[tuple[ChunkModel, str, str | None]]:
        statement = (
            select(ChunkModel, DocumentModel.filename, DocumentModel.collection_id)
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
            .order_by(ChunkModel.document_id, ChunkModel.chunk_index)
        )
        if current_only:
            statement = statement.where(ChunkModel.is_current.is_(True))
        result = await self._session.execute(statement)
        return [(chunk, filename, collection_id) for chunk, filename, collection_id in result.all()]

    async def count(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(ChunkModel))
        return int(result.scalar_one())

    async def current_for_document(self, document_id: str) -> list[ChunkModel]:
        result = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id, ChunkModel.is_current.is_(True))
            .order_by(ChunkModel.chunk_index)
        )
        return list(result.scalars().all())

    async def current_hashes_for_document(self, document_id: str) -> set[str]:
        result = await self._session.execute(
            select(ChunkModel.content_hash).where(
                ChunkModel.document_id == document_id, ChunkModel.is_current.is_(True)
            )
        )
        return {hash for hash in result.scalars() if hash}

    async def mark_document_stale(self, document_id: str) -> None:
        await self._session.execute(
            update(ChunkModel)
            .where(ChunkModel.document_id == document_id, ChunkModel.is_current.is_(True))
            .values(is_current=False)
        )
        await self._session.commit()

    async def count_current(self) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(ChunkModel).where(ChunkModel.is_current.is_(True))
        )
        return int(result.scalar_one())

    async def delete_for_document(self, document_id: str) -> None:
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
        await self._session.commit()


class TraceRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def record(self, **kwargs) -> RetrievalRunModel:
        row = RetrievalRunModel(**kwargs)
        self._session.add(row)
        await self._session.commit()
        return row

    async def get(self, run_id: str) -> RetrievalRunModel | None:
        return await self._session.get(RetrievalRunModel, run_id)


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
