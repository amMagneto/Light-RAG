import hashlib
import re
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import DocumentNotFoundError, IngestionError, UnsupportedFormatError
from app.core.logging import get_logger
from app.db.repositories import ChunkRepository, DocumentRepository, DocumentVersionRepository
from app.ingestion.chunker import choose_chunk_config, chunk_document
from app.ingestion.indexer import vector_key
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, parse

log = get_logger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9._\- ]+")


def _safe_name(filename: str) -> str:
    name = Path(filename or "document").name
    return _SAFE.sub("_", name).strip() or "document"


class IngestionService:
    def __init__(self, ctx):
        self._ctx = ctx
        self._settings: Settings = ctx.settings

    async def ingest(self, filename: str, payload: bytes, collection_id: str | None = None) -> dict:
        """Versioned incremental ingest.

        Same filename + same content hash -> no-op duplicate.
        Same filename + changed content   -> new version; only net-new chunk
        content gets embedded, vanished content is purged from the index.
        """
        import asyncio

        safe = _safe_name(filename)
        ext = Path(safe).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
        payload_hash = hashlib.sha256(payload).hexdigest()

        storage_path = self._settings.uploads_dir / f"{payload_hash[:16]}_{safe}"
        self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        storage_path.write_bytes(payload)

        async with self._ctx.sessionmaker() as session:
            doc_repo = DocumentRepository(session)
            version_repo = DocumentVersionRepository(session)
            chunk_repo = ChunkRepository(session)

            doc = await doc_repo.get_by_filename(safe)
            supersedes_id = None
            if doc is None:
                doc = await doc_repo.create(
                    filename=safe,
                    mime_type=None,
                    status="PARSING",
                    file_size_kb=round(len(payload) / 1024, 2),
                    content_hash=payload_hash,
                    storage_path=str(storage_path),
                    collection_id=collection_id,
                )
                version_number = 1
                previous_hashes: set[str] = set()
            else:
                latest = await version_repo.latest_for_document(doc.id)
                if latest is not None and latest.content_hash == payload_hash:
                    current_count = len(await chunk_repo.current_for_document(doc.id))
                    return {
                        "document_id": doc.id,
                        "filename": safe,
                        "status": "READY",
                        "chunk_count": current_count,
                        "page_count": doc.page_count,
                        "version": latest.version,
                        "duplicate": True,
                        "chunks_embedded": 0,
                        "chunks_reused": current_count,
                    }
                version_number = (latest.version + 1) if latest else 1
                supersedes_id = latest.id if latest else None
                previous_hashes = await chunk_repo.current_hashes_for_document(doc.id)
                await chunk_repo.mark_document_stale(doc.id)
                await doc_repo.update(
                    doc.id,
                    status="PARSING",
                    error=None,
                    content_hash=payload_hash,
                    file_size_kb=round(len(payload) / 1024, 2),
                    storage_path=str(storage_path),
                )

            version_row = await version_repo.create(
                document_id=doc.id,
                version=version_number,
                content_hash=payload_hash,
                supersedes_id=supersedes_id,
            )
            doc_id = doc.id

        try:
            parsed = await asyncio.to_thread(parse, str(storage_path))
            async with self._ctx.sessionmaker() as session:
                await DocumentRepository(session).update(
                    doc_id, mime_type=parsed.mime_type, page_count=parsed.page_count, status="CHUNKING"
                )

            chunks = await asyncio.to_thread(
                chunk_document, parsed, self._settings.chunk_size, self._settings.chunk_overlap
            )
            if not chunks:
                raise IngestionError("Document produced no chunks (empty content?)")

            chunk_rows = [
                {
                    "document_id": doc_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_path_json": chunk.section_path,
                    "token_count": chunk.token_count,
                    "embedding_model": self._settings.embedding_model,
                    "content_hash": chunk.content_hash,
                    "version_id": version_row.id,
                    "is_current": True,
                }
                for chunk in chunks
            ]
            async with self._ctx.sessionmaker() as session:
                stored_models = await ChunkRepository(session).create_many(chunk_rows)

            async with self._ctx.sessionmaker() as session:
                size, _ = (
                    choose_chunk_config(parsed.page_count, parsed.size_kb)
                    if parsed.page_count
                    else (self._settings.chunk_size, self._settings.chunk_overlap)
                )
                await DocumentRepository(session).update(doc_id, status="INDEXING", chunk_size_used=size)

            stats = await self._sync_vectors(
                doc_id=doc_id,
                document_name=safe,
                stored_models=stored_models,
                previous_hashes=previous_hashes,
                collection_id=collection_id,
            )

            await self.refresh_sparse()

            async with self._ctx.sessionmaker() as session:
                await DocumentRepository(session).update(doc_id, status="READY", version=version_number)
                await DocumentVersionRepository(session).set_chunk_count(version_row.id, len(stored_models))

            return {
                "document_id": doc_id,
                "filename": safe,
                "status": "READY",
                "chunk_count": len(stored_models),
                "page_count": parsed.page_count,
                "version": version_number,
                "duplicate": False,
                **stats,
            }
        except Exception as exc:
            log.exception("Ingestion failed for %s", filename)
            try:
                async with self._ctx.sessionmaker() as session:
                    await DocumentRepository(session).set_status(doc_id, "FAILED", error=str(exc))
            except Exception:
                pass
            if isinstance(exc, (UnsupportedFormatError, IngestionError)):
                raise
            raise IngestionError(f"Ingestion failed: {exc}") from exc

    async def _sync_vectors(self, doc_id: str, document_name: str, stored_models, previous_hashes: set[str], collection_id: str | None = None) -> dict:
        """Content-addressed incremental sync: delete vanished keys, skip already-
        embedded content, embed and upsert only net-new chunks."""
        import asyncio

        new_keys: dict[str, dict] = {}
        for model in stored_models:
            key = vector_key(doc_id, model.content_hash)
            new_keys[key] = {
                "vector_key": key,
                "id": model.id,
                "document_id": doc_id,
                "document_name": document_name,
                "collection_id": collection_id,
                "content": model.content,
                "chunk_index": model.chunk_index,
                "page_start": model.page_start,
                "page_end": model.page_end,
                "section_path": list(model.section_path_json or []),
            }

        stale_keys = [vector_key(doc_id, h) for h in previous_hashes if h not in {m.content_hash for m in stored_models}]
        deleted = await asyncio.to_thread(self._ctx.index.delete_keys, stale_keys)

        existing = await asyncio.to_thread(self._ctx.index.existing_keys, list(new_keys))
        to_embed = [new_keys[key] for key in new_keys if key not in existing]
        embedded = await asyncio.to_thread(self._ctx.index.upsert_chunks, to_embed)

        return {
            "chunks_embedded": embedded,
            "chunks_reused": len(new_keys) - len(to_embed),
            "chunks_purged": deleted,
        }

    async def refresh_sparse(self) -> None:
        async with self._ctx.sessionmaker() as session:
            rows = await ChunkRepository(session).all_with_documents(current_only=True)
        from app.services.query_service import sparse_documents

        self._ctx.sparse.refresh(sparse_documents(rows))

    async def delete(self, document_id: str) -> None:
        import asyncio

        from sqlalchemy import delete as sa_delete

        from app.db.models import DocumentVersionModel

        async with self._ctx.sessionmaker() as session:
            repo = DocumentRepository(session)
            row = await repo.get(document_id)
            if row is None:
                raise DocumentNotFoundError(document_id)
            await ChunkRepository(session).delete_for_document(document_id)
            await session.execute(sa_delete(DocumentVersionModel).where(DocumentVersionModel.document_id == document_id))
            await repo.delete(document_id)
        await asyncio.to_thread(self._ctx.index.delete_document, document_id)
        await self.refresh_sparse()

    async def list_documents(self) -> list[dict]:
        from sqlalchemy import select

        from app.db.models import DocumentVersionModel
        from app.schemas.documents import DocumentMeta

        async with self._ctx.sessionmaker() as session:
            rows = await DocumentRepository(session).list_all()
            counts: dict[str, int] = {}
            for chunk in await ChunkRepository(session).all_with_documents(current_only=True):
                counts[chunk[0].document_id] = counts.get(chunk[0].document_id, 0) + 1
            version_rows = (
                await session.execute(select(DocumentVersionModel).order_by(DocumentVersionModel.version.desc()))
            ).scalars().all()
        versions_by_doc: dict[str, list[dict]] = {}
        for v in version_rows:
            versions_by_doc.setdefault(v.document_id, []).append(
                {"version": v.version, "content_hash": v.content_hash, "chunk_count": v.chunk_count}
            )
        metas = []
        for row in rows:
            meta = DocumentMeta(
                id=row.id,
                filename=row.filename,
                mime_type=row.mime_type,
                status=row.status,
                version=row.version,
                page_count=row.page_count,
                file_size_kb=row.file_size_kb,
                chunk_size_used=row.chunk_size_used,
                chunk_overlap_used=row.chunk_overlap_used,
                content_hash=row.content_hash,
                error=row.error,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            metas.append(
                {
                    "meta": meta.model_dump(mode="json"),
                    "chunk_count": counts.get(row.id, 0),
                    "collection_id": row.collection_id,
                    "versions": versions_by_doc.get(row.id, []),
                }
            )
        return metas

    async def version_diff(self, document_id: str, from_version: int, to_version: int) -> dict:
        """Chunk-level diff between two versions of one document, by content hash."""
        from sqlalchemy import select

        from app.db.models import DocumentVersionModel

        async with self._ctx.sessionmaker() as session:
            result = await session.execute(
                select(DocumentVersionModel).where(DocumentVersionModel.document_id == document_id)
            )
            versions = {v.version: v for v in result.scalars().all()}
            if from_version not in versions or to_version not in versions:
                raise DocumentNotFoundError(f"version {from_version} or {to_version} not found")

            from_chunks = await self._chunks_of_version(versions[from_version])
            to_chunks = await self._chunks_of_version(versions[to_version])

        from_by_hash = {c.content_hash: c for c in from_chunks}
        to_by_hash = {c.content_hash: c for c in to_chunks}

        def summary(chunk) -> dict:
            return {
                "section_path": list(chunk.section_path_json or []),
                "excerpt": chunk.content[:160],
                "page_start": chunk.page_start,
            }

        return {
            "document_id": document_id,
            "from_version": from_version,
            "to_version": to_version,
            "added": [summary(c) for h, c in to_by_hash.items() if h not in from_by_hash],
            "removed": [summary(c) for h, c in from_by_hash.items() if h not in to_by_hash],
            "unchanged": sum(1 for h in to_by_hash if h in from_by_hash),
        }

    async def _chunks_of_version(self, version_row):
        from sqlalchemy import select

        from app.db.models import ChunkModel

        async with self._ctx.sessionmaker() as session:
            result = await session.execute(
                select(ChunkModel).where(ChunkModel.version_id == version_row.id).order_by(ChunkModel.chunk_index)
            )
            return list(result.scalars().all())

    async def reindex_all(self) -> int:
        import asyncio

        async with self._ctx.sessionmaker() as session:
            rows = await ChunkRepository(session).all_with_documents(current_only=True)
        self._ctx.index.reset()
        batch = [
            {
                "vector_key": vector_key(chunk.document_id, chunk.content_hash),
                "id": chunk.id,
                "document_id": chunk.document_id,
                "document_name": filename,
                "collection_id": collection_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": list(chunk.section_path_json or []),
            }
            for chunk, filename, collection_id in rows
        ]
        total = await asyncio.to_thread(self._ctx.index.upsert_chunks, batch)
        await self.refresh_sparse()
        return total
