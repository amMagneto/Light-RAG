import hashlib
import re
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import DocumentNotFoundError, IngestionError, UnsupportedFormatError
from app.core.logging import get_logger
from app.db.repositories import ChunkRepository, DocumentRepository
from app.ingestion.chunker import chunk_document
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

    async def ingest(self, filename: str, payload: bytes) -> dict:
        import asyncio

        doc_id = None
        try:
            safe = _safe_name(filename)
            ext = Path(safe).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise UnsupportedFormatError(f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

            doc_id_pending = hashlib.sha256(payload).hexdigest()[:16]
            storage_path = self._settings.uploads_dir / f"{doc_id_pending}_{safe}"
            self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            storage_path.write_bytes(payload)

            async with self._ctx.sessionmaker() as session:
                existing = await DocumentRepository(session).get(doc_id_pending)
                if existing is not None and existing.status == "READY":
                    chunks_existing = await ChunkRepository(session).for_document(doc_id_pending)
                    return {
                        "document_id": existing.id,
                        "filename": existing.filename,
                        "status": "READY",
                        "chunk_count": len(chunks_existing),
                        "page_count": existing.page_count,
                        "duplicate": True,
                    }

            async with self._ctx.sessionmaker() as session:
                repo = DocumentRepository(session)
                row = await repo.create(
                    id=doc_id_pending,
                    filename=safe,
                    mime_type=None,
                    status="PARSING",
                    file_size_kb=round(len(payload) / 1024, 2),
                    content_hash=hashlib.sha256(payload).hexdigest(),
                    storage_path=str(storage_path),
                )
            doc_id = row.id

            parsed = await asyncio.to_thread(parse, str(storage_path))
            async with self._ctx.sessionmaker() as session:
                await DocumentRepository(session).update(
                    doc_id,
                    mime_type=parsed.mime_type,
                    page_count=parsed.page_count,
                    status="CHUNKING",
                )

            chunks = await asyncio.to_thread(chunk_document, parsed, self._settings.chunk_size, self._settings.chunk_overlap)
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
                }
                for chunk in chunks
            ]
            stored_rows = []
            async with self._ctx.sessionmaker() as session:
                stored_rows = await ChunkRepository(session).create_many(chunk_rows)

            async with self._ctx.sessionmaker() as session:
                await DocumentRepository(session).update(
                    doc_id,
                    status="INDEXING",
                    chunk_size_used=self._effective_size(parsed),
                    chunk_overlap_used=self._settings.chunk_overlap,
                )

            index_rows = [
                {
                    "id": model.id,
                    "document_id": doc_id,
                    "document_name": safe,
                    "content": model.content,
                    "chunk_index": model.chunk_index,
                    "page_start": model.page_start,
                    "page_end": model.page_end,
                    "section_path": list(model.section_path_json or []),
                }
                for model in stored_rows
            ]
            await asyncio.to_thread(self._ctx.index.add_chunks, index_rows)
            await self.refresh_sparse()

            async with self._ctx.sessionmaker() as session:
                await DocumentRepository(session).set_status(doc_id, "READY")
            return {
                "document_id": doc_id,
                "filename": safe,
                "status": "READY",
                "chunk_count": len(stored_rows),
                "page_count": parsed.page_count,
            }
        except Exception as exc:
            log.exception("Ingestion failed for %s", filename)
            if doc_id:
                try:
                    async with self._ctx.sessionmaker() as session:
                        await DocumentRepository(session).set_status(doc_id, "FAILED", error=str(exc))
                except Exception:
                    pass
            if isinstance(exc, (UnsupportedFormatError, IngestionError)):
                raise
            raise IngestionError(f"Ingestion failed: {exc}") from exc

    def _effective_size(self, parsed) -> int | None:
        from app.ingestion.chunker import choose_chunk_config

        size, _ = choose_chunk_config(parsed.page_count, parsed.size_kb) if parsed.page_count else (
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )
        return size

    async def refresh_sparse(self) -> None:
        async with self._ctx.sessionmaker() as session:
            rows = await ChunkRepository(session).all_with_documents()
        from app.services.query_service import sparse_documents

        self._ctx.sparse.refresh(sparse_documents(rows))

    async def delete(self, document_id: str) -> None:
        import asyncio

        async with self._ctx.sessionmaker() as session:
            repo = DocumentRepository(session)
            row = await repo.get(document_id)
            if row is None:
                raise DocumentNotFoundError(document_id)
            await repo.delete(document_id)
        await asyncio.to_thread(self._ctx.index.delete_document, document_id)
        await self.refresh_sparse()

    async def list_documents(self) -> list[dict]:
        from app.schemas.documents import DocumentMeta

        async with self._ctx.sessionmaker() as session:
            rows = await DocumentRepository(session).list_all()
            counts: dict[str, int] = {}
            for chunk in await ChunkRepository(session).all_with_documents():
                counts[chunk[0].document_id] = counts.get(chunk[0].document_id, 0) + 1
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
            metas.append({"meta": meta.model_dump(mode="json"), "chunk_count": counts.get(row.id, 0)})
        return metas

    async def reindex_all(self) -> int:
        import asyncio

        async with self._ctx.sessionmaker() as session:
            rows = await ChunkRepository(session).all_with_documents()
        self._ctx.index.reset()
        documents = {}
        for chunk, filename in rows:
            documents.setdefault(
                chunk.document_id,
                {"document_name": filename, "chunks": []},
            )["chunks"].append(
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": filename,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_path": list(chunk.section_path_json or []),
                }
            )
        total = 0
        for document in documents.values():
            total += await asyncio.to_thread(self._ctx.index.add_chunks, document["chunks"])
        await self.refresh_sparse()
        return total
