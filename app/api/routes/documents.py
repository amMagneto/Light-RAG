from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from app.api.auth import get_identity
from app.core.exceptions import DocumentNotFoundError, IngestionError
from app.schemas.documents import JobResponse, UploadResponse, VersionDiff

router = APIRouter(tags=["documents"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/documents", status_code=202, response_model=JobResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    collection: str | None = Form(default=None),
    identity=Depends(get_identity),
) -> JobResponse:
    """Queues ingestion and returns immediately. Poll /documents/jobs/{job_id}."""
    ctx = request.app.state.ctx
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    filename = file.filename or "document"

    collection_id = None
    if collection and collection.strip():
        from app.api.auth import collection_filter
        from app.db.repositories import CollectionRepository

        async with ctx.sessionmaker() as session:
            row = await CollectionRepository(session).get_or_create(collection.strip())
        collection_id = row.id
        scope = collection_filter(identity)
        if scope is not None and collection_id not in scope:
            raise HTTPException(status_code=403, detail=f"Not authorized for collection '{collection}'")

    from pathlib import Path

    from app.ingestion.loaders import SUPPORTED_EXTENSIONS

    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{Path(filename).suffix.lower()}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    try:
        job_id = ctx.job_worker.submit(filename, payload, collection_id)
    except IngestionError as exc:
        status = 415 if "Unsupported format" in exc.message else 422
        raise HTTPException(status_code=status, detail=exc.message) from exc
    return JobResponse(job_id=job_id, filename=filename, status="QUEUED")


@router.get("/documents/jobs/{job_id}", response_model=JobResponse)
async def job_status(request: Request, job_id: str) -> JobResponse:
    from sqlalchemy import select

    from app.db.models import IngestionJobModel

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        row = await session.get(IngestionJobModel, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobResponse(
        job_id=row.id,
        filename=row.filename,
        status=row.status,
        document_id=row.document_id,
        error=row.error,
    )


@router.get("/documents/jobs")
async def recent_jobs(request: Request):
    from app.db.repositories import IngestionJobRepository

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        rows = await IngestionJobRepository(session).recent()
    return {
        "jobs": [
            {
                "job_id": row.id,
                "filename": row.filename,
                "status": row.status,
                "document_id": row.document_id,
                "error": row.error,
            }
            for row in rows
        ]
    }


async def _load_doc_meta(ctx, document_id: str) -> dict | None:
    items = await ctx.ingestion_service.list_documents()
    for item in items:
        if item["meta"]["id"] == document_id:
            return item
    return None


@router.get("/documents")
async def list_documents(request: Request):
    ctx = request.app.state.ctx
    items = await ctx.ingestion_service.list_documents()
    return {"documents": items}


@router.get("/documents/{document_id}/versions")
async def document_versions(request: Request, document_id: str):
    from app.db.repositories import DocumentRepository, DocumentVersionRepository

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        if await DocumentRepository(session).get(document_id) is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        rows = await DocumentVersionRepository(session).list_for_document(document_id)
    return {
        "document_id": document_id,
        "versions": [
            {
                "version": row.version,
                "content_hash": row.content_hash,
                "chunk_count": row.chunk_count,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
    }


@router.get("/documents/{document_id}/diff", response_model=VersionDiff)
async def document_diff(
    request: Request,
    document_id: str,
    from_version: int = Query(...),
    to_version: int = Query(...),
) -> VersionDiff:
    ctx = request.app.state.ctx
    try:
        result = await ctx.ingestion_service.version_diff(document_id, from_version, to_version)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc
    return VersionDiff(**result)


@router.get("/documents/{document_id}/status")
async def document_status(request: Request, document_id: str):
    ctx = request.app.state.ctx
    item = await _load_doc_meta(ctx, document_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return item


@router.delete("/documents/{document_id}")
async def delete_document(request: Request, document_id: str):
    ctx = request.app.state.ctx
    try:
        await ctx.ingestion_service.delete(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc
    return {"status": "deleted", "document_id": document_id}


@router.post("/admin/reindex")
async def reindex(request: Request):
    ctx = request.app.state.ctx
    count = await ctx.ingestion_service.reindex_all()
    return {"status": "rebuilt", "chunk_count": count}
