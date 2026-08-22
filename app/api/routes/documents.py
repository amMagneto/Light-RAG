from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.exceptions import DocumentNotFoundError, IngestionError
from app.schemas.documents import UploadResponse

router = APIRouter(tags=["documents"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/documents", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    ctx = request.app.state.ctx
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    try:
        result = await ctx.ingestion_service.ingest(file.filename or "document", payload)
    except IngestionError as exc:
        status = 415 if "Unsupported format" in exc.message else 422
        raise HTTPException(status_code=status, detail=exc.message) from exc
    return UploadResponse(**result)


@router.get("/documents")
async def list_documents(request: Request):
    ctx = request.app.state.ctx
    items = await ctx.ingestion_service.list_documents()
    return {"documents": items}


@router.get("/documents/{document_id}/status")
async def document_status(request: Request, document_id: str):
    ctx = request.app.state.ctx
    items = await ctx.ingestion_service.list_documents()
    for item in items:
        if item["meta"]["id"] == document_id:
            return item
    raise HTTPException(status_code=404, detail=f"Document {document_id} not found")


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
