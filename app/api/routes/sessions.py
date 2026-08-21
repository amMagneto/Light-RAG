from fastapi import APIRouter, HTTPException, Request

from app.schemas.responses import HealthResponse, SessionOut

router = APIRouter(tags=["sessions"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    from sqlalchemy import text

    ctx = request.app.state.ctx
    database = "ok"
    documents_ready = 0
    indexed_chunks = 0
    try:
        async with ctx.sessionmaker() as session:
            await session.execute(text("SELECT 1"))
            from app.db.repositories import ChunkRepository, DocumentRepository

            rows = await DocumentRepository(session).list_all()
            documents_ready = sum(1 for row in rows if row.status == "READY")
            indexed_chunks = await ChunkRepository(session).count()
    except Exception as exc:
        database = f"error: {exc}"

    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        app=ctx.settings.app_name,
        llm_available=ctx.llm.available,
        llm_model=ctx.llm.model_name if ctx.llm.available else None,
        database=database,
        documents_ready=documents_ready,
        indexed_chunks=indexed_chunks,
    )


@router.get("/sessions/{session_id}", response_model=SessionOut)
async def get_session(request: Request, session_id: str) -> SessionOut:
    from sqlalchemy import select

    from app.db.models import MessageModel
    from app.schemas.responses import MessageOut

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        result = await session.execute(
            select(MessageModel).where(MessageModel.session_id == session_id).order_by(MessageModel.created_at)
        )
        rows = list(result.scalars().all())
    if not rows:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionOut(
        session_id=session_id,
        messages=[
            MessageOut(role=row.role, content=row.content, model=row.model)
            for row in rows
        ],
    )


@router.get("/traces/{run_id}")
async def get_trace(request: Request, run_id: str):
    from fastapi.responses import JSONResponse

    from app.db.repositories import TraceRepository

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        row = await TraceRepository(session).get(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Trace {run_id} not found")
    return JSONResponse(
        {
            "run_id": row.id,
            "session_id": row.session_id,
            "query": row.query,
            "normalized_query": row.normalized_query,
            "intent": row.intent,
            "attempts": row.attempts,
            "verdict": row.verdict,
            "stages_ms": row.stages_json or {},
            "candidate_count": row.candidate_count,
            "confidence": row.confidence,
            "model": row.model,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
    )
