from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import build_context, load_sparse
from app.api.routes import chat, documents, sessions
from app.core.config import get_settings
from app.core.exceptions import LightRAGError
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    ctx = build_context(settings)
    from app.db.engine import init_db

    await init_db(ctx.engine)
    await load_sparse(ctx)
    app.state.ctx = ctx
    log.info(
        "%s started | llm=%s | db=%s",
        settings.app_name,
        "available" if ctx.llm.available else "unavailable",
        settings.database_url.split("://")[0],
    )
    yield
    await ctx.engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Light-RAG",
        description="Vertical retrieval platform: typed pipeline, hybrid search, evidence grading, grounded generation.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(LightRAGError)
    async def lightrag_error_handler(request, exc: LightRAGError):
        status = 404 if "not found" in exc.message.lower() else 400
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=status, content={"detail": exc.message, "details": exc.details})

    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(sessions.router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {"app": "Light-RAG", "docs": "/docs", "api": "/api/v1"}

    return app


app = create_app()
