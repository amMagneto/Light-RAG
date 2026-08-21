import hashlib
import re

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class DummyEmbeddings:
    """Deterministic hash-based embeddings so tests run offline with zero downloads."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vector[digest % self.dim] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_ctx(tmp_path):
    from app.api.dependencies import AppContext
    from app.core.config import Settings
    from app.db.engine import make_engine, make_sessionmaker
    from app.reasoning.llm import LLMClient
    from app.retrieval.dense import DenseRetriever
    from app.ingestion.indexer import VectorIndex

    settings = Settings(
        groq_api_key=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
        chroma_dir=str(tmp_path / "chroma"),
        storage_dir=str(tmp_path / "storage"),
        log_level="WARNING",
        enable_memory_extraction=False,
    )
    engine = make_engine(settings.database_url)
    index = VectorIndex(settings, embeddings=DummyEmbeddings())
    return AppContext(
        settings=settings,
        engine=engine,
        sessionmaker=make_sessionmaker(engine),
        llm=LLMClient(settings),
        index=index,
        dense=DenseRetriever(index),
    )


__all__ = ["DummyEmbeddings", "make_ctx", "upload_and_wait"]


def upload_and_wait(client, ctx, name: str, payload: bytes, mime: str = "text/markdown", collection: str | None = None, headers: dict | None = None) -> dict:
    """POST the file (expect 202 + job), block on the worker, return final job dict."""
    kwargs: dict = {"files": {"file": (name, payload, mime)}}
    if collection:
        kwargs["data"] = {"collection": collection}
    if headers:
        kwargs["headers"] = headers
    response = client.post("/api/v1/documents", **kwargs)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    row = ctx.job_worker.wait_for(job_id)
    assert row is not None and row.status == "DONE", f"job failed: {row.error if row else 'missing'}"
    return {
        "job_id": job_id,
        "document_id": row.document_id,
        "status": "READY",
    }
