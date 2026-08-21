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


__all__ = ["DummyEmbeddings", "make_ctx", "AsyncSession", "async_sessionmaker"]
