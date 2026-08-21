from app.ingestion.indexer import VectorIndex
from app.schemas.retrieval import RetrievalResult


class DenseRetriever:
    def __init__(self, index: VectorIndex):
        self._index = index

    def search_sync(self, query: str, k: int) -> list[tuple[object, float]]:
        return self._index.query(query, k)

    async def search(self, query: str, k: int) -> RetrievalResult:
        import asyncio
        import time

        start = time.perf_counter()
        hits = await asyncio.to_thread(self.search_sync, query, k)
        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query,
            candidates=[candidate for candidate, _ in hits],
            latency_ms=latency_ms,
        )
