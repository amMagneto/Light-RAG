import re
import threading
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.schemas.retrieval import RetrievalCandidate


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    return frozenset(
        "a an and are as at be by for from has have how in is it its of on or that the this to was were what "
        "when where which who will with".split()
    )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._rows: list[dict] = []
        self._size_signature: tuple[int, ...] | None = None

    @staticmethod
    def _signature(rows: list[dict]) -> tuple[int, ...]:
        return tuple(len(r["chunks"]) for r in rows)

    def refresh(self, documents: list[dict]) -> None:
        """documents: [{document_id, document_name, chunks: [row dicts with id/content/...]}]"""
        flat: list[dict] = []
        for doc in documents:
            for chunk in doc["chunks"]:
                flat.append(
                    {
                        "chunk_id": chunk["id"],
                        "document_id": doc["document_id"],
                        "document_name": doc["document_name"],
                        "content": chunk["content"],
                        "chunk_index": chunk.get("chunk_index", 0),
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        "section_path": chunk.get("section_path") or [],
                    }
                )
        signature = self._signature(documents)
        with self._lock:
            corpus = [tokenize(row["content"]) for row in flat]
            self._bm25 = BM25Okapi(corpus) if any(corpus) else None
            self._rows = flat
            self._size_signature = signature

    def needs_refresh(self, signature: tuple[int, ...]) -> bool:
        return signature != self._size_signature

    def search(self, query: str, k: int) -> list[RetrievalCandidate]:
        with self._lock:
            bm25 = self._bm25
            rows = list(self._rows)
        if bm25 is None or not rows:
            return []
        terms = [token for token in tokenize(query) if token not in _stopwords()]
        if not terms:
            terms = tokenize(query)
        term_set = set(terms)

        scores = bm25.get_scores(terms)
        matching = [i for i, row in enumerate(rows) if term_set & set(tokenize(row["content"]))]
        if not matching:
            return []

        order = sorted(matching, key=lambda i: scores[i], reverse=True)[:k]
        positive = [(i, float(scores[i])) for i in order if scores[i] > 0]
        chosen = positive or [(i, 0.01) for i in order]

        candidates = []
        for row_index, score in chosen:
            row = rows[row_index]
            candidates.append(
                RetrievalCandidate(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    document_name=row["document_name"],
                    score=score,
                    source="sparse",
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    section_path=list(row["section_path"]),
                )
            )
        return candidates
