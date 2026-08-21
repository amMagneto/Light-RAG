import threading

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings
from app.schemas.retrieval import RetrievalCandidate


class VectorIndex:
    def __init__(self, settings: Settings, embeddings: Embeddings | None = None):
        self._settings = settings
        self._embeddings = embeddings
        self._lock = threading.RLock()
        self._stores: dict[str, Chroma] = {}

    def _embedding_function(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self._settings.embedding_model)
        return self._embeddings

    def _store(self, collection: str) -> Chroma:
        with self._lock:
            if collection not in self._stores:
                self._stores[collection] = Chroma(
                    embedding_function=self._embedding_function(),
                    persist_directory=self._settings.chroma_dir,
                    collection_name=collection,
                )
            return self._stores[collection]

    @property
    def chunk_store(self) -> Chroma:
        return self._store(self._settings.collection_name)

    @property
    def memory_store(self) -> Chroma:
        return self._store(self._settings.memory_collection_name)

    def add_chunks(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        texts = [f"passage: {row['content']}" for row in rows]
        metadatas = [
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "document_name": row["document_name"],
                "chunk_index": row["chunk_index"],
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "section_path": " > ".join(row.get("section_path") or []),
            }
            for row in rows
        ]
        ids = [row["id"] for row in rows]
        store = self.chunk_store
        with self._lock:
            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return len(rows)

    def query(self, text: str, k: int) -> list[tuple[RetrievalCandidate, float]]:
        results = self.chunk_store.similarity_search_with_score(f"query: {text}", k=k)
        candidates: list[tuple[RetrievalCandidate, float]] = []
        for document, distance in results:
            meta = document.metadata or {}
            content = document.page_content
            for prefix in ("passage: ", "query: "):
                if content.startswith(prefix):
                    content = content[len(prefix):]
            section_raw = meta.get("section_path") or ""
            candidates.append(
                (
                    RetrievalCandidate(
                        chunk_id=str(meta.get("chunk_id", "")),
                        document_id=str(meta.get("document_id", "")),
                        document_name=meta.get("document_name"),
                        score=max(0.0, 1.0 / (1.0 + float(distance))),
                        source="dense",
                        content=content,
                        chunk_index=int(meta.get("chunk_index") or 0),
                        page_start=meta.get("page_start"),
                        page_end=meta.get("page_end"),
                        section_path=[s for s in str(section_raw).split(" > ") if s],
                    ),
                    float(distance),
                )
            )
        return candidates

    def delete_document(self, document_id: str) -> None:
        try:
            with self._lock:
                self.chunk_store._collection.delete(where={"document_id": document_id})  # noqa: SLF001
        except Exception:
            pass

    def count(self) -> int:
        try:
            return int(self.chunk_store._collection.count())  # noqa: SLF001
        except Exception:
            return 0

    def reset(self) -> None:
        with self._lock:
            try:
                self.chunk_store.delete_collection()
            except Exception:
                pass
            self._stores.pop(self._settings.collection_name, None)

    def add_memory(self, memory_id: str, content: str) -> None:
        with self._lock:
            self.memory_store.add_texts(texts=[f"passage: {content}"], ids=[memory_id])

    def search_memories(self, text: str, k: int) -> list[str]:
        documents = self.memory_store.similarity_search(f"query: {text}", k=k)
        contents = []
        for document in documents:
            content = document.page_content
            if content.startswith("passage: "):
                content = content[len("passage: "):]
            contents.append(content)
        return contents
