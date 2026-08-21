import asyncio
import uuid
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.repositories import MemoryRepository
from app.ingestion.indexer import VectorIndex
from app.reasoning.llm import LLMClient
from app.schemas.memory import MemoryType

log = get_logger(__name__)

MEMORY_SYSTEM = """You are the memory extraction module of a retrieval system.
From the exchange below, extract durable facts worth remembering across sessions:
- preference: how the user wants answers (style, scope, constraints)
- fact: stable facts about the user's projects, organization, or domain
- context: ongoing project/initiative context useful later
- entity: important named entities and what they refer to
Only include information that would still matter in future conversations.
Return an empty list if nothing is durable. Maximum 3 items."""


class ExtractedMemory(BaseModel):
    memory_type: MemoryType = "fact"
    content: str = Field(min_length=1)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractionOutput(BaseModel):
    memories: list[ExtractedMemory] = Field(default_factory=list)


class MemoryService:
    def __init__(
        self,
        llm: LLMClient,
        index: VectorIndex,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
    ):
        self._llm = llm
        self._index = index
        self._settings = settings
        self._sessionmaker = sessionmaker

    @asynccontextmanager
    async def _session(self):
        async with self._sessionmaker() as session:
            yield session

    async def recall(self, query: str) -> list[str]:
        try:
            notes = await asyncio.to_thread(
                self._index.search_memories, query, self._settings.memory_recall_k
            )
            return [note for note in notes if len(note.strip()) > 8][: self._settings.memory_recall_k]
        except Exception as exc:
            log.warning("Memory recall failed: %s", exc)
            return []

    async def extract(self, session_id: str | None, user_text: str, answer_text: str) -> int:
        if not self._settings.enable_memory_extraction or not self._llm.available:
            return 0
        user = f"USER:\n{user_text[:1500]}\n\nASSISTANT:\n{answer_text[:1500]}"
        try:
            output = await self._llm.structured(MEMORY_SYSTEM, user, ExtractionOutput)
        except Exception as exc:
            log.warning("Memory extraction failed: %s", exc)
            return 0

        stored = 0
        try:
            async with self._session() as session:
                repo = MemoryRepository(session)
                for item in output.memories[:3]:
                    record = await repo.add(
                        memory_id=uuid.uuid4().hex,
                        memory_type=item.memory_type,
                        content=item.content,
                        importance=item.importance,
                        confidence=0.6,
                        session_id=session_id,
                    )
                    await asyncio.to_thread(self._index.add_memory, record.id, item.content)
                    stored += 1
        except Exception as exc:
            log.warning("Memory persistence failed: %s", exc)
        return stored
