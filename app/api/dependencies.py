from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.context.builder import ContextBuilder
from app.db.engine import make_engine, make_sessionmaker
from app.generation.generator import Generator
from app.ingestion.indexer import VectorIndex
from app.memory.service import MemoryService
from app.reasoning.grader import EvidenceGrader
from app.reasoning.llm import LLMClient
from app.reasoning.planner import QueryPlanner
from app.reasoning.verifier import AnswerVerifier
from app.retrieval.dense import DenseRetriever
from app.retrieval.reranker import LexicalReranker
from app.retrieval.sparse import SparseIndex
from app.services.query_service import QueryService, RetrievalEngine


@dataclass
class AppContext:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    llm: LLMClient
    index: VectorIndex
    dense: DenseRetriever
    sparse: SparseIndex = field(default_factory=SparseIndex)
    reranker: LexicalReranker = field(default_factory=LexicalReranker)
    planner: QueryPlanner = field(init=False)
    grader: EvidenceGrader = field(init=False)
    generator: Generator = field(init=False)
    verifier: AnswerVerifier = field(default_factory=AnswerVerifier)
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)
    memory: MemoryService = field(init=False)
    retrieval: RetrievalEngine = field(init=False)
    query_service: QueryService = field(init=False)
    ingestion_service: object = field(init=False)

    def __post_init__(self) -> None:
        self.planner = QueryPlanner(self.llm)
        self.grader = EvidenceGrader(self.llm, self.settings)
        self.generator = Generator(self.llm)
        self.memory = MemoryService(self.llm, self.index, self.settings, self.sessionmaker)
        self.retrieval = RetrievalEngine(self.settings, self.index, self.sparse, self.reranker)
        self.query_service = QueryService(self)

        from app.services.ingestion_service import IngestionService

        self.ingestion_service = IngestionService(self)


def build_context(settings: Settings) -> AppContext:
    engine = make_engine(settings.database_url)
    index = VectorIndex(settings)
    return AppContext(
        settings=settings,
        engine=engine,
        sessionmaker=make_sessionmaker(engine),
        llm=LLMClient(settings),
        index=index,
        dense=DenseRetriever(index),
    )


async def load_sparse(ctx: AppContext) -> None:
    from app.db.repositories import ChunkRepository
    from app.services.query_service import sparse_documents

    async with ctx.sessionmaker() as session:
        rows = await ChunkRepository(session).all_with_documents()
    ctx.sparse.refresh(sparse_documents(rows))
