from pydantic import BaseModel, Field

from app.schemas.documents import DocumentMeta
from app.schemas.generation import Citation, Claim
from app.schemas.retrieval import RetrievalCandidate


class TraceInfo(BaseModel):
    run_id: str
    intent: str | None = None
    attempts: int = 1
    verdict: str | None = None
    normalized_query: str | None = None
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    candidate_count: int = 0


class SourceRef(BaseModel):
    citation: Citation
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    confidence: float = 0.0
    uncertainty: str | None = None
    trace: TraceInfo | None = None
    sources: list[SourceRef] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    normalized_query: str | None = None
    verdict: str
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    latency_ms: float = 0.0
    trace: TraceInfo | None = None


class MessageOut(BaseModel):
    role: str
    content: str
    model: str | None = None


class SessionOut(BaseModel):
    session_id: str
    messages: list[MessageOut] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    llm_available: bool
    llm_model: str | None = None
    database: str
    documents_ready: int = 0
    indexed_chunks: int = 0
