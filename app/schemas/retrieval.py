from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

RetrievalSource = Literal["dense", "sparse", "metadata", "entity"]


class RetrievalCandidate(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str | None = None
    score: float = 0.0
    source: RetrievalSource = "dense"
    content: str
    chunk_index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query: str
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    latency_ms: float = 0.0


class EvidenceAssessment(BaseModel):
    chunk_id: str
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    supports_query: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    lexical_overlap: float = Field(default=0.0, ge=0.0, le=1.0)


class RetrievalVerdict(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    AMBIGUOUS = "ambiguous"
    CONTRADICTORY = "contradictory"


class RetrievalAssessment(BaseModel):
    verdict: RetrievalVerdict
    assessments: list[EvidenceAssessment] = Field(default_factory=list)
    rationale: str = ""
