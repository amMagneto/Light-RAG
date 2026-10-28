from pydantic import BaseModel, Field


class Citation(BaseModel):
    index: int
    chunk_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    document_id: str
    document_name: str | None = None
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0


class Claim(BaseModel):
    text: str
    citation_indices: list[int] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GeneratedAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    uncertainty: str | None = None
    self_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
