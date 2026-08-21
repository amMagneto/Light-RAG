from enum import Enum

from pydantic import BaseModel, Field


class FailureCode(str, Enum):
    """Distinct failure modes trigger distinct downstream behavior:
    no_evidence / insufficient_retrieval → refine & re-retrieve,
    ambiguous_query → clarify with the user, contradictory_evidence →
    retrieve competing sources, out_of_scope → refuse."""
    OUT_OF_SCOPE = "out_of_scope"
    NO_EVIDENCE = "no_evidence"
    AMBIGUOUS_QUERY = "ambiguous_query"
    INSUFFICIENT_RETRIEVAL = "insufficient_retrieval"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    GENERATION_FAILURE = "generation_failure"


class QueryIntent(str, Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    SUMMARY = "summary"
    ANALYTICAL = "analytical"
    TEMPORAL = "temporal"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    session_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class QueryPlan(BaseModel):
    original_query: str
    normalized_query: str
    intent: QueryIntent = QueryIntent.FACTUAL
    requires_decomposition: bool = False
    subqueries: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    used_history: bool = False


class QueryRewrite(BaseModel):
    original_query: str
    rewritten_query: str
    rationale: str = ""
    missing_information: list[str] = Field(default_factory=list)
