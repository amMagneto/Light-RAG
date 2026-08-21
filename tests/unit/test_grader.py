import asyncio

from app.core.config import Settings
from app.reasoning.grader import EvidenceGrader
from app.reasoning.llm import LLMClient
from app.schemas.retrieval import RetrievalCandidate, RetrievalVerdict


def _settings() -> Settings:
    return Settings(groq_api_key=None)


def _cand(chunk_id: str, content: str) -> RetrievalCandidate:
    return RetrievalCandidate(chunk_id=chunk_id, document_id="d", content=content, source="dense")


def test_sufficient_when_strong_lexical_match():
    grader = EvidenceGrader(LLMClient(_settings()), _settings())
    query = "How many approvals are required for production deployments?"
    candidates = [
        _cand("c1", "Production deployments require two approvals from senior engineers before release."),
        _cand("c2", "Deployments to staging are automatic on merge."),
    ]
    assessment = asyncio.run(grader.grade(query, candidates))
    assert isinstance(assessment.verdict, RetrievalVerdict)
    assert assessment.verdict in {RetrievalVerdict.SUFFICIENT, RetrievalVerdict.AMBIGUOUS}
    assert assessment.assessments[0].relevance > assessment.assessments[1].relevance


def test_insufficient_when_nothing_matches():
    grader = EvidenceGrader(LLMClient(_settings()), _settings())
    assessment = asyncio.run(
        grader.grade("quantum entanglement budget", [_cand("c1", "The cafeteria menu rotates weekly.")])
    )
    assert assessment.verdict == RetrievalVerdict.INSUFFICIENT
