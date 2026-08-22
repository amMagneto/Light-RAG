from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.logging import get_logger
from app.reasoning.llm import LLMClient
from app.schemas.retrieval import (
    EvidenceAssessment,
    RetrievalAssessment,
    RetrievalCandidate,
    RetrievalVerdict,
)

log = get_logger(__name__)


class GraderOutput(BaseModel):
    assessments: list[EvidenceAssessment]
    rationale: str = ""
    contradictory: bool = False


GRADER_SYSTEM = """You are the evidence grading module of a retrieval system.
For each candidate passage, judge how well it supports answering the query.
Return one assessment per chunk: relevance in [0,1], supports_query (boolean), confidence in [0,1].
Set supports_query=false when the passage is merely topically adjacent.
Provide an overall rationale and set contradictory=true only if passages strongly conflict on facts needed to answer."""


def heuristic_relevance(query: str, content: str, rank: int) -> float:
    """LLM-free relevance estimate blending lexical match with a rank prior.

    Raw retrieval scores are intentionally NOT used here: they are
    incomparable across sources (BM25 unbounded, dense 1/(1+d), RRF ~1/60).
    Hybrid rank position carries that signal coherently instead. The additive
    prior is bounded (+0.25 max) and the caller applies a separate lexical
    sanity gate before trusting a top-ranked verdict.
    """
    from app.retrieval.reranker import LexicalReranker

    lexical = LexicalReranker().score(query, content)
    rank_prior = 1.0 / (1.0 + 0.35 * max(0, rank))
    return round(min(1.0, lexical * 0.75 + rank_prior * 0.25), 4)


class EvidenceGrader:
    def __init__(self, llm: LLMClient, settings: Settings):
        self._llm = llm
        self._settings = settings

    def _heuristic_grade(self, query: str, candidates: list[RetrievalCandidate]) -> list[EvidenceAssessment]:
        from app.retrieval.reranker import LexicalReranker

        reranker = LexicalReranker()
        return [
            EvidenceAssessment(
                chunk_id=candidate.chunk_id,
                relevance=round(heuristic_relevance(query, candidate.content, position), 4),
                supports_query=False,
                confidence=0.4,
                lexical_overlap=reranker.score(query, candidate.content),
            )
            for position, candidate in enumerate(candidates)
        ]

    async def grade(self, query: str, candidates: list[RetrievalCandidate]) -> RetrievalAssessment:
        assessments = self._heuristic_grade(query, candidates)
        by_id = {a.chunk_id: a for a in assessments}

        if self._llm.available and candidates:
            import json

            payload = [
                {
                    "chunk_id": c.chunk_id,
                    "excerpt": c.content[:700],
                    "section": " > ".join(c.section_path),
                    "retrieval_score": round(c.score, 4),
                }
                for c in candidates
            ]
            user = f"Query:\n{query}\n\nCandidates:\n{json.dumps(payload, indent=1)}"
            try:
                response = await self._llm.structured(GRADER_SYSTEM, user, GraderOutput)
            except Exception as exc:
                log.warning("LLM grading failed; using heuristics: %s", exc)
            else:
                for item in response.assessments:
                    existing = by_id.get(item.chunk_id)
                    if existing is not None:
                        existing.relevance = max(0.0, min(1.0, item.relevance))
                        existing.supports_query = item.supports_query
                        existing.confidence = max(0.0, min(1.0, item.confidence))
                if response.contradictory:
                    return RetrievalAssessment(
                        verdict=RetrievalVerdict.CONTRADICTORY,
                        assessments=assessments,
                        rationale=response.rationale,
                    )

        relevant = [a for a in assessments if a.relevance >= self._settings.min_relevance]
        strong = [a for a in relevant if a.relevance >= self._settings.strong_relevance]
        first_rank = next(
            (position for position, a in enumerate(assessments) if a.relevance >= self._settings.min_relevance),
            None,
        )

        llm_graded = self._llm.available
        lexical_gate_ok = (
            first_rank is not None
            and assessments[first_rank].lexical_overlap >= self._settings.min_lexical_for_top_rank
        )
        top_rank_hit = (
            relevant is not None
            and bool(relevant)
            and first_rank is not None
            and first_rank <= 3
            and (llm_graded or lexical_gate_ok)
        )

        if strong or len(relevant) >= 2 or top_rank_hit:
            verdict = RetrievalVerdict.SUFFICIENT
            rationale = f"{len(relevant)} relevant chunk(s), best at rank {first_rank + 1}."
        elif not relevant:
            verdict = RetrievalVerdict.INSUFFICIENT
            rationale = "No chunk met the minimum relevance threshold."
        else:
            verdict = RetrievalVerdict.AMBIGUOUS
            rationale = "Only weakly relevant evidence found."
        return RetrievalAssessment(verdict=verdict, assessments=assessments, rationale=rationale)
