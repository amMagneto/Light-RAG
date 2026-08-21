from typing import TYPE_CHECKING

from app.schemas.generation import Citation, Claim, GeneratedAnswer
from app.schemas.query import FailureCode
from app.schemas.retrieval import RetrievalVerdict

if TYPE_CHECKING:
    from app.schemas.responses import ConfidenceBand


def derive_failure_code(verdict: RetrievalVerdict, candidate_count: int) -> FailureCode:
    if verdict is RetrievalVerdict.CONTRADICTORY:
        return FailureCode.CONTRADICTORY_EVIDENCE
    if verdict is RetrievalVerdict.AMBIGUOUS:
        return FailureCode.AMBIGUOUS_QUERY
    if candidate_count == 0:
        return FailureCode.NO_EVIDENCE
    return FailureCode.INSUFFICIENT_RETRIEVAL


def confidence_band(confidence: float, has_support: bool) -> "ConfidenceBand":
    if not has_support or confidence <= 0.0:
        return "unsupported"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


class AnswerVerifier:
    """Post-generation guardrail: claims may only cite evidence that exists.
    Confidence is evidence-derived: claim coverage x verdict penalty x
    source-agreement signal — never the LLM's self-reported number alone."""

    def verify(
        self,
        generated: GeneratedAnswer,
        citations: list[Citation],
        verdict: RetrievalVerdict,
        agreement: float = 1.0,
    ) -> tuple[GeneratedAnswer, float, str]:
        valid_indices = {c.index for c in citations}

        kept_claims: list[Claim] = []
        for claim in generated.claims:
            if not claim.citation_indices:
                continue
            if set(claim.citation_indices).issubset(valid_indices) and claim.citation_indices:
                kept_claims.append(claim)

        coverage = len(kept_claims) / max(1, len(generated.claims))
        verdict_penalty = {
            RetrievalVerdict.SUFFICIENT: 1.0,
            RetrievalVerdict.AMBIGUOUS: 0.75,
            RetrievalVerdict.INSUFFICIENT: 0.5,
            RetrievalVerdict.CONTRADICTORY: 0.4,
        }[verdict]

        base = generated.self_confidence
        if kept_claims:
            base = (base + sum(c.confidence for c in kept_claims) / len(kept_claims)) / 2
        agreement_factor = max(0.0, min(1.0, agreement))
        confidence = round(
            min(1.0, base * coverage * verdict_penalty * (0.85 + 0.15 * agreement_factor)), 4
        )
        band = confidence_band(confidence, has_support=bool(kept_claims and citations))

        verified = GeneratedAnswer(
            answer=generated.answer,
            claims=kept_claims,
            uncertainty=generated.uncertainty,
            self_confidence=generated.self_confidence,
        )
        return verified, confidence, band

    def filter_citations_for_claims(
        self,
        citations: list[Citation],
        claims: list[Claim],
    ) -> list[Citation]:
        used = {idx for claim in claims for idx in claim.citation_indices}
        return [c for c in citations if c.index in used]
