from app.schemas.generation import Citation, Claim, GeneratedAnswer
from app.schemas.retrieval import RetrievalVerdict


class AnswerVerifier:
    """Post-generation guardrail: claims may only cite evidence that exists."""

    def verify(
        self,
        generated: GeneratedAnswer,
        citations: list[Citation],
        verdict: RetrievalVerdict,
    ) -> tuple[GeneratedAnswer, float]:
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
        confidence = round(min(1.0, base * coverage * verdict_penalty), 4)

        verified = GeneratedAnswer(
            answer=generated.answer,
            claims=kept_claims,
            uncertainty=generated.uncertainty,
            self_confidence=generated.self_confidence,
        )
        return verified, confidence

    def filter_citations_for_claims(
        self,
        citations: list[Citation],
        claims: list[Claim],
    ) -> list[Citation]:
        used = {idx for claim in claims for idx in claim.citation_indices}
        return [c for c in citations if c.index in used]
