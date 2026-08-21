from app.reasoning.verifier import AnswerVerifier
from app.schemas.generation import Citation, Claim, GeneratedAnswer
from app.schemas.retrieval import RetrievalVerdict


def _citation(index: int) -> Citation:
    return Citation(
        index=index,
        chunk_id=f"chunk-{index}",
        document_id="d1",
        document_name="doc.pdf",
        page_number=index,
    )


def _generated(*claims: Claim, confidence: float = 0.8) -> GeneratedAnswer:
    return GeneratedAnswer(answer="answer text", claims=list(claims), self_confidence=confidence)


def test_claims_citing_missing_evidence_are_dropped():
    verifier = AnswerVerifier()
    citations = [_citation(1), _citation(2)]
    claims = [
        Claim(text="supported claim", citation_indices=[1], confidence=0.9),
        Claim(text="hallucinated citation", citation_indices=[99], confidence=0.9),
    ]
    verified, confidence = verifier.verify(_generated(*claims), citations, RetrievalVerdict.SUFFICIENT)
    assert len(verified.claims) == 1
    assert verified.claims[0].text == "supported claim"
    assert 0 < confidence < 1


def test_coverage_reduces_confidence():
    verifier = AnswerVerifier()
    citations = [_citation(1)]
    good = Claim(text="ok", citation_indices=[1], confidence=0.8)
    bad = Claim(text="bad", citation_indices=[7], confidence=0.8)

    _, full = verifier.verify(_generated(good), citations, RetrievalVerdict.SUFFICIENT)
    _, partial = verifier.verify(_generated(good, bad), citations, RetrievalVerdict.SUFFICIENT)
    assert partial < full


def test_verdict_penalty_applied():
    verifier = AnswerVerifier()
    citations = [_citation(1)]
    claims = [Claim(text="c", citation_indices=[1], confidence=0.9)]
    _, sufficient = verifier.verify(_generated(*claims), citations, RetrievalVerdict.SUFFICIENT)
    _, contradictory = verifier.verify(_generated(*claims), citations, RetrievalVerdict.CONTRADICTORY)
    assert contradictory < sufficient * 0.6


def test_filter_citations_keeps_only_claimed():
    verifier = AnswerVerifier()
    citations = [_citation(1), _citation(2), _citation(3)]
    claims = [Claim(text="c", citation_indices=[1, 3])]
    kept = verifier.filter_citations_for_claims(citations, claims)
    assert [c.index for c in kept] == [1, 3]


def test_claims_without_citations_dropped_entirely():
    verifier = AnswerVerifier()
    citations = [_citation(1)]
    claims = [Claim(text="uncited", citation_indices=[])]
    verified, _ = verifier.verify(_generated(*claims), citations, RetrievalVerdict.SUFFICIENT)
    assert verified.claims == []
