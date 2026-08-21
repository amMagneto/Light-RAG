from app.reasoning.verifier import AnswerVerifier, confidence_band, derive_failure_code
from app.schemas.generation import Citation, Claim, GeneratedAnswer
from app.schemas.query import FailureCode
from app.schemas.retrieval import RetrievalVerdict


def test_failure_codes_map_verdicts_to_distinct_behaviors():
    assert derive_failure_code(RetrievalVerdict.CONTRADICTORY, 8) is FailureCode.CONTRADICTORY_EVIDENCE
    assert derive_failure_code(RetrievalVerdict.AMBIGUOUS, 5) is FailureCode.AMBIGUOUS_QUERY
    assert derive_failure_code(RetrievalVerdict.INSUFFICIENT, 0) is FailureCode.NO_EVIDENCE
    assert derive_failure_code(RetrievalVerdict.INSUFFICIENT, 6) is FailureCode.INSUFFICIENT_RETRIEVAL


def test_confidence_bands_are_evidence_derived_thresholds():
    assert confidence_band(0.9, has_support=True) == "high"
    assert confidence_band(0.6, has_support=True) == "medium"
    assert confidence_band(0.3, has_support=True) == "low"
    assert confidence_band(0.99, has_support=False) == "unsupported"
    assert confidence_band(0.0, has_support=True) == "unsupported"


def test_agreement_signal_pulls_confidence_down():
    verifier = AnswerVerifier()
    citations = [Citation(index=1, chunk_id="c1", document_id="d")]
    claims = [Claim(text="c", citation_indices=[1], confidence=0.8)]
    generated = GeneratedAnswer(answer="a", claims=claims, self_confidence=0.9)

    _, solo, _ = verifier.verify(generated, citations, RetrievalVerdict.SUFFICIENT, agreement=1.0)
    _, split, _ = verifier.verify(generated, citations, RetrievalVerdict.SUFFICIENT, agreement=0.0)
    assert split < solo
    assert split >= solo * 0.84
