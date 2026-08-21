from evals.judging import build_ground_truth, is_relevant
from evals.metrics import aggregate, mrr_from_flags, recall_at_k
from app.schemas.retrieval import RetrievalCandidate


def _cand(chunk_id: str, doc: str, content: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="d",
        document_name=doc,
        content=content,
    )


def test_marker_relevance_requires_document_match():
    chunk = _cand("c1", "other.md", "Vendors need ISO-27001 certification.")
    assert not is_relevant(chunk, ["security_policy.md"], ["ISO-27001"])
    ok = _cand("c2", "security_policy.md", "Vendors need ISO-27001 certification.")
    assert is_relevant(ok, ["security_policy.md"], ["ISO-27001"])


def test_ground_truth_scans_whole_corpus():
    chunks = [
        _cand("a", "x.md", "The notice period of 60 days applies."),
        _cand("b", "x.md", "Unrelated cafeteria text."),
        _cand("c", "y.md", "Another mention of the 60 days notice."),
    ]
    relevant = build_ground_truth(chunks, ["x.md"], ["60 days"])
    assert relevant == {"a"}


def test_recall_is_true_fraction_of_corpus_evidence():
    flags = [True, False, True, False]
    assert recall_at_k(flags, 4) == 1.0
    assert recall_at_k(flags, 1) == 0.5
    assert recall_at_k([False] * 3, 3) is None


def test_mrr_prefers_early_hits():
    assert mrr_from_flags([False, True, False]) == 0.5
    assert mrr_from_flags([True]) == 1.0
    assert mrr_from_flags([False, False]) == 0.0


def test_aggregate_excludes_traps_from_ir_but_counts_misses():
    per_query = {
        "answerable_hit": [True, True],
        "answerable_miss": [False, False],
    }
    result = aggregate(per_query, ks=(2,))
    assert result["judged_queries"] == 1
    assert result["missed_queries"] == 1
    assert result["hit_rate@2"] == 0.5
