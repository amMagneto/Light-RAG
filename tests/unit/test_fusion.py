from app.retrieval.fusion import rrf_fuse
from app.schemas.retrieval import RetrievalCandidate


def _cand(chunk_id: str, source: str, score: float = 1.0) -> RetrievalCandidate:
    return RetrievalCandidate(chunk_id=chunk_id, document_id="d", content="x", source=source, score=score)


def test_rrf_prefers_items_present_in_both_lists():
    fused = rrf_fuse(
        {
            "dense": [_cand("A", "dense"), _cand("B", "dense"), _cand("C", "dense")],
            "sparse": [_cand("C", "sparse"), _cand("D", "sparse"), _cand("B", "sparse")],
        },
        k=60,
    )
    ids = [c.chunk_id for c in fused]
    assert set(ids) == {"A", "B", "C", "D"}
    assert ids[0] in {"B", "C"}
    b = next(c for c in fused if c.chunk_id == "B")
    a = next(c for c in fused if c.chunk_id == "A")
    assert b.score > a.score


def test_rrf_rank_not_raw_score_dominates():
    fused = rrf_fuse(
        {
            "dense": [_cand("X", "dense", score=99.0), _cand("Y", "dense", score=0.01)],
            "sparse": [],
        },
        k=60,
    )
    assert [c.chunk_id for c in fused][0] == "X"


def test_limit_and_source_merge():
    fused = rrf_fuse(
        {
            "dense": [_cand("A", "dense"), _cand("B", "dense"), _cand("C", "dense")],
            "sparse": [_cand("A", "sparse")],
        },
        limit=2,
    )
    assert len(fused) == 2
    a = fused[0] if fused[0].chunk_id == "A" else None
    assert a is not None and a.source == "dense"
