from app.retrieval.reranker import LexicalReranker
from app.schemas.retrieval import RetrievalCandidate


def _cand(chunk_id: str, content: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id="d",
        content=content,
        source="dense",
    )


def test_exact_match_scores_higher_than_unrelated():
    reranker = LexicalReranker()
    query = "deployment approvals production"
    good = _cand("g", "Deployments require two approvals from the on-call rotation before production releases.")
    bad = _cand("b", "The office kitchen closes at 3pm on Fridays.")
    ranked = reranker.rerank(query, [bad, good], top_k=2)
    assert ranked[0].chunk_id == "g"
    assert ranked[0].score > ranked[1].score


def test_phrase_bonus_applied():
    reranker = LexicalReranker()
    query = "remote work policy"
    with_phrase = _cand("p", "Our remote work policy applies to all engineers.")
    scattered = _cand("s", "Remote employees asked about the work. The policy is pending.")
    s1 = reranker.score("remote work policy", with_phrase.content)
    s2 = reranker.score("remote work policy", scattered.content)
    assert s1 > s2


def test_top_k_truncation():
    reranker = LexicalReranker()
    candidates = [_cand(str(i), f"topic {i} filler text") for i in range(20)]
    ranked = reranker.rerank("topic 7", candidates, top_k=3)
    assert len(ranked) == 3
    assert ranked[0].chunk_id == "7"
