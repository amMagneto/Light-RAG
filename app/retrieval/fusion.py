from app.schemas.retrieval import RetrievalCandidate

_SOURCE_PRIORITY = {"dense": 0, "sparse": 1, "metadata": 2, "entity": 3}


def rrf_fuse(
    ranked_lists: dict[str, list[RetrievalCandidate]],
    k: int = 60,
    limit: int = 40,
) -> list[RetrievalCandidate]:
    scores: dict[str, float] = {}
    best: dict[str, RetrievalCandidate] = {}
    sources_seen: dict[str, set[str]] = {}

    for source, candidates in ranked_lists.items():
        for rank, candidate in enumerate(candidates):
            contribution = 1.0 / (k + rank + 1)
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + contribution
            sources_seen.setdefault(candidate.chunk_id, set()).add(source)
            incumbent = best.get(candidate.chunk_id)
            if incumbent is None or candidate.score > incumbent.score:
                best[candidate.chunk_id] = candidate

    fused: list[tuple[float, str, RetrievalCandidate]] = []
    for chunk_id, score in scores.items():
        candidate = best[chunk_id]
        seen = sources_seen[chunk_id]
        primary = min(seen, key=lambda s: _SOURCE_PRIORITY.get(s, 99))
        merged = candidate.model_copy(deep=True)
        merged.score = round(score, 6)
        merged.source = primary  # type: ignore[assignment]
        fused.append((score, chunk_id, merged))

    fused.sort(key=lambda item: item[0], reverse=True)
    return [item[2] for item in fused[:limit]]
