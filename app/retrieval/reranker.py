import re
from collections import Counter

from app.retrieval.sparse import _stopwords, tokenize
from app.schemas.retrieval import RetrievalCandidate


class LexicalReranker:
    """Zero-dependency reranker: weighted term overlap + phrase + number bonuses.

    Replaceable by a cross-encoder without changing the interface.
    """

    def score(self, query: str, content: str) -> float:
        query_terms = [t for t in tokenize(query) if t not in _stopwords()]
        if not query_terms:
            query_terms = tokenize(query)
        if not query_terms:
            return 0.0

        content_lower = content.lower()
        term_counts = Counter(tokenize(content))
        total_terms = sum(term_counts.values()) or 1

        matched = 0.0
        number_hits = 0
        for term in query_terms:
            count = term_counts.get(term, 0)
            if count > 0:
                tf = count / total_terms
                matched += min(1.0, 0.5 + tf * 10.0)
                if any(ch.isdigit() for ch in term):
                    number_hits += 1

        coverage = matched / len(query_terms)
        phrase_bonus = 0.15 if " ".join(query_terms[:6]) in content_lower else 0.0
        number_bonus = min(0.15, number_hits * 0.05)
        return round(min(1.0, coverage * 0.9 + phrase_bonus + number_bonus), 4)

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        scored: list[tuple[float, int, RetrievalCandidate]] = []
        for position, candidate in enumerate(candidates):
            scored.append((self.score(query, candidate.content), -position, candidate))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        reranked = []
        for rank, (score, _, candidate) in enumerate(scored[:top_k]):
            updated = candidate.model_copy(deep=True)
            updated.score = score
            reranked.append(updated)
        return reranked
