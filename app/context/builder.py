from app.retrieval.sparse import tokenize
from app.schemas.generation import Citation
from app.schemas.retrieval import RetrievalCandidate


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ContextBuilder:
    def __init__(self, dedup_threshold: float = 0.8):
        self._dedup_threshold = dedup_threshold

    def _deduplicate(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        kept: list[tuple[set[str], RetrievalCandidate]] = []
        for candidate in candidates:
            tokens = set(tokenize(candidate.content))
            if any(_jaccard(tokens, existing) > self._dedup_threshold for existing, _ in kept):
                continue
            kept.append((tokens, candidate))
        return [candidate for _, candidate in kept]

    def build(
        self,
        candidates: list[RetrievalCandidate],
    ) -> tuple[list[Citation], str]:
        deduped = self._deduplicate(candidates)

        citations: list[Citation] = []
        blocks: list[str] = []
        for position, candidate in enumerate(deduped, start=1):
            citation = Citation(
                index=position,
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                document_name=candidate.document_name,
                page_number=candidate.page_start,
                section_path=list(candidate.section_path),
                relevance_score=round(candidate.score, 4),
            )
            citations.append(citation)
            provenance = " > ".join(candidate.section_path) or "document"
            pages = (
                f"pp. {candidate.page_start}-{candidate.page_end}"
                if candidate.page_start and candidate.page_end and candidate.page_end != candidate.page_start
                else f"p. {candidate.page_start}"
                if candidate.page_start
                else "no page"
            )
            header = f"[{position}] {candidate.document_name or 'unknown'} | {provenance} | {pages}"
            blocks.append(f"{header}\n{candidate.content}")

        return citations, "\n\n".join(blocks)
