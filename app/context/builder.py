from collections import OrderedDict

from app.retrieval.sparse import tokenize
from app.schemas.generation import Citation
from app.schemas.retrieval import RetrievalCandidate


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _truncate_at_paragraph(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind("\n\n")
    if boundary > limit // 2:
        return cut[:boundary]
    sentence = cut.rfind(". ")
    return cut[: sentence + 1] if sentence > limit // 2 else cut


class ContextBuilder:
    """Packs retrieved evidence for generation.

    Parent-child retrieval: retrieval precision comes from small chunks, but the
    LLM reads merged section-level parents. Children are grouped by
    (document, section path); each group becomes one citation whose text is the
    concatenated siblings, capped at parent_max_chars.
    """

    def __init__(self, dedup_threshold: float = 0.8, parent_expansion: bool = True, parent_max_chars: int = 2400):
        self._dedup_threshold = dedup_threshold
        self._parent_expansion = parent_expansion
        self._parent_max_chars = parent_max_chars

    def _deduplicate(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        kept: list[tuple[set[str], RetrievalCandidate]] = []
        for candidate in candidates:
            tokens = set(tokenize(candidate.content))
            if any(_jaccard(tokens, existing) > self._dedup_threshold for existing, _ in kept):
                continue
            kept.append((tokens, candidate))
        return [candidate for _, candidate in kept]

    def _group_into_parents(self, candidates: list[RetrievalCandidate]) -> list[list[RetrievalCandidate]]:
        groups: OrderedDict[tuple, list[RetrievalCandidate]] = OrderedDict()
        for candidate in candidates:
            key = (candidate.document_id, tuple(candidate.section_path))
            groups.setdefault(key, []).append(candidate)
        for children in groups.values():
            children.sort(key=lambda c: c.chunk_index)
        return list(groups.values())

    def build(self, candidates: list[RetrievalCandidate]) -> tuple[list[Citation], str]:
        deduped = self._deduplicate(candidates)
        parent_groups = (
            self._group_into_parents(deduped) if self._parent_expansion else [[c] for c in deduped]
        )

        citations: list[Citation] = []
        blocks: list[str] = []
        for position, children in enumerate(parent_groups, start=1):
            first = children[0]
            merged_text = "\n\n".join(child.content for child in children)
            merged_text = _truncate_at_paragraph(merged_text, self._parent_max_chars)

            pages = sorted(set(c.page_start for c in children if c.page_start is not None))
            citation = Citation(
                index=position,
                chunk_id=first.chunk_id,
                chunk_ids=[child.chunk_id for child in children],
                document_id=first.document_id,
                document_name=first.document_name,
                page_number=pages[0] if pages else None,
                section_path=list(first.section_path),
                relevance_score=round(max(child.score for child in children), 4),
            )
            citations.append(citation)

            provenance = " > ".join(first.section_path) or "document"
            if len(pages) >= 2:
                span = f"pp. {pages[0]}-{pages[-1]}"
            elif pages:
                span = f"p. {pages[0]}"
            else:
                span = "no page"
            header = f"[{position}] {first.document_name or 'unknown'} | {provenance} | {span}"
            blocks.append(f"{header}\n{merged_text}")

        return citations, "\n\n".join(blocks)
