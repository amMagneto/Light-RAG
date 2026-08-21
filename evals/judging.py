from app.schemas.retrieval import RetrievalCandidate


def is_relevant(
    candidate: RetrievalCandidate,
    expected_documents: list[str],
    evidence_markers: list[str],
) -> bool:
    """A retrieved chunk counts as relevant iff it comes from an expected document
    AND contains at least one ground-truth marker substring (case-insensitive).

    Rationale: the corpus is authored so each fact lives in specific passages.
    Marker matching gives deterministic, reproducible chunk-level judgments
    without manual labeling. Document filtering prevents cross-file accidents.
    """
    if expected_documents and candidate.document_name not in expected_documents:
        return False
    if not markers_present(candidate.content, evidence_markers):
        return False
    return True


def markers_present(content: str, markers: list[str]) -> bool:
    if not markers:
        return False
    text = content.lower()
    return any(marker.lower() in text for marker in markers)


def build_ground_truth(
    all_chunks: list[RetrievalCandidate],
    expected_documents: list[str],
    evidence_markers: list[str],
) -> set[str]:
    """Corpus-wide relevant chunk-id set for one question.

    Scanning ALL indexed chunks (not just what retrievers returned) makes
    recall a true fraction of total available evidence, including chunks
    no retriever ever surfaced.
    """
    return {
        chunk.chunk_id
        for chunk in all_chunks
        if is_relevant(chunk, expected_documents, evidence_markers)
    }
