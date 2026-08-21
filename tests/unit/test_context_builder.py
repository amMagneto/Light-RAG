from app.context.builder import ContextBuilder
from app.schemas.retrieval import RetrievalCandidate


def _cand(chunk_id: str, content: str, doc: str = "doc.md", section: list[str] | None = None) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=f"d-{chunk_id}",
        document_name=doc,
        content=content,
        section_path=section or [],
        page_start=3,
    )


def test_identical_chunks_deduplicated():
    builder = ContextBuilder()
    a = _cand("1", "The rollback script is scripts/rollback.sh and it restores the prior version.")
    b = _cand("2", "The rollback script is scripts/rollback.sh and it restores the prior version.")
    citations, context_text = builder.build([a, b])
    assert len(citations) == 1
    assert "[1]" in context_text
    assert "[2]" not in context_text


def test_citation_indices_sequential_from_one():
    builder = ContextBuilder()
    chunks = [_cand(str(i), f"unique content number {i} with distinct words") for i in range(5)]
    citations, _ = builder.build(chunks)
    assert [c.index for c in citations] == [1, 2, 3, 4, 5]
    assert all(c.page_number == 3 for c in citations)


def test_provenance_header_contains_document_and_section():
    builder = ContextBuilder()
    chunk = _cand("x", "body text about deployments", doc="runbook.pdf", section=["Runbook", "Deploy"])
    _, context_text = builder.build([chunk])
    assert "runbook.pdf" in context_text
    assert "Runbook > Deploy" in context_text
    assert "p. 3" in context_text


def test_distinct_similar_length_chunks_survive():
    builder = ContextBuilder()
    a = _cand("a", "Starter plan costs $29 per month with 3 projects included.")
    b = _cand("b", "Growth plan costs $99 per month with 25 projects included.")
    citations, _ = builder.build([a, b])
    assert len(citations) == 2


def test_empty_input_returns_empty():
    builder = ContextBuilder()
    citations, context_text = builder.build([])
    assert citations == []
    assert context_text == ""
