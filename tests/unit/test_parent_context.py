from app.context.builder import ContextBuilder
from app.schemas.retrieval import RetrievalCandidate


def _child(doc: str, section: list[str], idx: int, content: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=f"{doc}-{section[-1] if section else 'root'}-{idx}",
        document_id=doc,
        document_name=doc,
        content=content,
        chunk_index=idx,
        page_start=4,
        section_path=list(section),
    )


def test_children_merge_into_one_section_parent():
    builder = ContextBuilder(parent_expansion=True, parent_max_chars=5000)
    children = [
        _child("runbook.md", ["Deploy", "Rollback"], 0, "First paragraph of rollback procedure."),
        _child("runbook.md", ["Deploy", "Rollback"], 1, "Second paragraph with the actual script name."),
        _child("runbook.md", ["Deploy", "Approvals"], 2, "Different section stays separate."),
    ]
    citations, context_text = builder.build(children)

    assert len(citations) == 2, "two sections -> two parent blocks"
    rollback = next(c for c in citations if c.section_path == ["Deploy", "Rollback"])
    assert len(rollback.chunk_ids) == 2, "both siblings merged under one parent"
    assert rollback.page_number == 4
    block = next(line for line in context_text.split("\n") if line.startswith("["))
    assert "runbook.md" in block and "Deploy > Rollback" in block


def test_parent_text_truncates_at_char_cap():
    builder = ContextBuilder(parent_expansion=True, parent_max_chars=200)
    long = " ".join(f"sentence number {i} about deployments" for i in range(60))
    children = [_child("doc.md", ["S"], i, long) for i in range(3)]
    _, context_text = builder.build(children)
    body = context_text.split("\n", 1)[1]
    assert len(body) <= 210
    assert body.count("sentence") >= 1


def test_expansion_disabled_keeps_child_granularity():
    builder = ContextBuilder(parent_expansion=False)
    children = [
        _child("doc.md", ["S"], 0, "alpha"),
        _child("doc.md", ["S"], 1, "beta"),
    ]
    citations, _ = builder.build(children)
    assert len(citations) == 2
    assert all(len(c.chunk_ids) == 1 for c in citations)


def test_relevance_of_parent_is_max_of_children():
    builder = ContextBuilder()
    weak = _child("d", ["S"], 0, "filler")
    strong = _child("d", ["S"], 1, "the exact answer text")
    strong.score = 0.9
    citations, _ = builder.build([weak, strong])
    assert citations[0].relevance_score == 0.9
