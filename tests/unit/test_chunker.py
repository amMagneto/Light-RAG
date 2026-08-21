from app.ingestion.chunker import ChunkData, Section, choose_chunk_config, pack_sections
from app.ingestion.loaders import ParsedDocument, ParsedPage


def _parsed(text: str) -> ParsedDocument:
    from pathlib import Path

    return ParsedDocument(
        path=Path("fake.md"),
        mime_type="text/markdown",
        pages=[ParsedPage(page=None, text=text)],
        page_count=None,
    )


SAMPLE = """# Employee Handbook

Intro paragraph about the handbook and its scope.

## Benefits

Health insurance is provided to all full-time employees.

Dental coverage starts after ninety days of employment.

## Deployment

Deployments require two approvals from the on-call rotation.

Production releases happen every Tuesday.
"""


def test_markdown_headings_build_section_paths():
    from app.ingestion.chunker import build_sections

    sections = build_sections(_parsed(SAMPLE))
    paths = [tuple(s.path) for s in sections]
    assert ("Employee Handbook",) in paths
    assert ("Employee Handbook", "Benefits") in paths
    assert ("Employee Handbook", "Deployment") in paths


def test_content_lands_in_correct_section():
    from app.ingestion.chunker import build_sections

    sections = build_sections(_parsed(SAMPLE))
    deployment_text = "\n".join(
        text for s in sections if s.path and s.path[-1] == "Deployment" for text, _ in s.blocks
    )
    assert "two approvals" in deployment_text
    benefits_text = "\n".join(
        text for s in sections if s.path and s.path[-1] == "Benefits" for text, _ in s.blocks
    )
    assert "Dental coverage" in benefits_text


def test_packing_respects_chunk_size_and_indexes_sequentially():
    section = Section(path=["Doc", "S"])
    paragraph = "word " * 100
    for _ in range(10):
        section.blocks.append((paragraph.strip(), None))
    chunks = pack_sections([section], chunk_size=600, overlap=50)
    assert len(chunks) >= 2
    indexes = [c.chunk_index for c in chunks]
    assert indexes == list(range(len(indexes)))
    for chunk in chunks:
        assert isinstance(chunk, ChunkData)
        assert chunk.content_hash
        assert chunk.token_count > 0


def test_pages_propagate_to_chunks():
    from app.ingestion.chunker import build_sections

    parsed = ParsedDocument(
        path=None,
        mime_type="application/pdf",
        pages=[ParsedPage(page=3, text="# Alpha\n\nalpha body"), ParsedPage(page=7, text="## Beta\n\nbeta body")],
        page_count=2,
    )
    chunks = pack_sections(build_sections(parsed), chunk_size=500, overlap=0)
    by_path = {tuple(c.section_path): c for c in chunks}
    alpha = by_path[("Alpha",)]
    beta = by_path[("Alpha", "Beta")]
    assert alpha.page_start == 3
    assert beta.page_start == 7


def test_dynamic_config_tiers():
    assert choose_chunk_config(1, 10) == (300, 50)
    assert choose_chunk_config(10, 100) == (700, 150)
    assert choose_chunk_config(50, 900) == (1200, 250)
    assert choose_chunk_config(200, 9000) == (2000, 400)
    assert choose_chunk_config(None, 30) == (300, 50)
