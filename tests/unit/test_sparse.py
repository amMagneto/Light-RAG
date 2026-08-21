from app.retrieval.sparse import SparseIndex


def _docs():
    return [
        {
            "document_id": "d1",
            "document_name": "handbook.pdf",
            "chunks": [
                {"id": "c1", "content": "ISO-27001 certification is required for all vendors.", "chunk_index": 0},
                {"id": "c2", "content": "The cafeteria serves lunch until 14:00.", "chunk_index": 1},
            ],
        },
        {
            "document_id": "d2",
            "document_name": "policy.md",
            "chunks": [
                {"id": "c3", "content": "Vendors must renew ISO-27001 certification annually.", "chunk_index": 0},
            ],
        },
    ]


def test_bm25_finds_exact_terms():
    index = SparseIndex()
    index.refresh(_docs())
    hits = index.search("ISO-27001 certification vendors", k=2)
    ids = [h.chunk_id for h in hits]
    assert set(ids) == {"c1", "c3"}


def test_empty_index_returns_nothing():
    index = SparseIndex()
    assert index.search("anything", k=5) == []


def test_refresh_replaces_corpus():
    index = SparseIndex()
    index.refresh(_docs())
    index.refresh([{"document_id": "d9", "document_name": "x.md", "chunks": [{"id": "z", "content": "zebra", "chunk_index": 0}]}])
    hits = index.search("ISO-27001", k=5)
    assert hits == []
    hits = index.search("zebra", k=5)
    assert [h.chunk_id for h in hits] == ["z"]
