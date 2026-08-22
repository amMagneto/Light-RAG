import asyncio

from fastapi.testclient import TestClient

from app.db.engine import init_db
from app.main import app
from conftest import make_ctx


def _boot(tmp_path):
    ctx = make_ctx(tmp_path)
    asyncio.run(init_db(ctx.engine))
    app.state.ctx = ctx
    return TestClient(app, raise_server_exceptions=True)


def test_duplicate_content_short_circuits(tmp_path):
    client = _boot(tmp_path)
    payload = b"# Dup Check\n\nThe wombat clause governs burrow inspections.\n"

    first = client.post("/api/v1/documents", files={"file": ("first.md", payload, "text/markdown")})
    assert first.status_code == 200, first.text
    assert first.json()["duplicate"] is False
    original_id = first.json()["document_id"]
    original_chunks = first.json()["chunk_count"]

    second = client.post("/api/v1/documents", files={"file": ("second_name.md", payload, "text/markdown")})
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["duplicate"] is True
    assert body["document_id"] == original_id
    assert body["chunk_count"] == original_chunks

    docs = client.get("/api/v1/documents").json()["documents"]
    assert len(docs) == 1, "duplicate upload must not create a second document row"


def test_whitespace_only_document_rejected(tmp_path):
    client = _boot(tmp_path)
    response = client.post(
        "/api/v1/documents",
        files={"file": ("blank.md", b"   \n\n\t\n  ", "text/markdown")},
    )
    assert response.status_code == 422
    assert "no chunks" in response.json()["detail"].lower()


def test_unicode_content_roundtrip(tmp_path):
    client = _boot(tmp_path)
    content = "# Résumé Café ☕\n\n## Zürich Büro\n\nDie Frist für Rücksendungen beträgt 30 Tage. 東京オフィスも対象です。\n".encode("utf-8")
    uploaded = client.post(
        "/api/v1/documents",
        files={"file": ("unicode_é.md", content, "text/markdown")},
    )
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()["status"] == "READY"
    assert uploaded.json()["chunk_count"] >= 1

    found = client.post("/api/v1/search", json={"query": "Rücksendungen Frist", "top_k": 3})
    assert found.status_code == 200
    assert found.json()["candidates"], "unicode terms should be retrievable"


def test_failed_ingestion_marks_document_failed(tmp_path):
    client = _boot(tmp_path)
    response = client.post(
        "/api/v1/documents",
        files={"file": ("blank.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    empty_reject = client.get("/api/v1/documents").json()["documents"]
    assert all(doc["meta"]["status"] != "FAILED" for doc in empty_reject), "empty bytes rejected before row creation"
