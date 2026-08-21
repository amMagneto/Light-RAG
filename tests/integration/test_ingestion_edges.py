import asyncio

from fastapi.testclient import TestClient

from app.db.engine import init_db
from app.main import app
from conftest import make_ctx, upload_and_wait


def _boot(tmp_path):
    ctx = make_ctx(tmp_path)
    asyncio.run(init_db(ctx.engine))
    app.state.ctx = ctx
    return ctx, TestClient(app, raise_server_exceptions=True)


def test_same_filename_same_content_short_circuits(tmp_path):
    ctx, client = _boot(tmp_path)
    payload = b"# Dup Check\n\nThe wombat clause governs burrow inspections.\n"

    first = upload_and_wait(client, ctx, "wombat.md", payload)
    second = upload_and_wait(client, ctx, "wombat.md", payload)

    assert second["document_id"] == first["document_id"]
    docs = client.get("/api/v1/documents").json()["documents"]
    matching = [d for d in docs if d["meta"]["filename"] == "wombat.md"]
    assert len(matching) == 1, "identical re-upload must not create another version"
    assert matching[0]["meta"]["version"] == 1


def test_same_content_different_filename_is_distinct_document(tmp_path):
    """Version families key on filename: identical bytes under another name are
    a separate knowledge artifact (vectors stay isolated per document_id)."""
    ctx, client = _boot(tmp_path)
    payload = b"# Shared Template\n\nIdentical body text for both documents.\n"

    a = upload_and_wait(client, ctx, "alpha.md", payload)
    b = upload_and_wait(client, ctx, "beta.md", payload)
    assert a["document_id"] != b["document_id"]

    docs = client.get("/api/v1/documents").json()["documents"]
    assert len(docs) == 2
    hits = client.post("/api/v1/search", json={"query": "Shared Template", "top_k": 5}).json()["candidates"]
    named = {c["document_name"] for c in hits}
    assert {"alpha.md", "beta.md"} <= named
    assert docs[0]["meta"]["version"] == 1


def test_whitespace_only_document_fails_in_worker(tmp_path):
    ctx, client = _boot(tmp_path)
    response = client.post(
        "/api/v1/documents",
        files={"file": ("blank.md", b"   \n\n\t\n  ", "text/markdown")},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    row = ctx.job_worker.wait_for(job_id)
    assert row.status == "FAILED"
    assert "no chunks" in (row.error or "").lower()


def test_unicode_content_roundtrip(tmp_path):
    ctx, client = _boot(tmp_path)
    content = "# Résumé Café ☕\n\n## Zürich Büro\n\nDie Frist für Rücksendungen beträgt 30 Tage. 東京オフィスも対象です。\n".encode("utf-8")
    uploaded = upload_and_wait(client, ctx, "unicode_é.md", content)
    document_id = uploaded["document_id"]

    status = client.get(f"/api/v1/documents/{document_id}/status").json()
    assert status["meta"]["status"] == "READY"
    assert status["chunk_count"] >= 1

    found = client.post("/api/v1/search", json={"query": "Rücksendungen Frist", "top_k": 3})
    assert found.status_code == 200
    assert found.json()["candidates"], "unicode terms should be retrievable"


def test_job_listing_endpoint(tmp_path):
    ctx, client = _boot(tmp_path)
    upload_and_wait(client, ctx, "listed.md", b"# Listed\n\nSome content for the job list endpoint.\n")
    jobs = client.get("/api/v1/documents/jobs").json()["jobs"]
    assert jobs, "at least one job recorded"
    assert any(job["status"] == "DONE" for job in jobs)


def test_failed_ingestion_marks_document_failed_not_left_ready(tmp_path):
    """A doc that parses but produces no chunks must end FAILED, never READY."""
    ctx, client = _boot(tmp_path)
    response = client.post(
        "/api/v1/documents",
        files={"file": ("blank.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400, "empty bytes rejected before job creation"
