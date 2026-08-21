import asyncio

from fastapi.testclient import TestClient

from app.db.engine import init_db
from app.main import app
from conftest import make_ctx


def _boot(tmp_path):
    ctx = make_ctx(tmp_path)
    asyncio.run(init_db(ctx.engine))
    app.state.ctx = ctx
    return ctx, TestClient(app, raise_server_exceptions=True)


GUIDE_MD = b"""# Vendor Guide

## Security Requirements

All vendors must hold ISO-27001 certification before onboarding.

## Facilities

The cafeteria stops serving lunch at 14:00 sharp.
"""


def test_document_lifecycle_search_and_chat_guard(tmp_path):
    _, client = _boot(tmp_path)

    response = client.post(
        "/api/v1/documents",
        files={"file": ("guide.md", GUIDE_MD, "text/markdown")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "READY"
    assert body["chunk_count"] >= 2
    document_id = body["document_id"]

    response = client.get("/api/v1/documents")
    assert any(d["meta"]["id"] == document_id for d in response.json()["documents"])

    response = client.get(f"/api/v1/documents/{document_id}/status")
    assert response.status_code == 200
    assert response.json()["meta"]["status"] == "READY"

    response = client.post("/api/v1/search", json={"query": "ISO-27001 vendor certification", "top_k": 4})
    assert response.status_code == 200, response.text
    search = response.json()
    assert search["candidates"], "expected retrieval hits"
    assert search["candidates"][0]["document_name"] == "guide.md"
    assert search["verdict"] in {"sufficient", "insufficient", "ambiguous"}
    assert search["trace"]["run_id"]

    response = client.post("/api/v1/chat", json={"query": "what do vendors need?"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]

    response = client.delete(f"/api/v1/documents/{document_id}")
    assert response.status_code == 200
    response = client.get(f"/api/v1/documents/{document_id}/status")
    assert response.status_code == 404


def test_unsupported_format_rejected(tmp_path):
    _, client = _boot(tmp_path)
    response = client.post(
        "/api/v1/documents",
        files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_empty_upload_rejected(tmp_path):
    _, client = _boot(tmp_path)
    response = client.post("/api/v1/documents", files={"file": ("empty.md", b"", "text/markdown")})
    assert response.status_code == 400


def test_health_reports_components(tmp_path):
    ctx, client = _boot(tmp_path)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "ok"
    assert body["llm_available"] is False
