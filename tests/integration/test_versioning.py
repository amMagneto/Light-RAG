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


GUIDE_V1 = b"""# Vendor Guide

## Security Requirements

All vendors must hold ISO-27001 certification before onboarding.

## Facilities

The cafeteria stops serving lunch at 14:00 sharp.
"""

GUIDE_V2 = b"""# Vendor Guide

## Security Requirements

All vendors must hold SOC 2 Type II certification before onboarding.

## Facilities

The cafeteria stops serving lunch at 14:00 sharp.

## Travel

Airfare above 500 euros needs manager sign-off.
"""


def test_same_content_upload_is_duplicate_short_circuit(tmp_path):
    ctx, client = _boot(tmp_path)
    first = upload_and_wait(client, ctx, "guide.md", GUIDE_V1)
    second = upload_and_wait(client, ctx, "guide.md", GUIDE_V1)

    assert second["document_id"] == first["document_id"]
    docs = client.get("/api/v1/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["meta"]["version"] == 1
    assert len(docs[0]["versions"]) == 1


def test_changed_content_creates_new_version_and_supersedes_old(tmp_path):
    ctx, client = _boot(tmp_path)
    v1 = upload_and_wait(client, ctx, "guide.md", GUIDE_V1)
    doc_id = v1["document_id"]

    result = client.post(
        "/api/v1/search",
        json={"query": "ISO-27001 vendor certification", "top_k": 4},
    ).json()
    assert any(c["document_name"] == "guide.md" for c in result["candidates"])

    v2 = upload_and_wait(client, ctx, "guide.md", GUIDE_V2)
    assert v2["document_id"] == doc_id

    docs = client.get("/api/v1/documents").json()["documents"]
    assert len(docs) == 1, "same filename stays one document family"
    meta = docs[0]["meta"]
    assert meta["version"] == 2
    assert len(docs[0]["versions"]) == 2

    versions = client.get(f"/api/v1/documents/{doc_id}/versions").json()["versions"]
    assert [v["version"] for v in versions] == [2, 1]

    search_after = client.post(
        "/api/v1/search",
        json={"query": "ISO-27001 vendor certification", "top_k": 4},
    ).json()
    old_hits = [c for c in search_after["candidates"] if "ISO-27001" in c["content"]]
    assert old_hits == [], "superseded chunk content must not be retrievable"

    soc = client.post("/api/v1/search", json={"query": "SOC 2 Type II certification", "top_k": 4}).json()
    assert any("SOC 2 Type II" in c["content"] for c in soc["candidates"])


def test_version_diff_reports_added_removed_unchanged(tmp_path):
    ctx, client = _boot(tmp_path)
    v1 = upload_and_wait(client, ctx, "guide.md", GUIDE_V1)
    upload_and_wait(client, ctx, "guide.md", GUIDE_V2)
    doc_id = v1["document_id"]

    diff = client.get(f"/api/v1/documents/{doc_id}/diff", params={"from_version": 1, "to_version": 2}).json()
    added_text = " ".join(section["excerpt"] for section in diff["added"])
    removed_text = " ".join(section["excerpt"] for section in diff["removed"])

    assert "SOC 2 Type II" in added_text or "Travel" in added_text
    assert "ISO-27001" in removed_text
    assert diff["unchanged"] >= 1, "cafeteria paragraph should carry over"
    assert diff["to_version"] == 2 and diff["from_version"] == 1


def test_incremental_embedding_only_new_content(tmp_path):
    """Re-ingesting a document where one section changed must embed fewer
    chunks than a from-scratch ingest of the same size."""
    ctx, client = _boot(tmp_path)
    index = ctx.index

    upload_and_wait(client, ctx, "guide.md", GUIDE_V1)
    baseline_chunks = index.count()

    embed_calls = {"n": 0}
    original_embed = index._embed_texts

    def counting_embed(texts):
        embed_calls["n"] += len(texts)
        return original_embed(texts)

    index._embed_texts = counting_embed
    try:
        upload_and_wait(client, ctx, "guide.md", GUIDE_V2)
    finally:
        index._embed_texts = original_embed

    # v2 changed one section and added one -> cafeteria text reused (no embed),
    # everything else embedded. Strictly less than full re-embed of all chunks.
    total_v2 = index.count()
    assert total_v2 > baseline_chunks, "new section should add chunks"
    assert embed_calls["n"] < total_v2, (
        f"incremental ingest embedded {embed_calls['n']}/{total_v2} chunks — cache reuse broken"
    )
