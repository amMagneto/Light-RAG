import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from app.db.engine import init_db
from app.main import app
from conftest import make_ctx

UNIQUE_MD = b"""# Concurrency Fixture

## Alpha Section

The zephyr engine processes quokka payloads at dawn.

## Beta Section

Fallback routing uses the marmot protocol during storms.
"""


def _boot(tmp_path):
    ctx = make_ctx(tmp_path)
    asyncio.run(init_db(ctx.engine))
    app.state.ctx = ctx
    return ctx


def test_parallel_searches_are_consistent(tmp_path):
    _boot(tmp_path)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            upload = await client.post(
                "/api/v1/documents",
                files={"file": ("fixture.md", UNIQUE_MD, "text/markdown")},
            )
            assert upload.status_code == 200, upload.text

            async def search(query: str):
                response = await client.post(
                    "/api/v1/search",
                    json={"query": query, "top_k": 3},
                )
                assert response.status_code == 200
                return response.json()

            results = await asyncio.gather(
                search("quokka payload processing"),
                search("marmot fallback routing"),
                search("zephyr engine"),
                search("quokka payload processing"),
                search("storm protocol"),
                search("dawn processing"),
            )

        return results

    results = asyncio.run(scenario())

    for result in results:
        assert result["candidates"], "expected hits in a tiny corpus"

    quokka_top = {json.dumps(sorted(c["chunk_id"] for c in r["candidates"][:1])) for r in (results[0], results[3])}
    assert len(quokka_top) == 1, "identical concurrent queries should rank identically"


def test_sequential_upload_delete_cycle(tmp_path):
    ctx = _boot(tmp_path)
    client = TestClient(app, raise_server_exceptions=True)

    for cycle in range(3):
        uploaded = client.post(
            "/api/v1/documents",
            files={"file": (f"cycle{cycle}.md", UNIQUE_MD, "text/markdown")},
        )
        assert uploaded.status_code == 200, uploaded.text
        doc_id = uploaded.json()["document_id"]
        deleted = client.delete(f"/api/v1/documents/{doc_id}")
        assert deleted.status_code == 200
        status = client.get(f"/api/v1/documents/{doc_id}/status")
        assert status.status_code == 404

    assert ctx.index.count() == 0, "vector index should be empty after deletes"
