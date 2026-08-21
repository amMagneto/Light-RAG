import asyncio

from fastapi.testclient import TestClient

from app.db.engine import init_db
from app.main import app
from conftest import make_ctx, upload_and_wait

ENG_MD = b"""# Project Atlas Notes

## Protocols

The quokka payload protocol routes dawn traffic through zephyr relays.
"""

HR_MD = b"""# People Handbook

## Time Off

Employees accrue 25 days of paid vacation per year, in addition to sick leave.
"""


def _boot(tmp_path):
    ctx = make_ctx(tmp_path)
    asyncio.run(init_db(ctx.engine))
    app.state.ctx = ctx
    return ctx, TestClient(app, raise_server_exceptions=True)


def test_acl_lifecycle_open_to_enforced(tmp_path):
    ctx, client = _boot(tmp_path)

    assert client.get("/api/v1/me").json()["mode"] == "open"

    eng = client.post("/api/v1/collections", json={"name": "engineering"}).json()
    hr = client.post("/api/v1/collections", json={"name": "hr"}).json()
    assert eng["id"] and hr["id"]

    upload_and_wait(client, ctx, "atlas.md", ENG_MD, collection="engineering")
    upload_and_wait(client, ctx, "handbook.md", HR_MD, collection="hr")

    admin_key_resp = client.post(
        "/api/v1/admin/api-keys", json={"name": "root", "is_admin": True}
    )
    assert admin_key_resp.status_code == 200
    admin_key = admin_key_resp.json()["api_key"]
    admin_headers = {"X-API-Key": admin_key}

    hr_key = client.post(
        "/api/v1/admin/api-keys",
        json={"name": "hr-team", "collections": ["hr"]},
        headers=admin_headers,
    ).json()["api_key"]
    hr_headers = {"X-API-Key": hr_key}

    me = client.get("/api/v1/me", headers=hr_headers).json()
    assert me["mode"] == "enforced"
    assert me["is_admin"] is False
    assert me["collections"] == ["hr"]

    unauthenticated = client.post("/api/v1/search", json={"query": "quokka"})
    assert unauthenticated.status_code == 401

    bad_key = client.post(
        "/api/v1/search", json={"query": "quokka"}, headers={"X-API-Key": "lrag_bogus"}
    )
    assert bad_key.status_code == 401

    scoped = client.post(
        "/api/v1/search",
        json={"query": "project atlas quokka protocol", "top_k": 5},
        headers=hr_headers,
    ).json()
    assert all(c["document_name"] != "atlas.md" for c in scoped["candidates"]), (
        "unauthorized content must never be retrievable"
    )

    own = client.post(
        "/api/v1/search",
        json={"query": "vacation days PTO", "top_k": 5},
        headers=hr_headers,
    ).json()
    assert any(c["document_name"] == "handbook.md" for c in own["candidates"])

    elevated = client.post(
        "/api/v1/search",
        json={"query": "project atlas quokka protocol", "top_k": 5},
        headers=admin_headers,
    ).json()
    assert any(c["document_name"] == "atlas.md" for c in elevated["candidates"]), (
        "admin must see across collections"
    )

    forbidden_upload = client.post(
        "/api/v1/documents",
        files={"file": ("sneaky.md", b"# x\n\ncontent\n")},
        data={"collection": "engineering"},
        headers=hr_headers,
    )
    assert forbidden_upload.status_code == 403


def test_component_level_pre_retrieval_filters(tmp_path):
    """Pin the guarantee at the retrieval legs themselves: neither dense nor
    sparse can return content from collections outside the scope."""
    ctx, client = _boot(tmp_path)
    eng = client.post("/api/v1/collections", json={"name": "engineering"}).json()
    hr = client.post("/api/v1/collections", json={"name": "hr"}).json()
    upload_and_wait(client, ctx, "atlas.md", ENG_MD, collection="engineering")
    upload_and_wait(client, ctx, "handbook.md", HR_MD, collection="hr")

    dense_scoped = ctx.index.query("quokka protocol atlas", 5, allowed_collection_ids=[hr["id"]])
    dense_all = ctx.index.query("quokka protocol atlas", 5)
    assert all((c.document_name or "") != "atlas.md" for c, _ in dense_scoped), (
        "dense leg must pre-filter unauthorized collections"
    )
    assert any(c.document_name == "atlas.md" for c, _ in dense_all)

    sparse_scoped = ctx.sparse.search("quokka protocol atlas", 5, allowed_collection_ids=[hr["id"]])
    sparse_all = ctx.sparse.search("quokka protocol atlas", 5)
    assert all(c.document_name != "atlas.md" for c in sparse_scoped)
    assert any(c.document_name == "atlas.md" for c in sparse_all)

    empty_scope = ctx.sparse.search("quokka", 5, allowed_collection_ids=[])
    assert empty_scope == []


def test_revoked_key_loses_access(tmp_path):
    ctx, client = _boot(tmp_path)
    assert client.get("/api/v1/admin/api-keys").json()["keys"] == []

    key = client.post("/api/v1/admin/api-keys", json={"name": "temp", "is_admin": True}).json()["api_key"]
    headers = {"X-API-Key": key}
    assert client.get("/api/v1/me", headers=headers).json()["mode"] == "enforced"

    listed = client.get("/api/v1/admin/api-keys", headers=headers).json()["keys"]
    target = next(k for k in listed if k["name"] == "temp")
    revoked = client.delete(f"/api/v1/admin/api-keys/{target['id']}", headers=headers)
    assert revoked.status_code == 200

    assert client.get("/api/v1/me", headers=headers).status_code == 401
