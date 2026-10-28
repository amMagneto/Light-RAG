from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.auth import get_identity
from app.core.security import generate_api_key, hash_key

router = APIRouter(tags=["admin"])


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    is_admin: bool = False
    collections: list[str] = Field(default_factory=list)


def _require_manager(identity) -> None:
    """Open mode bootstraps the first key; afterwards only admins manage access."""
    if identity.mode == "enforced" and not identity.is_admin:
        raise HTTPException(status_code=403, detail="Admin key required")


@router.post("/collections")
async def create_collection(request: Request, payload: CollectionCreate, identity=Depends(get_identity)):
    _require_manager(identity)
    from app.db.repositories import CollectionRepository

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        row = await CollectionRepository(session).get_or_create(payload.name.strip())
    return {"id": row.id, "name": row.name}


@router.get("/collections")
async def list_collections(request: Request, identity=Depends(get_identity)):
    from app.db.repositories import CollectionRepository

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        rows = await CollectionRepository(session).list_all()
    scope = None if identity.is_admin or identity.mode == "open" else set(identity.allowed_collection_ids or [])
    return {
        "collections": [
            {"id": row.id, "name": row.name}
            for row in rows
            if scope is None or row.id in scope
        ]
    }


@router.post("/admin/api-keys")
async def create_api_key(request: Request, payload: ApiKeyCreate, identity=Depends(get_identity)):
    _require_manager(identity)
    if payload.is_admin and payload.collections:
        raise HTTPException(status_code=422, detail="Admin keys cannot be scoped to collections")

    from app.db.repositories import ApiKeyRepository, CollectionRepository

    ctx = request.app.state.ctx
    plaintext = generate_api_key()
    async with ctx.sessionmaker() as session:
        repo = ApiKeyRepository(session)
        collection_ids = []
        for name in payload.collections:
            row = await CollectionRepository(session).get_or_create(name.strip())
            collection_ids.append(row.id)
        await repo.create(
            name=payload.name,
            key_hash=hash_key(plaintext),
            is_admin=payload.is_admin,
            collection_ids=collection_ids,
        )
    return {
        "name": payload.name,
        "api_key": plaintext,
        "is_admin": payload.is_admin,
        "collections": payload.collections,
        "note": "Store this key now — it is hashed at rest and cannot be retrieved again.",
    }


@router.get("/admin/api-keys")
async def list_api_keys(request: Request, identity=Depends(get_identity)):
    _require_manager(identity)
    from app.db.repositories import ApiKeyRepository

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        rows = await ApiKeyRepository(session).list_all()
    return {
        "keys": [
            {
                "id": row.id,
                "name": row.name,
                "is_admin": row.is_admin,
                "revoked": row.revoked,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }


@router.delete("/admin/api-keys/{key_id}")
async def revoke_api_key(request: Request, key_id: str, identity=Depends(get_identity)):
    _require_manager(identity)
    from app.db.repositories import ApiKeyRepository

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        revoked = await ApiKeyRepository(session).revoke(key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail=f"Key {key_id} not found")
    return {"status": "revoked", "key_id": key_id}


@router.get("/me")
async def me(request: Request, identity=Depends(get_identity)):
    from app.db.repositories import CollectionRepository

    ctx = request.app.state.ctx
    async with ctx.sessionmaker() as session:
        rows = await CollectionRepository(session).list_all()
    names = {row.id: row.name for row in rows}
    allowed = None
    if identity.mode == "enforced" and not identity.is_admin:
        allowed = [names.get(cid, cid) for cid in (identity.allowed_collection_ids or [])]
    return {
        "mode": identity.mode,
        "key_name": identity.key_name,
        "is_admin": identity.is_admin or identity.mode == "open",
        "collections": allowed,
    }
