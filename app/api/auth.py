from dataclasses import dataclass

from fastapi import Request

from app.core.security import hash_key


@dataclass
class Identity:
    mode: str
    key_id: str | None = None
    key_name: str | None = None
    is_admin: bool = False
    allowed_collection_ids: list[str] | None = None


OPEN_IDENTITY = Identity(mode="open", is_admin=True)


def collection_filter(identity: Identity) -> list[str] | None:
    """None means unrestricted (open mode or admin). Otherwise the exact
    collection ids this identity may retrieve from."""
    if identity.mode == "open" or identity.is_admin:
        return None
    return identity.allowed_collection_ids or []


async def resolve_identity(ctx, api_key_plaintext: str | None) -> Identity:
    from app.db.repositories import ApiKeyRepository

    async with ctx.sessionmaker() as session:
        repo = ApiKeyRepository(session)
        total_keys = await repo.count()
        if total_keys == 0:
            return OPEN_IDENTITY
        if not api_key_plaintext:
            return Identity(mode="enforced")
        row = await repo.get_by_hash(hash_key(api_key_plaintext))
        if row is None:
            return Identity(mode="enforced")
        if row.is_admin:
            return Identity(mode="enforced", key_id=row.id, key_name=row.name, is_admin=True)
        grants = await repo.grants_for_key(row.id)
        return Identity(
            mode="enforced",
            key_id=row.id,
            key_name=row.name,
            is_admin=False,
            allowed_collection_ids=grants,
        )


def extract_api_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    authorization = request.headers.get("authorization") or ""
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def get_identity(request: Request) -> Identity:
    from fastapi import HTTPException

    ctx = request.app.state.ctx
    identity = await resolve_identity(ctx, extract_api_key(request))
    if identity.mode == "enforced" and identity.key_id is None:
        raise HTTPException(
            status_code=401,
            detail="Valid API key required (X-API-Key header). Enforcement is active because at least one key exists.",
        )
    return identity
