from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def make_engine(url: str) -> AsyncEngine:
    connect_args = {"timeout": 20} if url.startswith("sqlite") else {}
    return create_async_engine(url, echo=False, future=True, connect_args=connect_args)


def make_sessionmaker(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, expire_on_commit=False)


async def init_db(engine) -> None:
    from app.db import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _add_missing_columns(conn)


_ADDITIVE_COLUMNS = {
    "chunks": {
        "content_hash": "VARCHAR(64)",
        "version_id": "VARCHAR(36)",
        "is_current": "BOOLEAN DEFAULT 1",
    },
    "documents": {
        "content_hash": "VARCHAR(64)",
        "storage_path": "VARCHAR(1024)",
        "collection_id": "VARCHAR(36)",
    },
}


async def _add_missing_columns(conn) -> None:
    """create_all never alters existing tables; add newer columns in place."""
    from sqlalchemy import text

    for table, columns in _ADDITIVE_COLUMNS.items():
        existing = await conn.execute(
            text(f"PRAGMA table_info({table})") if engine_is_sqlite(conn) else text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = :t"
            ),
            {"t": table} if not engine_is_sqlite(conn) else {},
        )
        present = {row[1] if engine_is_sqlite(conn) else row[0] for row in existing}
        for column, ddl in columns.items():
            if column not in present:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def engine_is_sqlite(conn) -> bool:
    return conn.dialect.name == "sqlite"
