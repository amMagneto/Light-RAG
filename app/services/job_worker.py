import asyncio
import threading
import time

from app.core.logging import get_logger
from app.db.engine import make_engine, make_sessionmaker
from app.db.repositories import IngestionJobRepository

log = get_logger(__name__)


class IngestionJobWorker:
    """Background ingestion on a dedicated loop thread.

    POST /documents returns 202 immediately; parse/chunk/embed/index happens on
    the worker loop. Status is polled from the database, so callers on any event
    loop (uvicorn's or TestClient's per-request portals) stay correct.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._settings = ctx.settings
        self._engine = make_engine(self._settings.database_url)
        self._sessionmaker = make_sessionmaker(self._engine)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="ingestion-worker")
        self._thread.start()

    def _run_coro(self, coro, timeout: float = 60.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _create_job(self, filename: str):
        async with self._sessionmaker() as session:
            return await IngestionJobRepository(session).create(filename)

    async def _get_job(self, job_id: str):
        async with self._sessionmaker() as session:
            return await IngestionJobRepository(session).get(job_id)

    async def _set_status(self, job_id: str, status: str, **kwargs):
        async with self._sessionmaker() as session:
            await IngestionJobRepository(session).set_status(job_id, status, **kwargs)

    async def _process(self, job_id: str, filename: str, payload: bytes, collection_id: str | None = None) -> None:
        try:
            await self._set_status(job_id, "RUNNING")
            result = await self._ctx.ingestion_service.ingest(filename, payload, collection_id)
            await self._set_status(job_id, "DONE", document_id=result["document_id"])
            log.info("job %s DONE: %s", job_id, result)
        except Exception as exc:
            log.exception("job %s FAILED", job_id)
            await self._set_status(job_id, "FAILED", error=str(exc))

    def submit(self, filename: str, payload: bytes, collection_id: str | None = None) -> str:
        """Queue a job; returns job_id immediately."""
        job = self._run_coro(self._create_job(filename))
        asyncio.run_coroutine_threadsafe(
            self._process(job.id, filename, payload, collection_id), self._loop
        )
        log.info("job %s queued for %s (%d bytes)", job.id, filename, len(payload))
        return job.id

    def wait_for(self, job_id: str, timeout: float = 300.0):
        """Block until DONE/FAILED; returns final job row."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = self._run_coro(self._get_job(job_id))
            if row is not None and row.status in {"DONE", "FAILED"}:
                return row
            time.sleep(0.05)
        raise TimeoutError(f"job {job_id} did not finish in {timeout}s")

    def aclose(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def active_count(self) -> int:
        return threading.active_count()
