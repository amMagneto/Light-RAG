import asyncio

from app.core.config import Settings
from app.reasoning.llm import LLMClient, SlidingWindowLimiter


def test_limiter_allows_burst_under_limit():
    async def run():
        limiter = SlidingWindowLimiter(rpm_limit=5, window_seconds=1.0)
        start = asyncio.get_event_loop().time()
        for _ in range(5):
            await limiter.acquire()
        return asyncio.get_event_loop().time() - start

    elapsed = asyncio.run(run())
    assert elapsed < 0.5


def test_limiter_blocks_over_limit():
    async def run():
        limiter = SlidingWindowLimiter(rpm_limit=3, window_seconds=0.5)
        loop = asyncio.get_event_loop()
        start = loop.time()
        for _ in range(3):
            await limiter.acquire()
        fast_time = loop.time() - start
        start = loop.time()
        await limiter.acquire()
        blocked_time = loop.time() - start
        return fast_time, blocked_time

    _, blocked = asyncio.run(run())
    assert blocked >= 0.4


def test_llm_client_unavailable_raises_without_calls():
    settings = Settings(groq_api_key=None)
    client = LLMClient(settings)

    from pydantic import BaseModel

    class Out(BaseModel):
        x: int = 1

    try:
        asyncio.run(client.structured("sys", "user", Out))
        raised = False
    except Exception as exc:
        raised = "GROQ_API_KEY" in str(exc)
    assert raised


def test_zero_limit_disables_throttling():
    async def run():
        limiter = SlidingWindowLimiter(rpm_limit=0, window_seconds=60.0)
        for _ in range(50):
            await limiter.acquire()
        return True

    assert asyncio.run(run())
