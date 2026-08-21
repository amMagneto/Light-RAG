import asyncio
import json
import re
import time
from collections import deque
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.exceptions import LLMError, LLMUnavailableError
from app.core.logging import get_logger

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

_TRANSIENT_MARKERS = ("429", "rate limit", "rate_limit", "too many requests", "overloaded", "timeout", "temporarily")


class SlidingWindowLimiter:
    """Client-side RPM guard so we never trip provider rate limits."""

    def __init__(self, rpm_limit: int, window_seconds: float = 60.0):
        self._limit = rpm_limit
        self._window = window_seconds
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._limit <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= self._window:
                    self._calls.popleft()
                if len(self._calls) < self._limit:
                    self._calls.append(now)
                    return
                wait = self._window - (now - self._calls[0]) + 0.01
            log.debug("RPM limit reached; waiting %.2fs", wait)
            await asyncio.sleep(wait)

    def pending(self) -> int:
        return len(self._calls)


class LLMClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._llm = None
        self._limiter = SlidingWindowLimiter(settings.llm_rpm_limit)

    @property
    def available(self) -> bool:
        return self._settings.llm_available

    @property
    def model_name(self) -> str:
        return self._settings.llm_model

    @property
    def limiter(self) -> SlidingWindowLimiter:
        return self._limiter

    def _client(self):
        if not self.available:
            raise LLMUnavailableError("LLM unavailable: GROQ_API_KEY is missing")
        if self._llm is None:
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=self._settings.llm_model,
                temperature=self._settings.llm_temperature,
                api_key=self._settings.groq_api_key,
                max_retries=1,
            )
        return self._llm

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(marker in text for marker in _TRANSIENT_MARKERS)

    async def _invoke_with_retry(self, messages: list, transient_tries: int = 3) -> str:
        backoffs = (3.0, 8.0, 20.0)
        last_error: Exception | None = None
        for attempt in range(transient_tries + 1):
            await self._limiter.acquire()
            try:
                response = await self._client().ainvoke(messages)
                return response.content
            except LLMError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < transient_tries and self._is_transient(exc):
                    delay = backoffs[min(attempt, len(backoffs) - 1)]
                    log.warning("Transient LLM error (%s); retrying in %.0fs", exc, delay)
                    await asyncio.sleep(delay)
                    continue
                raise
        raise LLMError(f"LLM invoke failed after retries: {last_error}")

    async def structured(self, system: str, user: str, schema: Type[T], max_attempts: int = 2) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=1)
        instruction = (
            f"{system}\n\n"
            "Respond with ONLY a single JSON object. No markdown fences, no commentary.\n"
            f"JSON schema:\n{schema_json}"
        )
        messages = [("system", instruction), ("human", user)]
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            corrective_text = ""
            if attempt > 0 and last_error is not None:
                corrective_text = f"\n\nYour previous response was invalid: {last_error}. Return strictly valid JSON."
            try:
                raw_response = await self._invoke_with_retry([*messages, ("human", corrective_text.strip())])
            except LLMUnavailableError:
                raise
            except Exception as exc:
                last_error = exc
                log.warning("LLM invoke failed permanently (attempt %s): %s", attempt + 1, exc)
                continue

            raw = _FENCE.sub("", raw_response.strip()).strip()
            try:
                return schema.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                log.warning("Structured output invalid (attempt %s): %s", attempt + 1, exc)

        raise LLMError(f"Failed to obtain valid structured output from {self._settings.llm_model}: {last_error}")
