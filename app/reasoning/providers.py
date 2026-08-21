from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import ConfigError


@runtime_checkable
class LLMProvider(Protocol):
    """Every reasoning stage (planner, grader, generator, memory) depends on
    this surface only — swapping Groq for OpenAI/local means implementing it."""

    available: bool
    model_name: str
    limiter: Any

    async def structured(self, system: str, user: str, schema: type[BaseModel], max_attempts: int = 2) -> BaseModel: ...


class NullLLMProvider:
    """Explicit offline provider: everything degrades to heuristics,
    generation endpoints return 503 instead of crashing."""

    available = False
    model_name = "none"
    limiter = None

    async def structured(self, system: str, user: str, schema: type[BaseModel], max_attempts: int = 2) -> BaseModel:
        from app.core.exceptions import LLMUnavailableError

        raise LLMUnavailableError("LLM unavailable: provider is 'null'")


def get_provider(settings: Settings) -> LLMProvider:
    name = (settings.llm_provider or "groq").lower()
    if name == "groq":
        from app.reasoning.llm import LLMClient

        return LLMClient(settings)
    if name == "null":
        return NullLLMProvider()
    raise ConfigError(f"Unknown llm_provider '{name}' (supported: groq, null)")
