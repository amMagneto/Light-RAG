from app.core.exceptions import LLMError
from app.generation.prompts import GENERATION_SYSTEM, build_generation_user
from app.reasoning.llm import LLMClient
from app.schemas.generation import GeneratedAnswer
from app.schemas.query import QueryPlan


class Generator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def generate(
        self,
        plan: QueryPlan,
        context_text: str,
        history: list[tuple[str, str]],
        memory_notes: list[str] | None = None,
    ) -> GeneratedAnswer:
        if not self._llm.available:
            raise LLMError("Generation unavailable: GROQ_API_KEY is missing or expired")
        user = build_generation_user(
            query=plan.normalized_query,
            context_text=context_text,
            history=history,
            memory_notes=memory_notes or [],
        )
        return await self._llm.structured(GENERATION_SYSTEM, user, GeneratedAnswer)
