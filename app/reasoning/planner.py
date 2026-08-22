from app.core.logging import get_logger
from app.reasoning.llm import LLMClient
from app.schemas.query import QueryPlan, QueryRewrite, QueryRequest

log = get_logger(__name__)

PLANNER_SYSTEM = """You are the query planning module of a retrieval system.
Given the conversation history and the user's question, produce a query plan:
- normalized_query: a standalone version of the question with pronouns and vague references resolved using the conversation. If no resolution is needed, return the question as-is.
- CRITICAL: only resolve references that the question itself points at. Never inject entities, topics, or constraints from earlier turns that the current question does not mention. A follow-up like "how often is X done?" must stay about X, even if the previous turn discussed Y.
- intent: one of factual | procedural | comparative | summary | analytical | temporal
- requires_decomposition: true only if the question clearly needs multiple independent lookups
- subqueries: independent retrieval queries (empty list when not decomposing)
- filters: metadata filters you can infer (e.g. {"document": "handbook"}). Empty object otherwise."""

REWRITE_SYSTEM = """You are the query rewriting module of a retrieval system.
Retrieval failed to find sufficient evidence for the original query.
Rewrite it into a better retrieval query:
- try different vocabulary, synonyms, broader or narrower phrasing
- keep the information need identical
- missing_information: short descriptions of what the previous retrieval lacked"""


class QueryPlanner:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def plan(self, request: QueryRequest, history: list[tuple[str, str]]) -> QueryPlan:
        fallback = QueryPlan(
            original_query=request.query,
            normalized_query=request.query,
            used_history=False,
        )
        if not self._llm.available:
            return fallback
        history_text = "\n".join(f"{role.upper()}: {content[:400]}" for role, content in history[-6:]) or "(none)"
        user = f"Conversation:\n{history_text}\n\nUser question:\n{request.query}"
        try:
            plan = await self._llm.structured(PLANNER_SYSTEM, user, QueryPlan)
            plan.original_query = request.query
            plan.used_history = bool(history)
            return plan
        except Exception as exc:
            log.warning("Query planning fell back to passthrough: %s", exc)
            return fallback

    async def rewrite(self, plan: QueryPlan, rationale: str) -> QueryRewrite:
        fallback = QueryRewrite(original_query=plan.normalized_query, rewritten_query=plan.original_query)
        if not self._llm.available:
            return fallback
        user = (
            f"Original query: {plan.original_query}\n"
            f"Normalized query: {plan.normalized_query}\n"
            f"Why retrieval was judged insufficient: {rationale or 'no relevant evidence found'}\n\n"
            "Produce the rewrite."
        )
        try:
            rewrite = await self._llm.structured(REWRITE_SYSTEM, user, QueryRewrite)
            rewrite.original_query = plan.normalized_query
            if not rewrite.rewritten_query.strip():
                return fallback
            return rewrite
        except Exception as exc:
            log.warning("Query rewriting fell back to passthrough: %s", exc)
            return fallback
