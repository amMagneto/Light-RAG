import time

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.repositories import ChunkRepository, SessionRepository, TraceRepository
from app.ingestion.indexer import VectorIndex


def sparse_documents(rows: list[tuple[object, str, str | None]]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for chunk, filename, collection_id in rows:
        doc = grouped.setdefault(
            chunk.document_id,
            {
                "document_id": chunk.document_id,
                "document_name": filename,
                "collection_id": collection_id,
                "chunks": [],
            },
        )
        doc["chunks"].append(
            {
                "id": chunk.id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": list(chunk.section_path_json or []),
            }
        )
    return list(grouped.values())


class RetrievalEngine:
    def __init__(
        self,
        settings: Settings,
        index: VectorIndex,
        sparse,
        reranker,
    ):
        self._settings = settings
        self._index = index
        self._sparse = sparse
        self._reranker = reranker

    async def retrieve(self, query: str, top_k: int | None = None, allowed_collection_ids: list[str] | None = None):
        """Returns (fused, reranked, signals). ACL scoping happens BEFORE
        retrieval: dense filters inside Chroma via metadata $in; sparse
        post-filters before returning. signals carries source agreement."""
        import asyncio

        top_k = top_k or self._settings.rerank_top_k
        dense_task = asyncio.to_thread(
            self._index.query, query, self._settings.dense_top_k, allowed_collection_ids
        )
        sparse_task = asyncio.to_thread(
            self._sparse.search, query, self._settings.sparse_top_k, allowed_collection_ids
        )
        dense_hits, sparse_candidates = await asyncio.gather(dense_task, sparse_task)

        from app.retrieval.fusion import rrf_fuse

        fused = rrf_fuse(
            {
                "dense": [candidate for candidate, _ in dense_hits],
                "sparse": sparse_candidates,
            },
            k=self._settings.rrf_k,
            limit=self._settings.candidate_pool,
        )
        reranked = self._reranker.rerank(query, fused, top_k=top_k)

        dense_ids = {candidate.chunk_id for candidate, _ in dense_hits}
        sparse_ids = {candidate.chunk_id for candidate in sparse_candidates}
        if dense_ids and sparse_ids:
            both = sum(1 for c in reranked if c.chunk_id in dense_ids and c.chunk_id in sparse_ids)
            agreement = round(both / len(reranked), 4) if reranked else 0.0
        else:
            agreement = 0.5
        return fused, reranked, {"agreement": agreement}


class QueryService:
    def __init__(self, ctx):
        self._ctx = ctx
        self._settings: Settings = ctx.settings

    @staticmethod
    def _scope(identity) -> list[str] | None:
        from app.api.auth import collection_filter

        return collection_filter(identity) if identity is not None else None

    async def execute_search(self, request, identity=None):
        from app.schemas.responses import SearchResponse, TraceInfo
        from app.schemas.retrieval import RetrievalResult

        stages: dict[str, float] = {}
        started = time.perf_counter()

        async with self._ctx.sessionmaker() as session:
            sessions = SessionRepository(session)
            session_id = None
            if request.session_id:
                session_id = await sessions.get_or_create(request.session_id)

            stage_start = time.perf_counter()
            history = await sessions.history(session_id, limit=6) if session_id else []
            stages["session"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        plan = await self._ctx.planner.plan(request, history)
        stages["planning"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        fused, reranked, signals = await self._ctx.retrieval.retrieve(plan.normalized_query, request.top_k, self._scope(identity))
        stages["retrieval"] = (time.perf_counter() - stage_start) * 1000

        attempts = 1
        stage_start = time.perf_counter()
        assessment = await self._ctx.grader.grade(plan.normalized_query, reranked)
        while (
            assessment.verdict.value != "sufficient"
            and attempts <= self._settings.max_refinement_attempts
            and self._llm_ready()
        ):
            rewrite = await self._ctx.planner.rewrite(plan, assessment.rationale)
            if rewrite.rewritten_query.strip() == plan.normalized_query.strip():
                break
            plan.normalized_query = rewrite.rewritten_query
            fused, reranked, signals = await self._ctx.retrieval.retrieve(rewrite.rewritten_query, request.top_k, self._scope(identity))
            assessment = await self._ctx.grader.grade(rewrite.rewritten_query, reranked)
            attempts += 1
        stages["grading"] = (time.perf_counter() - stage_start) * 1000
        stages["total"] = (time.perf_counter() - started) * 1000

        async with self._ctx.sessionmaker() as session:
            trace = await TraceRepository(session).record(
                session_id=session_id,
                query=request.query,
                normalized_query=plan.normalized_query,
                intent=plan.intent.value,
                attempts=attempts,
                verdict=assessment.verdict.value,
                stages_json={k: round(v, 2) for k, v in stages.items()},
                candidate_count=len(reranked),
                model=None,
            )

        result = RetrievalResult(
            query=request.query,
            candidates=reranked,
            latency_ms=round(stages["total"], 2),
        )
        return SearchResponse(
            query=request.query,
            normalized_query=plan.normalized_query,
            verdict=assessment.verdict.value,
            candidates=result.candidates,
            latency_ms=result.latency_ms,
            trace=TraceInfo(
                run_id=trace.id,
                intent=plan.intent.value,
                attempts=attempts,
                verdict=assessment.verdict.value,
                stage_latencies_ms={k: round(v, 2) for k, v in stages.items()},
                candidate_count=len(reranked),
            ),
        )

    async def execute_chat(self, request, identity=None):
        from app.core.exceptions import LLMError
        from app.schemas.generation import Citation
        from app.schemas.responses import ChatResponse, SourceRef, TraceInfo

        if not self._ctx.llm.available:
            raise LLMError("Generation unavailable: GROQ_API_KEY is missing or expired")

        stages: dict[str, float] = {}
        started = time.perf_counter()

        async with self._ctx.sessionmaker() as session:
            sessions = SessionRepository(session)
            session_id = await sessions.get_or_create(request.session_id)
            stage_start = time.perf_counter()
            history = await sessions.history(session_id, limit=8)
            stages["session"] = (time.perf_counter() - stage_start) * 1000

        memory_notes: list[str] = []
        try:
            memory_notes = await self._ctx.memory.recall(request.query)
        except Exception:
            pass

        stage_start = time.perf_counter()
        plan = await self._ctx.planner.plan(request, history)
        stages["planning"] = (time.perf_counter() - stage_start) * 1000

        attempts = 1
        stage_start = time.perf_counter()
        fused, reranked, signals = await self._ctx.retrieval.retrieve(plan.normalized_query, request.top_k, self._scope(identity))
        assessment = await self._ctx.grader.grade(plan.normalized_query, reranked)
        while assessment.verdict.value != "sufficient" and attempts <= self._settings.max_refinement_attempts:
            rewrite = await self._ctx.planner.rewrite(plan, assessment.rationale)
            if rewrite.rewritten_query.strip() == plan.normalized_query.strip():
                break
            plan.normalized_query = rewrite.rewritten_query
            fused, reranked, signals = await self._ctx.retrieval.retrieve(rewrite.rewritten_query, request.top_k, self._scope(identity))
            assessment = await self._ctx.grader.grade(rewrite.rewritten_query, reranked)
            attempts += 1
        stages["retrieval_grading"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        citations, context_text = self._ctx.context_builder.build(reranked)
        stages["context"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        generated = await self._ctx.generator.generate(plan, context_text, history, memory_notes)
        stages["generation"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        verified, confidence, band = self._ctx.verifier.verify(
            generated, citations, assessment.verdict, agreement=signals.get("agreement", 1.0)
        )
        cited = self._ctx.verifier.filter_citations_for_claims(citations, verified.claims)
        stages["verification"] = (time.perf_counter() - stage_start) * 1000
        stages["total"] = (time.perf_counter() - started) * 1000

        failure_code = None
        conflict = None
        from app.schemas.retrieval import RetrievalVerdict

        if assessment.verdict.value != "sufficient":
            from app.reasoning.verifier import derive_failure_code

            failure_code = derive_failure_code(assessment.verdict, len(reranked))
        if assessment.verdict is RetrievalVerdict.CONTRADICTORY:
            from app.schemas.responses import ConflictInfo

            conflict = ConflictInfo(detected=True, note=assessment.rationale, sources=cited)

        async with self._ctx.sessionmaker() as session:
            sessions = SessionRepository(session)
            await sessions.add_message(session_id, "user", request.query)
            await sessions.add_message(session_id, "assistant", generated.answer, self._ctx.llm.model_name)
            trace = await TraceRepository(session).record(
                session_id=session_id,
                query=request.query,
                normalized_query=plan.normalized_query,
                intent=plan.intent.value,
                attempts=attempts,
                verdict=assessment.verdict.value,
                stages_json={k: round(v, 2) for k, v in stages.items()},
                candidate_count=len(reranked),
                confidence=confidence,
                model=self._ctx.llm.model_name,
            )

        try:
            await self._ctx.memory.extract(session_id, request.query, generated.answer)
        except Exception:
            pass

        by_index = {c.index: c for c in citations}
        sources = [
            SourceRef(
                citation=citation,
                snippet=_snippet(by_index[citation.index].chunk_id, fused),
            )
            for citation in cited
        ]

        return ChatResponse(
            session_id=session_id,
            answer=generated.answer,
            citations=[c.model_copy(deep=True) for c in cited],
            claims=verified.claims,
            confidence=confidence,
            confidence_band=band,
            failure_code=failure_code,
            conflict=conflict,
            uncertainty=verified.uncertainty,
            trace=TraceInfo(
                run_id=trace.id,
                intent=plan.intent.value,
                attempts=attempts,
                verdict=assessment.verdict.value,
                normalized_query=plan.normalized_query,
                stage_latencies_ms={k: round(v, 2) for k, v in stages.items()},
                candidate_count=len(reranked),
            ),
            sources=sources,
        )

    def _llm_ready(self) -> bool:
        return self._ctx.llm.available


def _snippet(chunk_id: str, candidates) -> str:
    for candidate in candidates:
        if candidate.chunk_id == chunk_id:
            content = candidate.content.strip().replace("\n", " ")
            return content[:240] + ("â€¦" if len(content) > 240 else "")
    return ""
