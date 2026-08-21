# Architecture Decision Records

Why the system is built the way it is. Each record: context → decision → consequences, with measurements where they exist.

---

## ADR-001: ChromaDB as the vector store

**Status:** accepted

**Context:** Need persistent local vector search with metadata filtering, no external service, and a clean swap path to pgvector/Qdrant later.

**Decision:** ChromaDB via `langchain-chroma`, wrapped behind `VectorIndex` (`app/ingestion/indexer.py`). Nothing above the wrapper knows Chroma exists.

**Consequences:** Zero-infrastructure dev setup and per-test temp directories for free. Cost: `RLock` discipline required (an early non-reentrant lock deadlocked delete-during-lookup — found by integration testing). Migration to another store means reimplementing one class.

## ADR-002: Local e5-base-v2 embeddings, injectable interface

**Status:** accepted

**Context:** Hosted embedding APIs add latency, cost, network flakiness, and make hermetic evaluation impossible.

**Decision:** `intfloat/e5-base-v2` runs locally via `sentence-transformers`, honoring e5's `passage:` / `query:` prefix convention at index vs query time. The `VectorIndex` constructor accepts any LangChain `Embeddings` implementation.

**Consequences:** Deterministic offline tests (hash-based dummy embeddings power the whole 50-test suite with zero downloads) and reproducible evals. Cost: ~124 ms/query and a one-time model download; quality ceiling lower than hosted large models — acceptable for the measured corpus (recall@3 100%).

## ADR-003: Hybrid retrieval fused by Reciprocal Rank Fusion

**Status:** accepted

**Context:** Dense embeddings handle paraphrases but miss exact identifiers; BM25 nails identifiers but fails on vocabulary mismatch. Their score scales are incompatible (BM25 unbounded, dense ~1/(1+d)) so naive score addition is meaningless.

**Decision:** Run both legs in parallel, fuse with RRF (`score = Σ 1/(k + rank)`), k=60. Rank-based fusion sidesteps scale normalization entirely.

**Measured:** BM25 loses **12.5 pts** MRR on paraphrase queries and **16.7 pts** on comparatives; dense never drops on exact terms. Fusion recovers comparatives to **100%** while never losing recall against the best single leg (H3 verified every run).

## ADR-004: Lexical reranker now, cross-encoder slot reserved

**Status:** accepted

**Context:** Cross-encoder rerankers are the standard second stage but add model weight, latency, and a download dependency.

**Decision:** `LexicalReranker` (weighted term overlap + phrase + number bonuses) fills the second-stage role; its interface (`rerank(query, candidates, top_k)`) is the contract a future cross-encoder implements without touching callers.

**Consequences:** Zero-dependency default, 0.9 ms fusion+rerank. Measured trade: precision unchanged vs fused (34.7%), recall preserved (100%) — H4 passes. Known ceiling: zero-overlap paraphrases get no lexical signal (see ADR-007).

## ADR-005: SQLAlchemy async repositories; SQLite default, PostgreSQL by env var

**Status:** accepted

**Context:** The database is the system of record (sessions, documents, chunks, traces, memories), but requiring PostgreSQL to boot kills first-run experience.

**Decision:** Async SQLAlchemy 2.0 with repository classes; `DATABASE_URL` selects the engine — SQLite/aiosqlite by default, Postgres via one line. Schema created on startup.

**Consequences:** `docker compose up -d` upgrades to production posture without code changes. SQLite write contention under concurrency bounded with a 20 s busy timeout (verified by parallel-search tests).

## ADR-006: Pydantic contracts at every boundary — including LLM outputs

**Status:** accepted

**Context:** Untyped dicts flowing through a multi-stage pipeline mean every stage hopes the last one formatted things sanely. LLMs especially cannot be trusted to emit arbitrary JSON shapes.

**Decision:** Every stage input/output is a Pydantic model (`app/schemas/`). LLM calls go through `LLMClient.structured()`: schema injected into the prompt, output JSON-validated, one corrective retry, then explicit `LLMError`.

**Consequences:** Malformed planner/grader/generator output can never silently corrupt downstream stages — it becomes a logged error and triggers heuristic fallback instead. This is why the system degrades gracefully rather than hallucinating structure.

## ADR-007: Verdict-based evidence grading with a refinement loop

**Status:** accepted

**Context:** "I retrieved eight documents" ≠ "I found the answer." Retrieval must be able to say insufficient, ambiguous, or contradictory — otherwise refinement can't exist.

**Decision:** Grader returns per-chunk assessments plus an overall verdict; non-sufficient verdicts trigger budgeted rewrites (default 2 attempts). In heuristic mode, relevance blends lexical score with a hybrid rank prior (+0.25 max at rank 1), gated by an explicit `lexical_overlap ≥ 0.22` sanity check before a top-rank hit may alone justify SUFFICIENT.

**Measured:** Traps abstain **100.0%**; answerable-with-evidence graded SUFFICIENT **83.3%** in pure-heuristic mode. The gate exists because raw retrieval scores are incomparable across sources (the original bug: rank-1 evidence rejected 21/24 times) and because paraphrase evidence can carry literally zero lexical overlap — no single threshold separates that from trap noise (measured ceiling ≈ 0.18).

## ADR-008: Client-side RPM limiting and transient backoff

**Status:** accepted

**Context:** One chat turn can cost up to six LLM calls (plan, grade, generate, extract, ≤2 rewrites). Groq free tiers enforce low RPM; a burst of user questions would 429 into broken answers.

**Decision:** Sliding-window limiter (configurable `llm_rpm_limit`, default 25) acquired before every invoke; transient errors (429/overload/timeout) retried with 3→8→20 s backoff, separate from schema-validation retries.

**Consequences:** The pipeline self-paces instead of failing under bursts; unit-tested blocking behavior. Cost: worst-case added latency under sustained load — correct trade for availability.

## ADR-009: Persist a trace row per query

**Status:** accepted

**Context:** "Why did this answer take 4 seconds?" must have an answer that isn't a shrug.

**Decision:** Every chat/search writes a `retrieval_runs` row: normalized query, intent, attempts, verdict, candidate count, confidence, model, and per-stage latency map. `/chat` and `/search` responses embed the same trace.

**Consequences:** Stage-level attribution (e.g., planning 0.9 s vs grading 2.1 s from live runs) without log spelunking; the table doubles as the future metrics dashboard's data source.
