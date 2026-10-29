# Light-RAG — Vertical Retrieval Platform

For Nerds

A retrieval system whose final interface happens to be an LLM. Not a chatbot with a vector DB attached: generation is the **last stage** of a typed, inspectable pipeline.

Every boundary is a Pydantic contract. Every query persists a trace. Every quality claim below is a measured number you can reproduce in under a minute.

For Layman

The project is Light-RAG, a self-hosted document Q&A / retrieval platform written in Python. It is not just a chatbot wrapper: the README describes it as a "vertical retrieval platform" where the LLM is the last stage of an inspectable pipeline. You upload documents, ask questions, and get answers with citations, confidence scores, and a per-stage performance trace.

![FastAPI](https://img.shields.io/badge/API-FastAPI-009688) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B) ![ChromaDB](https://img.shields.io/badge/Vector%20store-ChromaDB-5A45FF) ![PostgreSQL](https://img.shields.io/badge/Storage-PostgreSQL%20%7C%20SQLite-4169E1)

## About

Light-RAG is a retrieval platform for document collections that change over time and answers that need to be auditable. It is intended for teams that need to know not only what the model answered, but which passages supported it, how confident the system was, and where each query spent its time.

Retrieval is treated as a first-class, inspectable pipeline rather than a preprocessing step for a chatbot. Planning, retrieval, fusion, reranking, evidence grading, context assembly, generation, and verification each have a typed contract; every query persists a trace with per-stage latency.

### Topics

- **Multi-format ingestion** — PDF, DOCX, TXT, and Markdown, with structure-aware chunking.
- **Versioned ingestion** — Re-uploads supersede prior chunks atomically, with chunk-level diffs between versions.
- **Incremental indexing** — Content-addressed vector IDs ensure only net-new content is embedded.
- **Hybrid retrieval** — Dense `e5-base-v2` and sparse BM25 retrieval, fused with Reciprocal Rank Fusion.
- **Evidence grading** — Classifies context as sufficient, insufficient, ambiguous, or contradictory before generation.
- **Structured generation** — Returns JSON with an answer, atomic claims, citations, and an evidence-derived confidence band.
- **Verified citations** — Drops unsupported claims and citations, then derives confidence from the surviving evidence.
- **Scoped retrieval** — Collections and hashed API keys keep results within authorized content.
- **Per-stage tracing** — Persists a latency breakdown for every stage of every query.
- **Reproducible evaluation** — A 27-question golden set with ablations, hypothesis checks, and regression floors.
- **Self-hosted deployment** — FastAPI and Streamlit over SQLite or PostgreSQL, with no external queue required.

## Contents

- [About](#about)
  - [Topics](#topics)
- [Architecture](#architecture)
- [Design principles](#design-principles)
- [Layout](#layout)
- [Getting started](#getting-started)
  - [PostgreSQL](#postgresql)
  - [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)
- [API surface](#api-surface)
- [Deployment notes](#deployment-notes)
- [Document lifecycle](#document-lifecycle)
- [Measured quality](#measured-quality)
- [Test matrix](#test-matrix)
- [Running the evaluation](#running-the-evaluation)
- [Limitations and trade-offs](#limitations-and-trade-offs)
- [Contributing](#contributing)

## Architecture

```text
Upload → job queued (202) ──► worker thread: parse → structure-aware chunk → hash-diff vs previous version
                              → embed ONLY net-new content (content-addressed keys)
                              → upsert/purge vectors → READY   [versions: v1 ⇄ v2 ⇄ v3]

Query → Authenticate (API key / open mode) → scope to authorized collections
      → Plan (classify / resolve references / decompose)
      → Hybrid Retrieval on small chunks (dense e5 ⊕ sparse BM25, pre-filtered by ACL)
      → RRF fusion → rerank
      → Evidence Grading (sufficient / insufficient / ambiguous / contradictory)
      → [refinement loop, budgeted rewrites]
      → Context Engine (children merge into section parents, dedup, citation indices)
      → Grounded Generation (structured JSON: answer + atomic claims + citations)
      → Verification (claims must cite existing evidence; evidence-derived confidence band)
      → Answer + Citations + Trace (per-stage latencies persisted)
```

| Layer | Implementation |
|---|---|
| API | FastAPI, async, versioned `/api/v1`, background ingestion jobs (202 + polling) |
| Access control | Collections + hashed API keys (`lrag_…`), scoping applied **before** retrieval; open/enforced dual mode (ADR-010) |
| System of record | SQLAlchemy 2.0 async — SQLite by default, PostgreSQL via one env var |
| Versioning | `document_versions` supersedes chain; chunk-level diff between any two versions |
| Vector index | ChromaDB (`intfloat/e5-base-v2`, `passage:`/`query:` prefixes, content-addressed ids) |
| Sparse retrieval | BM25 (`rank-bm25`) over current-version chunks |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Lexical reranker (interface-ready for a cross-encoder) |
| Query intelligence | LLM planner / rewriter behind an `LLMProvider` protocol — Pydantic-validated outputs, heuristic fallback |
| Grading | LLM grader + rank-prior heuristic fallback with lexical sanity gate |
| Guardrails | RPM limiter + transient backoff on every LLM call; claim-level citation verification; distinct failure codes per failure mode |
| Memory | Session history in DB + durable `memories` table, semantically recalled |
| Frontend | Streamlit (upload with job progress, doc/version management, chat with citations / confidence / trace) |

Chunking is hierarchy-aware: markdown headings, numbered sections and ALL-CAPS headers build a real `section_path` per chunk (e.g. `Employee Handbook > Benefits > Eligibility`), plus page spans, token counts and content hashes. Chunk size adapts to page count.

## Design principles

- **Evidence before generation.** The pipeline grades retrieved evidence before it writes an answer. Insufficient evidence produces an abstention, not a guess.
- **Typed boundaries.** Pydantic contracts separate every stage, so components can be tested, replaced, and evaluated independently.
- **Measure, don't assert.** Retrieval quality is reported through ablations, hypothesis checks, and regression floors rather than adjectives.
- **Versions are first-class.** Documents, chunks, and vectors are versioned; re-ingestion embeds only net-new content and removes superseded passages atomically.
- **Scoped by default.** Access control is applied before retrieval, so unauthorized content cannot enter the candidate set.
- **Degrade predictably.** Distinct failure codes, guardrails, and fallbacks make degradation observable rather than silent.

## Layout

```text
app/
├── api/
│   ├── auth.py        identity resolution, open/enforced modes, collection scoping
│   └── routes/        chat+search, documents+jobs, sessions+health+traces, admin (collections/keys)
├── core/              config, logging, exceptions, security (key hashing/generation)
├── schemas/           typed contracts for every stage
├── db/                engine + additive migrations, models, repositories
├── ingestion/         loaders (pdf/docx/txt/md), structure-aware chunker, indexer
├── retrieval/         dense, sparse(BM25), fusion(RRF), reranker
├── reasoning/         providers (protocol + groq/null), llm client (+RPM limiter),
│                      planner, grader, verifier (bands + failure codes)
├── context/           parent-child expansion, evidence dedup, citation builder
├── generation/        grounded prompts + structured generator
├── memory/            recall + validated extraction
└── services/          QueryService, IngestionService (versioned incremental), JobWorker
evals/                 golden dataset, judging, metrics, ablation runner, reports
tests/                 66 hermetic tests (unit / integration / eval regression)
docs/                  build vision (historical), implementation status, ADRs
```

## Getting started

**Prerequisites:** Python with `pip`; Docker is optional and only needed for PostgreSQL.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn app.main:app --reload     # terminal 1
streamlit run app.py              # terminal 2
```

Confirm the API is up:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

`.env` needs `GROQ_API_KEY=`. Without it the system still ingests, searches and grades — `/chat` returns 503 until you add a valid key.

### PostgreSQL

```powershell
docker compose up -d
# set in .env:
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/lightrag
```

Tables are created on startup, including `sessions`, `messages`, `documents`, `document_versions`, `chunks`, `retrieval_runs`, and `memories`.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/chat` returns `503` | `GROQ_API_KEY` is missing or invalid | Set it in `.env` and restart the API |
| Upload stays `QUEUED` | The ingestion worker is not running or is blocked | Check the API logs and restart the process |
| PostgreSQL connection fails | The database is not running or the DSN is wrong | Run `docker compose up -d` and verify `DATABASE_URL` |

## Configuration

Settings are read from `.env` at startup.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | For `/chat` | — | Enables LLM planning, grading, and generation. Without it, ingestion, search, and heuristic grading still run; `/chat` returns `503`. |
| `DATABASE_URL` | No | SQLite | SQLAlchemy async DSN. Use `postgresql+asyncpg://...` for PostgreSQL. |

Additional settings live in `app/core/`.

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/documents` | Upload → **202 + job_id**; ingestion runs in background. Optional `collection` form field |
| `GET /api/v1/documents/jobs/{id}` | Job status: QUEUED → RUNNING → DONE/FAILED |
| `GET /api/v1/documents/{id}/versions` | Version history with content hashes |
| `GET /api/v1/documents/{id}/diff?from_version=&to_version=` | Chunk-level diff: added / removed / unchanged (by content hash) |
| `DELETE /api/v1/documents/{id}` | Full purge: rows, versions, vectors, sparse index |
| `POST /api/v1/admin/reindex` | Rebuild vector index from current DB chunks |
| `POST /api/v1/chat` | Full pipeline → answer + claims + citations + confidence band + failure code + trace |
| `POST /api/v1/search` | Retrieval only → candidates + verdict (no generation) |
| `POST /api/v1/collections` · `GET /api/v1/collections` | Manage collections (open mode or admin) |
| `POST /api/v1/admin/api-keys` | Mint key (`lrag_…`, shown once); scope to collections or grant admin |
| `GET /api/v1/me` | Identity introspection: mode, key, allowed collections |
| `GET /api/v1/traces/{run_id}` | Persisted per-stage latency breakdown for any query |
| `GET /api/v1/sessions/{id}` | Full message history |
| `GET /api/v1/health` | DB status, LLM availability, corpus stats |

**Auth model:** with zero keys in the database the system runs in *open* mode (everything allowed — dev default). Minting the first key flips it to *enforced*: every request needs a valid key and retrieval is scoped to that key's granted collections **before** any search runs. API keys are stored hashed and shown once at creation. See ADR-010.

**Versioning:** the HTTP API is namespaced under `/api/v1`; breaking changes are introduced under a new version prefix.

Interactive docs at `/docs`.

## Deployment notes

- **Run a single API process.** Ingestion uses an in-process worker thread. Additional API processes duplicate workers and can race on the same job; move `JobWorker` to a shared queue before scaling horizontally.
- **Use PostgreSQL for shared state.** SQLite is appropriate for single-node development; PostgreSQL is recommended when multiple processes or hosts need consistent sessions, documents, and jobs after ingestion is moved out of process.
- **Persist both stores.** Back up the relational database and the Chroma data directory together; if the vector store is lost, rebuild it from current chunks with `POST /api/v1/admin/reindex`.
- **Enforce authentication before exposure.** Minting the first API key switches the system from open to enforced mode. Scope each key to the collections it needs and grant admin only when required.
- **Protect credentials and traffic.** API keys are bearer secrets; terminate TLS in front of the API and keep `.env` out of version control.
- **Account for LLM egress.** With `GROQ_API_KEY` set, query text and retrieved passages are sent to Groq for planning, grading, and generation. Use the null provider for retrieval-only deployments or a self-hosted `LLMProvider` for sensitive corpora.

---

## Document lifecycle

```text
policy.md  v1 ──► v2 ──► v3 (current)
```

- **Versioning is content-aware.** Re-uploading the same filename with identical bytes is a no-op; changed bytes create a new version whose chunks supersede the old ones out of retrieval — instantly, atomically. `GET …/diff` answers *"what changed between v1 and v2?"* as data: added/removed passages by section.
- **Indexing is incremental.** Every chunk's vector id is `sha256(document_id : content_hash)`. On re-ingest, unchanged sections keep their vectors untouched — only net-new content is embedded and vanished content is purged. Verified by counting embedding calls across a version bump in tests.
- **Retrieval and generation use different granularities.** Small chunks give precise matching; the context engine then merges siblings back into their section parent so the LLM reads coherent passages instead of fragments. One citation per merged block carries every contributing chunk id.

## Measured quality

Golden-dataset evaluation: **4 authored documents** (16 chunks), **27 questions** across **7 query archetypes** (six retrieval archetypes plus unanswerable traps), judged by corpus-wide marker matching — a retrieved chunk counts as relevant iff it comes from the expected document *and* contains the fact's marker substring. Ground truth is computed over **all indexed chunks**, so recall is a true fraction of available evidence, not of whatever the retriever happened to return. The grader runs in heuristic mode during evals, isolating retrieval quality from LLM variance at zero API cost.

Every number below is reproducible with `python -m evals.runner`; the latest report is written to `evals/reports/latest.md`.

### Overall leaderboard — ablation by configuration

| metric | dense only | sparse (BM25) | RRF fused | full pipeline |
|---|---:|---:|---:|---:|
| **MRR** | **100.0%** | 95.7% | 97.9% | 97.9% |
| recall@1 | 97.9% | 89.1% | 93.8% | 93.8% |
| recall@3 | 100.0% | 100.0% | 100.0% | 100.0% |
| recall@8 | 100.0% | 100.0% | 100.0% | 100.0% |
| precision@1 | 100.0% | 91.3% | 95.8% | 95.8% |
| hit rate@5 | 88.9% | 85.2% | 88.9% | 88.9% |

*How to read this:* MRR answers "how high does the right passage rank" — the difference between position 1 and position 4 is a tight prompt vs a polluted one. Hit rate includes unanswerable traps in the denominator, so misses stay visible instead of being filtered away. Precision@k looks low across the board because ground truths are single-chunk facts fetched at k=8 — 34.7% ≈ the theoretical ceiling for that setting, not a defect.

### Where each retriever earns its keep — MRR by query archetype

| archetype | dense only | sparse (BM25) | RRF fused | full pipeline |
|---|---:|---:|---:|---:|
| exact term (`ISO-27001`, `rollback.sh`) | 100.0% | 100.0% | 100.0% | 100.0% |
| numeric fact (`$29`, `60 days`) | 100.0% | 100.0% | 100.0% | 100.0% |
| paraphrase ("vacation" → PTO section) | 100.0% | 87.5% | 90.0% | 90.0% |
| procedural (rollback, canary steps) | 100.0% | 100.0% | 100.0% | 100.0% |
| comparative (cross-plan / cross-doc) | 100.0% | 83.3% | **100.0%** | **100.0%** |
| multi-constraint (two facts, one question) | 100.0% | 100.0% | 100.0% | 100.0% |

The story is in the deltas: BM25 drops **12.5 points** on paraphrases and **16.7 points** on comparatives; fusion recovers comparatives completely and paraphrases to within one ranking slip of dense. Neither leg is disposable.

### Hypothesis verification — checked programmatically on every run

| hypothesis | claim | verdict |
|---|---|---|
| `H1_exact_term_sparse_wins` | MRR(sparse) ≥ MRR(dense) − 2pts on exact-term queries | **PASS** (100.0% vs 100.0%) |
| `H2_paraphrase_dense_wins` | MRR(dense) > MRR(sparse) on paraphrase queries | **PASS** (100.0% vs 87.5%) |
| `H3_fusion_never_loses_recall` | recall@8(fused) ≥ best single leg − 2pts | **PASS** (100.0% vs 100.0%) |
| `H4_reranker_trades_cleanly` | reranker keeps precision without sacrificing recall | **PASS** (34.7% / 100.0%) |

### Verdict honesty — does it know what it doesn't know?

| check | result | rate |
|---|---|---:|
| Unanswerable traps correctly abstain | 3 / 3 | **100.0%** |
| Answerable → SUFFICIENT when evidence was retrieved | 20 / 24 | **83.3%** |

Traps ask about pet policies, gym benefits and stock tickers that don't exist in the corpus — a system that guesses would hallucinate; this one abstains. The 4 conservative calls are zero-lexical-overlap paraphrases where the heuristic grader abstains rather than risk a wrong answer; the LLM grading path handles them in production.

### Component latency budget (average per query)

| stage | latency |
|---|---:|
| dense search (embed + Chroma) | 124 ms |
| sparse search (BM25) | 1.2 ms |
| RRF fusion + rerank | 0.9 ms |
| LLM planning | ~0.9 s |
| LLM grading | ~2.1 s |
| LLM generation | ~0.9 s |
| **warm chat total (live)** | **~4.4 s** |

### Live end-to-end, real Groq key, paced under RPM limits

| step | observed |
|---|---|
| health | `llm_available: true`, db ok |
| upload security_policy.md | READY, 4 chunks |
| `/search` "vendor signing requirements" | sufficient · correct doc · intent=factual |
| chat: vendor certification | 1 citation, confidence **99.0%**, answer contains ISO-27001 |
| follow-up: "how often are access reviews?" | context resolved → "**quarterly**", confidence **99.0%** |
| session persistence | 4 messages (user+assistant ×2 turns) |

## Test matrix

| component | tests | what they prove |
|---|---:|---|
| structure-aware chunker | 5 | heading→section paths, page spans, packing limits, dynamic sizing tiers |
| BM25 sparse index | 3 | exact-term recall, corpus swap, negative-IDF small-corpus edge case |
| RRF fusion | 3 | rank-based (never raw-score) merging, source priority, truncation |
| lexical reranker | 3 | exact-match dominance, phrase bonus, top-k discipline |
| evidence grader | 2 | SUFFICIENT / INSUFFICIENT verdicts with zero LLM calls |
| answer verifier | 5 | hallucinated citations dropped, coverage×verdict confidence math |
| context engine (parent-child) | 4+5 | sibling merge into parents, char caps, citation indices, dedup, provenance |
| RPM limiter | 4 | burst passes free, over-limit blocks, missing key raises cleanly |
| failure taxonomy + bands | 3 | distinct failure codes per mode, evidence-derived confidence thresholds, source-agreement signal |
| eval logic itself | 5 | marker judging, true-corpus recall, trap exclusion from IR metrics |
| HTTP lifecycle | 4 | upload→job→search→guarded-chat→delete, format/empty rejections, health |
| document versioning | 4 | duplicate no-op, supersession out of retrieval, diff added/removed/unchanged |
| incremental indexing | 1 | embed-call counting proves unchanged chunks never re-embed |
| background jobs | 6 | 202 flow, worker completion, failure marking, unicode, job listing |
| concurrency | 2 | parallel searches consistent; upload/delete cycles leave no orphans |
| ACL | 3 | open→enforced lifecycle, 401/403 paths, pre-retrieval isolation on both legs, revocation |
| golden-set regression | 5 | CI-style floors: recall ≥ 80%, MRR ≥ 55%, traps ≥ 60%, grader ≥ 75% |

Hermetic by design: deterministic hash embeddings, temp SQLite + temp Chroma, dedicated worker loop — the suite downloads nothing and spends zero API quota.

## Running the evaluation

```powershell
python -m evals.runner                  # full run -> evals/reports/latest.md + latest.json
python -m evals.runner --subset 8      # quick smoke
python -m evals.runner --from-json     # re-render tables from the last run
python -m pytest tests/eval -q         # regression floors, CI style
```

```powershell
$env:PYTHONPATH="."
python -m pytest tests -q              # the whole 66-test suite
```

## Limitations and trade-offs

- Reranker is lexical; swap `LexicalReranker` for a cross-encoder behind the same call.
- Background ingestion is a single in-process worker thread — correct for a self-hosted deployment, not a distributed queue (swap `JobWorker` internals for arq/celery without touching the API contract).
- Sparse ACL filtering is post-score (rank dilution, never content leak); dense does true pre-filtering — see ADR-010.
- No user accounts / SSO yet: identity is API-key based; multi-tenant isolation beyond collections is future work.
- No streaming: chat returns after full generation (~4s warm) rather than token-by-token SSE.
- Heuristic grader abstains conservatively on zero-lexical-overlap paraphrases (4/27 golden questions) — the LLM grading path handles these.
- Golden corpus is small and clean by construction; expect absolute numbers to compress on messy real-world corpora while relative ablation behavior should hold.

These are deliberate scope choices, and the major ones are isolated behind interfaces so they can be replaced without changing the API contract.

## Contributing

Contributions should preserve the stage contracts and keep the test suite hermetic.

1. Run the full suite before opening a pull request: `python -m pytest tests -q`.
2. Add or update tests for behavior changes; evaluation floors live in `tests/eval`.
3. Put new providers, rerankers, or queue implementations behind the existing interfaces (`LLMProvider`, reranker, `JobWorker`).
4. Open an issue to discuss larger changes before implementation.

---

[Back to contents](#contents)
