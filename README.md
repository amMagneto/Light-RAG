# Light-RAG — Vertical Retrieval Platform

A retrieval system whose final interface happens to be an LLM. Not a chatbot with a vector DB attached: generation is the **last stage** of a typed, inspectable pipeline.

Every boundary is a Pydantic contract. Every query persists a trace. Every quality claim below is a measured number you can reproduce in under a minute.

## Architecture

```text
Query → Plan (classify / resolve / decompose)
      → Hybrid Retrieval (dense e5 + sparse BM25) → RRF fusion → rerank
      → Evidence Grading (sufficient / insufficient / ambiguous / contradictory)
      → [refinement loop, budgeted rewrites]
      → Context Engine (dedup, provenance, citation indices)
      → Grounded Generation (structured JSON: answer + atomic claims + citations)
      → Verification (claims must cite existing evidence; confidence recomputed)
      → Answer + Citations + Trace
```

| Layer | Implementation |
|---|---|
| API | FastAPI, async, versioned `/api/v1` |
| System of record | SQLAlchemy 2.0 async — SQLite by default, PostgreSQL via one env var |
| Vector index | ChromaDB (`intfloat/e5-base-v2`, `passage:`/`query:` prefixes) |
| Sparse retrieval | BM25 (`rank-bm25`) over chunk rows |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Lexical reranker (interface-ready for a cross-encoder) |
| Query intelligence | LLM planner / rewriter — every output Pydantic-validated, heuristic fallback when the LLM is off |
| Grading | LLM grader + rank-prior heuristic fallback with lexical sanity gate |
| Memory | Session history in DB + durable `memories` table, semantically recalled |
| Guardrails | RPM limiter + transient-error backoff on every LLM call; claim-level citation verification |
| Frontend | Streamlit (upload, doc management, chat with citations / confidence / trace) |

Chunking is hierarchy-aware: markdown headings, numbered sections and ALL-CAPS headers build a real `section_path` per chunk (e.g. `Employee Handbook > Benefits > Eligibility`), plus page spans, token counts and content hashes. Chunk size adapts to page count.

## Layout

```text
app/
├── api/routes/        chat, documents, sessions+health
├── core/              config, logging, exceptions
├── schemas/           typed contracts for every stage
├── db/                engine, models, repositories
├── ingestion/         loaders (pdf/docx/txt/md), structure-aware chunker, indexer
├── retrieval/         dense, sparse(BM25), fusion(RRF), reranker
├── reasoning/         llm client (+RPM limiter), planner, grader, verifier
├── context/           evidence dedup + citation builder
├── generation/        grounded prompts + structured generator
├── memory/            recall + validated extraction
└── services/          QueryService, IngestionService, RetrievalEngine
evals/                 golden dataset, judging, metrics, ablation runner, reports
tests/                 50 hermetic tests (unit / integration / eval regression)
docs/                  build vision (historical), implementation status, ADRs
```

## Getting started

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn app.main:app --reload     # terminal 1
streamlit run app.py              # terminal 2
```

`.env` needs `GROQ_API_KEY=`. Without it the system still ingests, searches and grades — `/chat` returns 503 until you add a valid key.

### PostgreSQL

```powershell
docker compose up -d
# set in .env:
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/lightrag
```

Tables are created on startup (`sessions`, `messages`, `documents`, `chunks`, `retrieval_runs`, `memories`).

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/chat` | Full pipeline → answer + claims + citations + confidence + trace |
| `POST /api/v1/search` | Retrieval only → candidates + verdict (no generation) |
| `POST /api/v1/documents` | Upload pdf/docx/txt/md → parse → chunk → embed → READY |
| `GET /api/v1/documents` | List documents with status + chunk counts |
| `DELETE /api/v1/documents/{id}` | Remove document everywhere |
| `POST /api/v1/admin/reindex` | Rebuild vector index from DB chunk rows |
| `GET /api/v1/health` | DB status, LLM availability, corpus stats |
| `GET /api/v1/sessions/{id}` | Full message history |

Interactive docs at `/docs`.

---

## Quality, measured

Golden-dataset evaluation: **4 authored documents** (16 chunks), **27 questions** across **7 query archetypes**, judged by corpus-wide marker matching — a retrieved chunk counts as relevant iff it comes from the expected document *and* contains the fact's marker substring. Ground truth is computed over **all indexed chunks**, so recall is a true fraction of available evidence, not of whatever the retriever happened to return. The grader runs in heuristic mode during evals, isolating retrieval quality from LLM variance at zero API cost.

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

## Test matrix — 50 tests, all green

| component | tests | what they prove |
|---|---:|---|
| structure-aware chunker | 5 | heading→section paths, page spans, packing limits, dynamic sizing tiers |
| BM25 sparse index | 3 | exact-term recall, corpus swap, negative-IDF small-corpus edge case |
| RRF fusion | 3 | rank-based (never raw-score) merging, source priority, truncation |
| lexical reranker | 3 | exact-match dominance, phrase bonus, top-k discipline |
| evidence grader | 2 | SUFFICIENT / INSUFFICIENT verdicts with zero LLM calls |
| answer verifier | 5 | hallucinated citations dropped, coverage×verdict confidence math |
| context engine | 5 | near-duplicate dedup, sequential citation ids, provenance headers |
| RPM limiter | 4 | burst passes free, over-limit blocks, missing key raises cleanly |
| eval logic itself | 5 | marker judging, true-corpus recall, trap exclusion from IR metrics |
| HTTP lifecycle | 4 | upload→search→guarded-chat→delete, format/empty rejections, health |
| concurrency | 2 | parallel searches consistent; upload/delete cycles leave no orphans |
| ingestion edges | 4 | duplicate short-circuit, whitespace reject, unicode roundtrip |
| golden-set regression | 5 | CI-style floors: recall ≥ 80%, MRR ≥ 55%, traps ≥ 60%, grader ≥ 75% |

Hermetic by design: deterministic hash embeddings, temp SQLite + temp Chroma — the suite downloads nothing and spends zero API quota.

## Running the evaluation

```powershell
python -m evals.runner                  # full run -> evals/reports/latest.md + latest.json
python -m evals.runner --subset 8      # quick smoke
python -m evals.runner --from-json     # re-render tables from the last run
python -m pytest tests/eval -q         # regression floors, CI style
```

```powershell
$env:PYTHONPATH="."
python -m pytest tests -q              # the whole 50-test suite
```

## Honest limitations

- Reranker is lexical; swap `LexicalReranker` for a cross-encoder behind the same call.
- No background ingestion worker yet — uploads process synchronously (status endpoint exists for the future worker).
- Contradiction detection requires the LLM grader path.
- Heuristic grader abstains conservatively on zero-lexical-overlap paraphrases (4/27 golden questions) — the LLM grading path handles these.
- Golden corpus is small and clean by construction; expect absolute numbers to compress on messy real-world corpora while relative ablation behavior should hold.
