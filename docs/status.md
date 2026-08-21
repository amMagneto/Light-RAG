# Implementation Status

Ledger mapping `build_vision.md` (the original wishlist) to what actually shipped. Every ✅ links to code or measured evidence — no aspirational checkmarks.

| Vision item | Status | Evidence |
|---|---|---|
| Answer provenance — every answer cites source + page | ✅ Shipped | `Citation` schema (`app/schemas/generation.py`) carries chunk id, page number, section path, relevance; claims must cite existing evidence (`app/reasoning/verifier.py`); live E2E returned grounded ISO-27001 citation at 99% confidence |
| Confidence scoring with honest fallback | ✅ Shipped | Retrieval verdicts (`sufficient / insufficient / ambiguous / contradictory`), `uncertainty` field on answers, `/chat` returns 503 when LLM unavailable; unanswerable traps abstain **100.0%** of the time |
| Multi-document cross-referencing | ✅ Shipped | Corpus-wide hybrid retrieval across all indexed documents; comparative archetype scores **MRR 100.0%** (cross-plan, cross-doc questions) |
| Hybrid search (BM25 alongside vectors), cited as deliberate choice | ✅ Shipped | ADR-003; ablation-measured: sparse drops 12.5 pts on paraphrases, dense drops nothing on exact terms, RRF never loses recall (H1–H4 all PASS) |
| Evaluation harness ("what makes it S-tier") | ✅ Shipped | `evals/` — 27-question golden dataset over 7 archetypes, corpus-wide marker judging, 4-config ablation, hypothesis verification, CI-style regression floors; see `README.md § Quality, measured` |
| Async ingestion pipeline (background indexing) | ◐ Partial | Ingestion is synchronous today, but the `DocumentStatus` lifecycle (`UPLOADED→PARSING→CHUNKING→INDEXING→READY/FAILED`) and `GET /documents/{id}/status` endpoint exist precisely so a background worker can slot in without API changes |
| Observability — know why an answer took 4.2s | ◐ Partial | Every query persists a trace row (`retrieval_runs`): per-stage latencies, intent, attempts, verdict, candidate count, normalized query. Missing piece: an aggregate metrics dashboard |
| ADR section signaling engineering maturity | ✅ Shipped | `docs/architecture.md` |
| Document versioning (`v1 → v2 → v3`, "what changed?") | ✅ Shipped | `document_versions` table with supersedes chain; same filename + changed hash = new version, old chunks superseded out of retrieval; chunk-level diff endpoint `GET /documents/{id}/diff`; golden test: ISO-27001 v1 becomes unretrievable after SOC 2 v2 lands |
| Incremental indexing (only re-embed changed chunks) | ✅ Shipped | Content-addressed vector keys (`sha256(doc:chunk_hash)`) + upsert/delete sync; verified by counting embed calls across a version bump — unchanged sections never re-embed |
| Parent-child retrieval | ✅ Shipped | Small children retrieve, section-level parents generate: `ContextBuilder` merges siblings by section path into capped parent blocks; one citation per parent with full `chunk_ids` provenance |
| Background ingestion jobs | ✅ Shipped | `POST /documents` → 202 + job_id; dedicated worker-loop thread processes parse/chunk/embed/index; poll `/documents/jobs/{id}`; Streamlit polls until DONE/FAILED |
| Failure taxonomy (distinct failure modes) | ✅ Shipped | `FailureCode` enum on chat responses: `no_evidence` vs `insufficient_retrieval` vs `ambiguous_query` vs `contradictory_evidence` — each mapped to distinct downstream behavior |
| Evidence-derived confidence (not LLM self-report) | ✅ Shipped | Confidence = claim coverage × verdict penalty × source-agreement signal, banded high/medium/low/unsupported; agreement = fraction of top-k surfaced by BOTH dense and sparse |
| Contradiction surfacing | ✅ Shipped | `CONTRADICTORY` verdict returns structured `conflict` object with note + competing sources instead of forcing the model to pick silently |
| Provider abstraction (LLM as interchangeable part) | ✅ Shipped | `LLMProvider` protocol (`app/reasoning/providers.py`) + factory (`groq` / `null`); planner/grader/generator/memory depend only on the protocol |
| Trace lookup API | ✅ Shipped | `GET /traces/{run_id}` returns persisted per-stage latencies, normalized query, verdict |
| Multi-tenancy foundations (collections) | ✅ Shipped | `collections` table; documents bind to a collection at upload (`?collection=`); default collection seeded on startup |
| ACL-aware retrieval (filter BEFORE retrieval) | ✅ Shipped | API keys hashed at rest, scoped grants key→collections; dense pre-filters inside Chroma `$in`, sparse post-filters pre-return — unauthorized content can never reach fusion/context/generation; open/enforced dual mode with zero-breakage default. See ADR-010 |

## Deliberate deviations from the vision

- **"Zero hallucination" tagline dropped.** Unfalsifiable marketing. We claim measured things instead: 97.9% MRR, 100% trap abstention, verified citations.
- **Dynamic chunking upgraded**, not just kept: hierarchy-aware section paths instead of size tiers alone.
- **RPM limiting added** (not in original vision) after Groq free-tier realities made client-side throttling non-negotiable.
