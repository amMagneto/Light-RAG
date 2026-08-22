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

## Deliberate deviations from the vision

- **"Zero hallucination" tagline dropped.** Unfalsifiable marketing. We claim measured things instead: 97.9% MRR, 100% trap abstention, verified citations.
- **Dynamic chunking upgraded**, not just kept: hierarchy-aware section paths instead of size tiers alone.
- **RPM limiting added** (not in original vision) after Groq free-tier realities made client-side throttling non-negotiable.
