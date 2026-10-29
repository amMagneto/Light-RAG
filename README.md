# About

## Repository description


Light-RAG is a retrieval platform for auditable question answering over versioned document collections. It combines hybrid search, evidence grading, and claim-level citation verification, and returns citations plus a per-stage trace with every answer.



A typed, inspectable RAG pipeline for versioned document collections: hybrid dense and BM25 retrieval, RRF fusion, reranking, evidence grading, parent-child context expansion, and claim-level citation verification. Every answer includes citations and a stage-by-stage trace.

---

## About section

### Overview

Light-RAG is a retrieval-augmented generation system for document collections that change over time and answers that need to be auditable. It is built around three requirements:

- Retrieval quality is measured, not assumed.
- Document updates are indexed incrementally.
- Generated claims are traceable to retrieved evidence.

Generation is the final stage of a typed pipeline, not the starting point. Every stage has a Pydantic-validated contract, and every query persists a trace with per-stage latency.

### Pipeline

| Stage | Function |
|---|---|
| Plan | Classify the query, resolve references, and decompose multi-part questions |
| Retrieve | Dense (e5 via ChromaDB) and sparse (BM25) search over access-filtered chunks |
| Fuse | Merge both result sets using Reciprocal Rank Fusion |
| Rerank | Apply lexical reranking behind an interface ready for a cross-encoder |
| Grade | Classify evidence as sufficient, insufficient, ambiguous, or contradictory |
| Expand | Merge child chunks into section-level parent context |
| Generate | Produce structured JSON with an answer, atomic claims, and citations |
| Verify | Reject unsupported citations and derive confidence from evidence |

### Capabilities

- **Hybrid retrieval.** Dense and sparse search with RRF fusion, evaluated independently.
- **Evidence grading.** Determines whether retrieved context is adequate before generation, with a budgeted refinement loop when it is not.
- **Verified citations.** Every claim must cite existing evidence; unsupported claims and citations are removed.
- **Versioned ingestion.** Re-uploads create versions whose chunks atomically supersede previous ones, with chunk-level diffs between any two versions.
- **Incremental indexing.** Content-addressed vector IDs ensure that only net-new content is embedded.
- **Access control.** Collections and hashed API keys scope retrieval before search runs.
- **Observability.** Per-stage latency traces are persisted for every query.
- **Reproducible evaluation.** Golden dataset, ablation leaderboard, hypothesis checks, and CI-style regression floors.

### Stack

| Area | Technology |
|---|---|
| API | FastAPI, async, versioned `/api/v1` |
| Storage | SQLAlchemy 2.0 with SQLite or PostgreSQL |
| Vector search | ChromaDB with `intfloat/e5-base-v2` |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Frontend | Streamlit |
| LLM provider | Groq behind an `LLMProvider` protocol |

### Evaluation

The repository includes a hermetic golden-set evaluation: 4 documents, 16 chunks, and 27 questions across 7 query archetypes. It reports MRR, recall@k, precision@k, and hit rate for dense, sparse, RRF, and full-pipeline configurations, alongside programmatic hypothesis checks and regression floors.

On the included corpus, the full pipeline reaches 97.9% MRR and 100% recall@3. The suite requires no network access and spends no API quota.

```bash
python -m evals.runner        # full run
python -m pytest tests -q     # 66-test suite
```

### Scope and limitations

Light-RAG is designed for self-hosted, single-node deployments. Ingestion runs in one in-process worker, the reranker is lexical, identity is API-key based, and responses are non-streaming. These components sit behind stable interfaces, so they can be replaced without changing the API contract.

---

## Topics

GitHub allows up to 20 topics.

| Group | Topics |
|---|---|
| Retrieval | `rag`, `retrieval-augmented-generation`, `hybrid-search`, `information-retrieval`, `semantic-search`, `bm25`, `vector-search`, `chromadb`, `reciprocal-rank-fusion`, `reranking` |
| Generation | `llm`, `grounded-generation`, `citation-verification`, `llm-evaluation` |
| Platform | `document-versioning`, `incremental-indexing`, `access-control`, `fastapi`, `streamlit`, `sqlalchemy` |

Copy-paste:

```text
rag, retrieval-augmented-generation, hybrid-search, information-retrieval, semantic-search, bm25, vector-search, chromadb, reciprocal-rank-fusion, reranking, llm, grounded-generation, citation-verification, llm-evaluation, document-versioning, incremental-indexing, access-control, fastapi, streamlit, sqlalchemy
```

Optional substitutions: `groq`, `python`, `question-answering`, `hallucination-mitigation`, `reranker`, `chroma`, `fastapi-backend`.
