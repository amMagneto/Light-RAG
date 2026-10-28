# Retrieval Strategy Tournament — Tier List

**Only strategies that were actually executed are listed below.**
Raw generated results: `evals/reports/tournament.md` and `evals/reports/tournament.json`.
Harness: `evals/tournament.py`. Extended question set: `evals/datasets/golden_extended.jsonl`.
Full experimental design: `docs/tournament_methodology.md`.

---

## 1. What was run

| Item | Value |
|---|---|
| Corpus | 4 golden documents (`evals/corpus/`), structure-aware chunking, 16 chunks |
| Questions | 27 official + 27 extended = **54** (51 answerable, 3 unanswerable traps) |
| Ground truth | `evals.judging.build_ground_truth` — expected document **and** marker substring, computed over all chunks |
| Runs | **39** = 24 complete strategies on the implemented chunking + 15 chunking × pipeline combinations |
| Grading | Repo's `EvidenceGrader` in heuristic mode (no LLM), top-8 context per query |
| Dense fidelity | In-memory e5-L2 ranking vs Chroma `VectorIndex`: **54/54 top-8 exact** |
| RRF fidelity | Local fusion vs app `rrf_fuse`: **54/54 top-8 exact** |
| Runtime | 279 s on CPU (3 embedding models + 1 cross-encoder, all local) |

Implemented-in-repo components used directly: e5 raw L2 dense leg, BM25 leg, RRF k=60, `LexicalReranker`,
`EvidenceGrader`, `pack_sections`/`chunk_document`, `rrf_fuse`. Tournament-only implementations of the same
primitives: normalized-cosine e5, all-MiniLM-L6-v2, bge-small-en-v1.5, PRF query expansion, MMR,
linear min-max fusion, weighted RRF, cross-encoder reranking, fixed-window chunking, corrected-overlap packing.

**Tier rule** (paired bootstrap vs overall leader, 2000 resamples over the 51 judged questions):
`S` = CI lower bound > −0.005 (statistically indistinguishable from leader), `A` > −0.03, `B` > −0.10, `C` otherwise.
"Trap-safe" is a separate honesty gate (3 traps must be graded insufficient/ambiguous).

---

## 2. Tier list (39 executed configurations)

### Tier S — indistinguishable from the leader on retrieval quality

| Strategy | nDCG@8 | MRR | note |
|---|---:|---:|---|
| dense e5 (raw L2) — *implemented* | 1.000 | 1.000 | leader by tie-break |
| dense e5 (normalized cos) | 1.000 | 1.000 | normalization changes nothing here |
| dense all-MiniLM-L6-v2 | 1.000 | 1.000 | smaller/faster, same ranking |
| dense bge-small-en-v1.5 | 1.000 | 1.000 | with retrieval instruction |
| dense e5 + MMR (λ=0.7) | 1.000 | 1.000 | diversity not needed on this corpus |
| linear min-max 70/30 (dense-heavy) | 1.000 | 1.000 | best fusion by score weighting |
| union dense+sparse (round-robin) | 1.000 | 1.000 | naive but ties because dense never misses |
| RRF + cross-encoder | 1.000 | 1.000 | fixes the one RRF demotion (see §3) |
| dense + cross-encoder | 1.000 | 1.000 | |
| RRF + lexical + cross-encoder | 1.000 | 1.000 | lexical damage repaired by CE |
| chunk: structure_900_ovl + RRF + CE | 1.000 | 1.000 | corrected-overlap structure-aware |
| chunk: structure_300 + RRF + CE | 1.000 | 1.000 | small chunks |
| chunk: structure_2000 + RRF + CE | 1.000 | 1.000 | large chunks |
| chunk: fixed_900_150 + RRF + CE | 1.000 | 1.000 | **quality S, honesty fail: 2/3 trap-safe** |

### Tier A — near-top, one-question-class degradation

| Strategy | nDCG@8 | MRR | note |
|---|---:|---:|---|
| dense e5 + PRF | 0.993 | 0.990 | pseudo-relevance expansion slightly hurts dense |
| RRF k=10 / k=60 / k=100 — *k=60 implemented* | 0.993 | 0.990 | k has no measurable effect |
| RRF dense ×2 | 0.993 | 0.990 | |
| RRF bge+BM25 / MiniLM+BM25 / e5norm+BM25 | 0.993 | 0.990 | dense-model swap invisible |
| chunk: structure_* + RRF k=60 | 0.993 | 0.990 | all three structure sizes tie |
| chunk: fixed_900_150 + RRF k=60 | 0.993 | 0.990 | **honesty fail 2/3** |
| chunk: fixed_900_0 + RRF + CE | 0.993 | 0.990 | **honesty fail 2/3** |

### Tier B — measurable quality loss

| Strategy | nDCG@8 | MRR | note |
|---|---:|---:|---|
| linear min-max 50/50 | 0.986 | 0.980 | equal weights are worse than 70/30 |
| linear min-max 30/70 | 0.986 | 0.980 | |
| **RRF + lexical rerank — the implemented "full pipeline"** | 0.984 | 0.980 | lexical stage is a net negative (see §3) |
| chunk: structure_* + RRF + lexical | 0.984 | 0.980 | same damage across chunk sizes |
| sparse BM25 + PRF | 0.978 | 0.971 | best sparse-only variant, still below dense |
| RRF sparse ×2 | 0.978 | 0.971 | collapsing toward BM25 loses |
| chunk: fixed_900_0 + RRF | 0.976 | 0.967 | structure awareness matters without CE |
| chunk: fixed_900_0 + RRF + lexical | 0.973 | 0.967 | |
| sparse BM25 — *implemented* | 0.966 | 0.961 | weakest tested retriever; recall@25 = 98% |

### Tier C

None. No executed strategy fell below a −0.10 nDCG CI vs the leader; the weakest (BM25) sits at −0.034.

---

## 3. What the runs actually showed

1. **The corpus saturates the top tier.** Four different dense models (including one with embeddings
   normalized correctly per e5's training objective) produce the *identical ranking vector*. So the
   normalization caveat is real in theory but **not measurable on this
   corpus** — it should not be claimed as a bug without a harder benchmark.

2. **Fusion can demote a chunk both legs didn't see.** `q12` ("What happens when someone leaves the
   company?") is a zero-lexical-overlap paraphrase: BM25 misses it entirely, even at k=25
   (the only recall@25 < 100% across all 39 runs). Dense ranks it #1, but RRF drops it to #2
   (nDCG 1.000 → 0.631) because consensus chunks outrank single-leg hits. The cross-encoder restores it to #1.
   Fusion is not monotonically better than its best leg.

3. **The implemented lexical reranker hurts on the expanded set.** RRF 0.993 → RRF+lexical 0.984:
   it demotes the correct passage on `q21` (comparative, cross-document) and `x27` (numeric, `unlimited projects`).
   The README's H4 only checks precision@3/recall@8 within a ±2pt / ±5pt band — those metrics are saturated,
   so H4 passed while nDCG/MRR degrade. On this evidence the incumbent "full pipeline" is Tier B and
   **plain RRF is the better implemented configuration**; RRF+CE is best.

4. **BM25 is the weakest tested retriever and is comparatively fragile:** nDCG 0.966, recall@25 98%,
   and a 92.8% official-subset nDCG vs 100% for dense. PRF recovers +0.012 nDCG but not enough to leave B.

5. **Fusion weight matters more than fusion family.** Dense-heavy linear 70/30 ties the leader;
   50/50 and sparse-heavy 30/70 drop to B. RRF dense×2 ties; RRF sparse×2 collapses toward BM25 (B).

6. **Chunking: structure awareness wins on honesty, not raw top-8.** Without the cross-encoder,
   structure-aware RRF 0.993 vs fixed-window RRF 0.976. With the cross-encoder both reach ~1.000 on quality,
   but fixed windows leak one trap (2/3 trap-safe) versus 3/3 for every structure-aware configuration —
   section-coherent chunks keep the heuristic grader honest. The implemented chunker (structure-aware,
   900-char, overlap broken to 0) is Tier A; fixing overlap neither helps nor hurts retrieval quality here
   (structure_900_ovl = same 0.993 with RRF).

7. **Metric artifacts to not misread.** recall@1/precision@1 are depressed for overlap chunking and
   multi-evidence questions (`R@1 = 91.8%` for `fixed_900_150 + CE` while nDCG = 1.000): overlapping windows
   duplicate markers, so one fact occupies two ground-truth chunks and rank 1 can only catch one of them.
   nDCG/MRR are the fair metrics in this setting; they rank the same configuration S.

8. **Heuristic `sufficient` rates cluster at 86–90%** and do not order strategies — the grader signal is
   too coarse on 3 traps to be a tie-breaker. Treat trap-safety as a pass/fail gate, not a score.

---

## 4. Not tested (therefore not tiered)

No claim is made about any of these; they were not executed:

- **Graph/entity retrieval** — the `entity` source exists only as a reserved literal in
  `RetrievalSource`/RRF priority; no entity extraction or graph leg is implemented.
- **LLM-based retrieval strategies** (HyDE, multi-query decomposition, LLM listwise reranking) — require
  live Groq calls; deliberately excluded to avoid spending quota and mixing LLM variance into retrieval
  comparison. The planner's decomposition path is also unused by the pipeline.
- **Learned sparse / late interaction** (SPLADE, ColBERT, SPLADE-v3) — not installed; no model tested.
- **Larger cross-encoders / LLM rerankers** (bge-reranker-v2-m3, monoT5) — only
  `cross-encoder/ms-marco-MiniLM-L-6-v2` was executed.
- **Graph/hierarchical indexes** (RAPTOR, GraphRAG), **HyDE-style query generation**, **semantic chunking**
  via embedding segmentation, **metadata filtering** (planner `filters` are produced and discarded).
- **Reranking depth ablations** (e.g., CE over top-100 vs top-40) and **hybrid with tuned per-corpus
  weights** beyond the three tested weightings.

---

## 5. Reproduction

```powershell
.\.venv\Scripts\python.exe -m evals.tournament             # full tournament, ~5 min CPU
.\.venv\Scripts\python.exe -m evals.tournament --from-json # re-render tables from tournament.json
```

Artifacts: `evals/reports/tournament.md` (tables), `evals/reports/tournament.json` (per-question nDCG,
verdicts, bootstrap CIs, tiers), `evals/datasets/golden_extended.jsonl` (added question set).
