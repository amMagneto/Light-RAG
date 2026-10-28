# Retrieval Tournament — Methodology

Reference for the tier list in `docs/retrieval_tournament.md` and the raw results in
`evals/reports/tournament.md`. Code: `evals/tournament.py`.

---

## 1. Objective

Answer one question with measurements rather than claims:

> Given this repository's chunking, embedding, fusion, reranking and grading primitives, how do the
> plausible retrieval strategies rank against each other on the repository's own evidence standard?

Sub-goals:

- Measure **relative** strategy quality under one controlled corpus and one controlled ground truth.
- Include every strategy that can be executed locally with the installed stack; tier only those.
- Keep the implemented pipeline (e5 → BM25 → RRF k=60 → lexical rerank) as one competitor among many,
  not as the reference to be defended.
- Make every result reproducible in one command.

## 2. Design principles

1. **Paired comparison.** All strategies see the identical question set, chunk sets, ground truth and
   metric code. Nothing about the corpus or labels changes between competitors.
2. **One variable at a time.** The main leaderboard varies the retrieval pipeline while chunking is
   held fixed at the implemented configuration. A separate ablation varies chunking while holding the
   pipeline fixed (three pipelines across six chunkings).
3. **Fidelity before comparison.** The harness reproduces the implemented legs and fusion exactly, and
   proves it with equality checks against the production code (Chroma / `rrf_fuse`), so a win for a
   tournament-only variant is a win against the same math the app runs, not a reimplementation artifact.
4. **No LLM in the loop.** Retrieval scoring is embedding/lexical only; grading uses the repo's
   heuristic grader. This removes API variance, cost and rate-limit interference.
5. **Separate quality from honesty.** Retrieval quality (ranking metrics) and verdict honesty
   (trap abstention) are scored independently, because a strategy can rank well and still claim
   evidence for unanswerable questions.
6. **Ceiling awareness.** The corpus is small and clean. The design adds an authored second question
   set to increase n and reduce saturation, and the analysis explicitly reports ties rather than
   inventing differences.

## 3. Corpus and chunk construction

Four golden documents (900–1,150 bytes each) are parsed into `ParsedDocument` with `page_count=None`
(the real `.md` loader path), then chunked under six configurations:

| Config | Construction | Chunks |
|---|---|---:|
| `structure_900` | `chunk_document(parsed, 900, 150)` — the implemented default; effective overlap 0 due to the known bug | 16 |
| `structure_900_ovl` | `build_sections` + corrected packer (carry captured before buffer reset) — true overlap | 16 |
| `structure_300` | `build_sections` + `pack_sections(300, 50)` — implemented small tier | 20 |
| `structure_2000` | `build_sections` + `pack_sections(2000, 400)` — implemented large tier | 16 |
| `fixed_900_0` | raw `_hard_split` windows at 900 chars, no section awareness, no overlap | 7 |
| `fixed_900_150` | same, with the previous window's tail prefixed | 7 |

Chunk identity is `config/document/index`; each chunk carries content, document name, section path and
index. The corrected packer is a local copy of the app function with one change: the carry text is
computed from the buffer **before** `emit()` clears it.

Distractors were deliberately **not** added. Mutated near-duplicates or authored filler would either
create unfair "contains the marker but wrong document" traps or inject author-written content into the
ground truth. Resolution is raised through more questions instead (next section).

## 4. Question sets and ground truth

| Set | Count | Origin |
|---|---:|---|
| `golden.jsonl` | 27 (24 answerable + 3 traps) | repository's official set |
| `golden_extended.jsonl` | 27 (27 answerable) | authored for this tournament from uncovered corpus facts |

Extended-set construction rules:

- Every fact is read from the corpus; every question has a marker substring that exists verbatim
  (case-insensitive) in the intended document.
- Markers were chosen to be answer-bearing and document-distinct where possible (e.g. `$99`, `TLS 1.3`,
  `rotated every 90 days`, `two months free`, `privacy`-free of the original markers).
- Marker uniqueness is not required for correctness: relevance is defined as **expected document AND
  marker present**, so a marker recurring elsewhere cannot create false positives; recurrence inside
  the expected document only enlarges the ground-truth set, which the metrics handle explicitly
  (nDCG uses an ideal DCG over the true set size, recall divides by it).
- Categories mirror the official ones: exact_term, numeric, paraphrase, procedural, comparative,
  temporal, multi_constraint.

Ground truth per question and per chunking config is computed by
`evals.judging.build_ground_truth` over **all** chunks of that config — not over retrieved ones —
so recall is a fraction of available evidence, matching the repository's own eval philosophy.
Traps have empty ground truth and are excluded from ranking metrics; they are scored only for honesty.

## 5. Retrieval legs

All legs are precomputed once per chunk configuration and shared by every strategy. Top depth is 25
for every leg, so no strategy is advantaged or disadvantaged by retrieval depth.

### 5.1 Dense variants

| Variant | Model | Prefixes | Vector handling | Score |
|---|---|---|---|---|
| `e5_l2` (implemented) | `intfloat/e5-base-v2` | `passage:` / `query:` | raw | `1 / (1 + squared L2)` |
| `e5_cos` | same | same | L2-normalized | cosine |
| `minilm` | `all-MiniLM-L6-v2` | none | normalized at encode | cosine |
| `bge_small` | `BAAI/bge-small-en-v1.5` | query instruction only | normalized at encode | cosine |

Query instruction for bge: `Represent this sentence for searching relevant passages: `.
Ranking is per-query descending score, truncated to 25.

### 5.2 Sparse variant

The repository's `SparseIndex` (BM25Okapi, `rank-bm25`) is used unchanged, including its tokenizer
(`[a-z0-9]+`), query stopword removal, lexical-overlap admission filter and 0.01 score floor.
Top-25 per query.

### 5.3 Query-side expansions

- **PRF (pseudo-relevance feedback):** take the dense top-3 chunks for the query, count terms in them
  excluding stopwords, existing query terms and tokens shorter than 3 chars, append the 5 most frequent
  terms to the query string. Sparse PRF re-runs BM25 on the expanded string; dense PRF re-encodes the
  expanded string with the e5 query prefix and re-ranks with the same L2 score.
- **MMR:** λ = 0.7 over the dense top-25, relevance = cosine(query, chunk), redundancy = max cosine to
  already-selected chunks; selects 25 in greedy order (diversity cannot change top-1, only the tail).

## 6. Fusion implementations

| Family | Formula | Variants tested |
|---|---|---|
| RRF | `score(c) = Σ_legs w_leg / (k + rank + 1)`, 0-based rank | k = 10/60/100; w = dense×2, sparse×2; dense legs e5_l2/e5_cos/minilm/bge |
| Linear min-max | per-leg scores min-max normalized per query over returned candidates; `w_d·dense + w_s·sparse` over the union | 50/50, 70/30, 30/70 |
| Union | round-robin interleave of dense and sparse lists, dedup by chunk id | one variant |

`e5_l2` is the dense leg unless noted. Missing leg entries contribute 0 in linear fusion.
The local RRF implementation is verified against the app's `rrf_fuse` (Section 9).

## 7. Reranking strategies

| Reranker | Input pool | Mechanism |
|---|---|---|
| none | RRF / dense / BM25 output | control |
| Lexical (`LexicalReranker`, implemented) | RRF top-40 or chunk variants | weighted term coverage + phrase + number bonuses |
| Cross-encoder | pools of 25–40 | `cross-encoder/ms-marco-MiniLM-L-6-v2`, batch 32, sigmoid-free logits |
| Lexical → cross-encoder | RRF top-40 → lexical → CE | two-stage |
| MMR | dense top-25 | diversity (listed with dense) |

Cross-encoder scores are cached per `(chunk_config, question, chunk)` so repeated pools (RRF k
variants, lexical-then-CE, chunk ablation) never re-score the same pair. Every strategy returns a
ranked list; the top-8 is what the grader sees, mirroring the production context selection.

## 8. Metrics

For each question and strategy, the ranked list is compared to the ground-truth chunk-id set:

- **nDCG@8** (primary): binary gains, `DCG = Σ 1/log2(i+2)` over relevant positions,
  `IDCG = Σ 1/log2(i+2)` for `min(8, |GT|)` positions. Handles multi-chunk ground truth correctly.
- **MRR**: reciprocal rank of the first relevant chunk.
- **recall@k** for k = 1, 3, 8, 25: relevant retrieved / |GT|.
- **precision@1**, **hit@5**.
- **judged**: questions with non-empty ground truth (all traps excluded).

Honesty metrics, from the repo's `EvidenceGrader` (heuristic mode) on the strategy's top-8:

- **sufficient rate**: among answerable questions with at least one ground-truth chunk in the top-8,
  the fraction graded `sufficient`.
- **trap-safe**: among the 3 traps, the count graded `insufficient` or `ambiguous` (3/3 = pass).

Metrics are averaged over judged questions per strategy; the official 27-question subset is also
reported separately to anchor against the repository's published numbers.

## 9. Validity checks

1. **Dense fidelity.** `structure_900` chunks are inserted into a real Chroma collection through
   `VectorIndex` with a `RawE5` LangChain adapter (no prefixes — `VectorIndex` adds `passage:`/`query:`),
   then queried for all 54 questions. The in-memory `e5_l2` top-8 must equal Chroma's top-8 in exact
   order. Result: **54/54**.
2. **Fusion fidelity.** The local weighted-RRF with (k=60, w=1, w=1) must equal `app.retrieval.fusion.rrf_fuse`
   over the same dense/sparse lists for all questions. Result: **54/54 top-8 exact**.
3. **Test-environment isolation.** No LLM client is constructed with a key; the grader's LLM path is
   disabled by construction. Models are local and cached; no network data affects results.
4. **Determinism.** Legs, fusion and rerankers are deterministic; the CE cache and bootstrap use fixed
   seeds/order. Re-running produces the same tables (the markdown can be re-rendered from
   `tournament.json` without any model execution).

## 10. Statistical treatment

- Per-question nDCG@8 vectors are **paired** across strategies (same questions, same ground truth).
- Each strategy is compared to the overall leader with a paired bootstrap: resample question indices
  with replacement 2000 times (NumPy `default_rng(7)`), compute the mean difference per resample,
  report the 2.5th/97.5th percentiles as the 95% CI.
- Tiers: `S` if CI lower bound > −0.005, `A` > −0.03, `B` > −0.10, `C` otherwise. A fragile flag
  (`recall@8 < 90%`) downgrades an S to A.
- Ties are reported as ties: exact rank agreement yields a CI of `[0, 0]` and an S rating. No ordering
  is asserted beyond what the CI supports.

## 11. Threats to validity

| Threat | Mitigation / status |
|---|---|
| Tiny, clean corpus saturates rankings (many `1.000` ties) | Extended question set raises n to 51 judged; saturation reported explicitly; tiers mark ties |
| Marker ground truth favors lexical methods on exact-term questions | Ground truth matches the repo's own standard; category mix includes paraphrase/comparative/multi-constraint where lexical fails |
| Extended questions authored by the same process as the analysis | Markers are objective substrings; questions were written before running any strategy; raw per-question results are published for re-judging |
| recall@1/P@1 penalize multi-evidence and overlap-duplicated ground truth | nDCG/MRR treated as primary; artifact explained where it appears (`fixed_900_150`) |
| Tier CIs are relative to the leader, not absolute quality | Absolute metrics are printed for every run |
| Only one cross-encoder and one PRF recipe tested | Stated as scope; not generalized |
| Strategy list is not literally exhaustive | The complete tested set is enumerated; untested families are listed as untested and excluded from tiers |
| LLM-dependent retrieval (HyDE, multi-query, LLM rerank) excluded | Deliberate: avoids quota spend and LLM variance in a retrieval measurement |

## 12. Reproduce

```powershell
.\.venv\Scripts\python.exe -m evals.tournament              # full run (~5 min CPU)
.\.venv\Scripts\python.exe -m evals.tournament --from-json  # re-render tables from tournament.json
```

Outputs: `evals/reports/tournament.md`, `evals/reports/tournament.json`
(per-question nDCG, per-strategy verdicts, bootstrap CIs, tiers, fidelity counts).
