# Retrieval Evaluation Report

Generated: `2026-08-21T20:15:13.648086` | questions: 27 | corpus chunks: 16 | k_max: 8

## Methodology

- **Corpus:** 4 authored documents; every fact has a known passage.
- **Judging:** a retrieved chunk is RELEVANT iff it comes from an expected document AND contains a ground-truth marker substring. Ground truth is computed corpus-wide over ALL indexed chunks, so recall is a true fraction of available evidence.
- **recall@k** is primary: generation cannot use evidence it never retrieves.
- **MRR** captures single-fact lookup UX; **precision@k** proxies context pollution; **hit_rate@k** includes unanswerable traps so misses stay visible.
- **Verdict honesty** is scored separately: traps must abstain (insufficient/ambiguous); answerable questions must say sufficient when evidence was actually retrieved.
- Grader runs in heuristic mode (LLM disabled) so numbers isolate retrieval quality from LLM variance and cost zero quota.

## Overall leaderboard

| metric | **dense** | **sparse** | **fused** | **full** |
|---|---:|---:|---:|---:|
| MRR | 100.0% | 95.7% | 97.9% | 97.9% |
| recall@1 | 97.9% | 89.1% | 93.8% | 93.8% |
| recall@3 | 100.0% | 100.0% | 100.0% | 100.0% |
| recall@5 | 100.0% | 100.0% | 100.0% | 100.0% |
| recall@8 | 100.0% | 100.0% | 100.0% | 100.0% |
| precision@1 | 100.0% | 91.3% | 95.8% | 95.8% |
| precision@3 | 34.7% | 34.8% | 34.7% | 34.7% |
| hit rate@1 | 88.9% | 77.8% | 85.2% | 85.2% |
| hit rate@5 | 88.9% | 85.2% | 88.9% | 88.9% |
| judged queries (of answerable) | 24 | 23 | 24 | 24 |

## Per-category MRR

| query archetype | **dense** | **sparse** | **fused** | **full** |
|---|---:|---:|---:|---:|
| exact_term | 100.0% | 100.0% | 100.0% | 100.0% |
| numeric | 100.0% | 100.0% | 100.0% | 100.0% |
| paraphrase | 100.0% | 87.5% | 90.0% | 90.0% |
| procedural | 100.0% | 100.0% | 100.0% | 100.0% |
| comparative | 100.0% | 83.3% | 100.0% | 100.0% |
| multi_constraint | 100.0% | 100.0% | 100.0% | 100.0% |
| *unanswerable traps* | scored under verdict honesty below |

## Verdict honesty (heuristic grader)

| check | result | rate |
|---|---|---:|
| unanswerable traps correctly abstain (insufficient/ambiguous) | 3/3 | **100.0%** |
| answerable → SUFFICIENT when evidence was actually retrieved | 20/24 | **83.3%** |

## Context engine & latency budget

| component | value |
|---|---:|
| context dedup ratio (kept / input chunks) | 100.0% |
| relevant fraction of final packed context | 11.6% |
| dense search (e5 embed + chroma, avg) | 163 ms |
| sparse search (BM25, avg) | 1.1 ms |
| RRF fusion + rerank (avg) | 1.0 ms |

## Hypothesis verification

| hypothesis | claim | evidence | verdict |
|---|---|---|---|
| `H1_exact_term_sparse_wins` | MRR(sparse) >= MRR(dense) on exact_term queries | mrr_dense=100.0%, mrr_sparse=100.0% | **PASS** |
| `H2_paraphrase_dense_wins` | MRR(dense) > MRR(sparse) on paraphrase queries | mrr_dense=100.0%, mrr_sparse=87.5% | **PASS** |
| `H3_fusion_never_loses_recall` | recall@8(fused) >= max(recall@8 legs) - 0.02 | recall_dense=100.0%, recall_sparse=100.0%, recall_fused=100.0% | **PASS** |
| `H4_reranker_trades_cleanly` | precision@3(full) >= precision@3(fused) - 0.02 AND recall@8(full) >= recall@8(fused) - 0.05 | precision3_fused=34.7%, precision3_full=34.7%, recall8_fused=100.0%, recall8_full=100.0% | **PASS** |
