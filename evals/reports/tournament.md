# Retrieval Strategy Tournament

Questions: 27 official + 27 extended = 54 total (51 answerable, 3 traps); judged per question via the repo's marker ground truth (expected document + marker substring).
Runtime: 0s | chunk configs: structure_900=16, structure_900_ovl=16, structure_300=20, structure_2000=16, fixed_900_0=7, fixed_900_150=7

Dense-fidelity check (in-memory e5 L2 vs Chroma): 54/54 top-8 exact.
RRF-fidelity check (local vs app rrf_fuse): 54/54 top-8 exact.

Tiers are relative to the overall leader, paired bootstrap over judged questions (2000 resamples): S = CI lower bound > -0.005 (statistically indistinguishable from leader), A > -0.03, B > -0.10, C otherwise. 'fragile-recall' flags recall@8 < 90%.

## Leaderboard (all questions, judged)

| tier | strategy | family | nDCG@8 | MRR | R@1 | R@3 | R@8 | R@25 | P@1 | sufficient | trap-safe | official nDCG@8 | judged | delta vs leader [95% CI] |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S | dense e5 (raw L2) | dense | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | dense e5 (normalized cos) | dense | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | dense all-MiniLM-L6-v2 | dense | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | dense bge-small-en-v1.5 | dense | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | dense e5 + MMR (l=0.7) | dense | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | linear minmax 70/30 | fusion | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 88.2% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | union dense+sparse | fusion | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 88.2% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | RRF + cross-encoder | rerank | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | dense + cross-encoder | rerank | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | RRF + lexical + cross-encoder | rerank | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | chunk:structure_900_ovl | RRF + cross-encoder | chunking+rerank | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | chunk:structure_300 | RRF + cross-encoder | chunking+rerank | 1.000 | 1.000 | 97.1% | 100.0% | 100.0% | 100.0% | 100.0% | 88.2% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | chunk:structure_2000 | RRF + cross-encoder | chunking+rerank | 1.000 | 1.000 | 99.0% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 3/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| S | chunk:fixed_900_150 | RRF + cross-encoder | chunking+rerank | 1.000 | 1.000 | 91.8% | 100.0% | 100.0% | 100.0% | 100.0% | 86.3% | 2/3 | 100.0% | 51 | +0.000 [+0.000, +0.000] |
| A | dense e5 + PRF | dense | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 86.3% | 3/3 | 100.0% | 51 | -0.007 [-0.022, +0.000] |
| A | RRF k=10 | fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | RRF k=60 | fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | RRF k=100 | fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | RRF dense x2 | fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | RRF bge+BM25 | fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | RRF MiniLM+BM25 | fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | RRF e5norm+BM25 | fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | chunk:structure_900_ovl | RRF k=60 | chunking+fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | chunk:structure_300 | RRF k=60 | chunking+fusion | 0.993 | 0.990 | 95.1% | 100.0% | 100.0% | 100.0% | 98.0% | 90.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | chunk:structure_2000 | RRF k=60 | chunking+fusion | 0.993 | 0.990 | 97.1% | 100.0% | 100.0% | 100.0% | 98.0% | 88.2% | 3/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | chunk:fixed_900_0 | RRF + cross-encoder | chunking+rerank | 0.993 | 0.990 | 96.1% | 100.0% | 100.0% | 100.0% | 98.0% | 86.3% | 2/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | chunk:fixed_900_150 | RRF k=60 | chunking+fusion | 0.993 | 0.990 | 89.9% | 100.0% | 100.0% | 100.0% | 98.0% | 86.3% | 2/3 | 98.5% | 51 | -0.007 [-0.022, +0.000] |
| A | chunk:fixed_900_150 | RRF + lexical rerank | chunking+rerank | 0.992 | 0.990 | 89.9% | 99.3% | 100.0% | 100.0% | 98.0% | 86.3% | 2/3 | 99.9% | 51 | -0.008 [-0.023, +0.000] |
| B | linear minmax 50/50 | fusion | 0.986 | 0.980 | 95.1% | 100.0% | 100.0% | 100.0% | 96.1% | 88.2% | 3/3 | 96.9% | 51 | -0.014 [-0.036, +0.000] |
| B | linear minmax 30/70 | fusion | 0.986 | 0.980 | 95.1% | 100.0% | 100.0% | 100.0% | 96.1% | 88.2% | 3/3 | 96.9% | 51 | -0.014 [-0.036, +0.000] |
| B | RRF + lexical rerank | rerank | 0.984 | 0.980 | 95.1% | 100.0% | 100.0% | 100.0% | 96.1% | 88.2% | 3/3 | 98.1% | 51 | -0.016 [-0.039, +0.000] |
| B | chunk:structure_900_ovl | RRF + lexical rerank | chunking+rerank | 0.984 | 0.980 | 95.1% | 100.0% | 100.0% | 100.0% | 96.1% | 88.2% | 3/3 | 98.1% | 51 | -0.016 [-0.038, +0.000] |
| B | chunk:structure_300 | RRF + lexical rerank | chunking+rerank | 0.984 | 0.980 | 93.1% | 100.0% | 100.0% | 100.0% | 96.1% | 90.2% | 3/3 | 98.1% | 51 | -0.016 [-0.038, +0.000] |
| B | chunk:structure_2000 | RRF + lexical rerank | chunking+rerank | 0.984 | 0.980 | 95.1% | 100.0% | 100.0% | 100.0% | 96.1% | 88.2% | 3/3 | 98.1% | 51 | -0.016 [-0.039, +0.000] |
| B | sparse BM25 + PRF | sparse | 0.978 | 0.971 | 93.1% | 100.0% | 100.0% | 100.0% | 94.1% | 86.3% | 3/3 | 98.5% | 51 | -0.022 [-0.051, +0.000] |
| B | RRF sparse x2 | fusion | 0.978 | 0.971 | 93.1% | 100.0% | 100.0% | 100.0% | 94.1% | 88.2% | 3/3 | 95.4% | 51 | -0.022 [-0.051, +0.000] |
| B | chunk:fixed_900_0 | RRF k=60 | chunking+fusion | 0.976 | 0.967 | 92.2% | 100.0% | 100.0% | 100.0% | 94.1% | 86.3% | 2/3 | 96.9% | 51 | -0.024 [-0.056, +0.000] |
| B | chunk:fixed_900_0 | RRF + lexical rerank | chunking+rerank | 0.973 | 0.967 | 92.2% | 99.0% | 100.0% | 100.0% | 94.1% | 86.3% | 2/3 | 99.5% | 51 | -0.027 [-0.058, -0.002] |
| B | sparse BM25 | sparse | 0.966 | 0.961 | 93.1% | 98.0% | 98.0% | 98.0% | 94.1% | 88.0% | 3/3 | 92.8% | 51 | -0.034 [-0.081, +0.000] |

## Chunking ablation (pipeline held constant)

| chunk config | chunks | pipeline | nDCG@8 | MRR | R@1 | R@8 | judged |
|---|---:|---|---:|---:|---:|---:|---:|
| fixed_900_0 | 7 | chunk:fixed_900_0 | RRF + cross-encoder | 0.993 | 0.990 | 96.1% | 100.0% | 51 |
| fixed_900_0 | 7 | chunk:fixed_900_0 | RRF k=60 | 0.976 | 0.967 | 92.2% | 100.0% | 51 |
| fixed_900_0 | 7 | chunk:fixed_900_0 | RRF + lexical rerank | 0.973 | 0.967 | 92.2% | 100.0% | 51 |
| fixed_900_150 | 7 | chunk:fixed_900_150 | RRF + cross-encoder | 1.000 | 1.000 | 91.8% | 100.0% | 51 |
| fixed_900_150 | 7 | chunk:fixed_900_150 | RRF k=60 | 0.993 | 0.990 | 89.9% | 100.0% | 51 |
| fixed_900_150 | 7 | chunk:fixed_900_150 | RRF + lexical rerank | 0.992 | 0.990 | 89.9% | 100.0% | 51 |
| structure_2000 | 16 | chunk:structure_2000 | RRF + cross-encoder | 1.000 | 1.000 | 99.0% | 100.0% | 51 |
| structure_2000 | 16 | chunk:structure_2000 | RRF k=60 | 0.993 | 0.990 | 97.1% | 100.0% | 51 |
| structure_2000 | 16 | chunk:structure_2000 | RRF + lexical rerank | 0.984 | 0.980 | 95.1% | 100.0% | 51 |
| structure_300 | 20 | chunk:structure_300 | RRF + cross-encoder | 1.000 | 1.000 | 97.1% | 100.0% | 51 |
| structure_300 | 20 | chunk:structure_300 | RRF k=60 | 0.993 | 0.990 | 95.1% | 100.0% | 51 |
| structure_300 | 20 | chunk:structure_300 | RRF + lexical rerank | 0.984 | 0.980 | 93.1% | 100.0% | 51 |
| structure_900_ovl | 16 | chunk:structure_900_ovl | RRF + cross-encoder | 1.000 | 1.000 | 99.0% | 100.0% | 51 |
| structure_900_ovl | 16 | chunk:structure_900_ovl | RRF k=60 | 0.993 | 0.990 | 97.1% | 100.0% | 51 |
| structure_900_ovl | 16 | chunk:structure_900_ovl | RRF + lexical rerank | 0.984 | 0.980 | 95.1% | 100.0% | 51 |

## Strategies with recall@25 < 100%

- structure_900 | sparse BM25: recall@25=98.0%, recall@8=98.0%, judged=51
