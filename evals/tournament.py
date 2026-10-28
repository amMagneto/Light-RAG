"""Retrieval strategy tournament.

Runs a fixed, broad set of retrieval strategies over identical corpora/legs and
writes evals/reports/tournament.{md,json}. Every strategy listed in the output
was actually executed. Strategies that were not executed are not listed.

Corpus: the repo's 4 golden documents. Queries: official 27-question golden set
plus an extended 27-question marker set (evals/datasets/golden_extended.jsonl).
Ground truth: evals.judging.build_ground_truth (doc + marker substring).
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.ingestion.chunker import (
    _hard_split,
    _tail,
    build_sections,
    chunk_document,
    pack_sections,
)
from app.ingestion.loaders import ParsedDocument, ParsedPage
from app.reasoning.grader import EvidenceGrader
from app.reasoning.llm import LLMClient
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import LexicalReranker
from app.retrieval.sparse import SparseIndex, _stopwords, tokenize
from app.schemas.retrieval import RetrievalCandidate
from evals.judging import build_ground_truth

EVALS_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVALS_DIR / "corpus"
REPORTS_DIR = EVALS_DIR / "reports"
KS = (1, 3, 8, 25)
DEFAULT_CONFIG = "structure_900"
TOP_N = 25


# --------------------------------------------------------------------------- #
# Corpus construction
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    cid: str
    doc: str
    content: str
    section_path: list[str]
    index: int


def parsed(doc: str, text: str) -> ParsedDocument:
    return ParsedDocument(
        path=ROOT / "evals" / "corpus" / doc,
        mime_type="text/markdown",
        pages=[ParsedPage(page=None, text=text)],
        page_count=None,
    )


def pack_sections_with_overlap(sections, chunk_size: int, overlap: int):
    """Corrected copy of app.ingestion.chunker.pack_sections: captures the carry
    text BEFORE emit() resets the buffer, so overlap actually works."""
    chunks: list[tuple[str, list[str]]] = []
    index = 0
    for section in sections:
        current_parts: list[str] = []
        current_len = 0

        def flush() -> str:
            nonlocal current_parts, current_len, index
            if not current_parts:
                return ""
            content = "\n\n".join(current_parts).strip()
            if content:
                chunks.append((content, list(section.path)))
                index += 1
            current_parts = []
            current_len = 0
            return content

        for block_text, _page in section.blocks:
            text = block_text
            if len(text) + 2 > chunk_size:
                flush()
                pieces = [p for p in _hard_split(text, chunk_size - overlap) if p.strip()]
                for piece in pieces:
                    chunks.append((piece.strip(), list(section.path)))
                    index += 1
                carry = _tail(pieces[-1], overlap) if pieces else ""
                current_parts = [carry] if carry.strip() else []
                current_len = len(carry)
                continue
            if current_len + len(text) + 2 > chunk_size and current_parts:
                emitted = flush()
                carry = _tail(emitted, overlap) if overlap > 0 else ""
                if carry.strip():
                    current_parts = [carry]
                    current_len = len(carry)
            current_parts.append(text)
            current_len += len(text) + 2
        flush()
    return chunks


def fixed_windows(text: str, size: int, overlap: int) -> list[str]:
    pieces = [p for p in _hard_split(text, size) if p.strip()]
    out = []
    for i, piece in enumerate(pieces):
        if i > 0 and overlap > 0:
            piece = (_tail(pieces[i - 1], overlap) + "\n" + piece).strip()
        out.append(piece)
    return out


def build_configs() -> dict[str, list[Chunk]]:
    texts = {p.name: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}
    configs: dict[str, list[Chunk]] = {}

    def add(name: str, doc: str, pairs) -> None:
        for i, (content, path) in enumerate(pairs):
            if content.strip():
                configs[name].append(Chunk(f"{name}/{doc}/{i}", doc, content, list(path), i))

    def section_pairs(doc: str, text: str, size: int, overlap: int, corrected: bool):
        secs = build_sections(parsed(doc, text))
        if corrected:
            return pack_sections_with_overlap(secs, size, overlap)
        return [(c.content, c.section_path) for c in pack_sections(secs, size, overlap)]

    specs = [
        ("structure_900", 900, 150, "app"),
        ("structure_900_ovl", 900, 150, "corrected"),
        ("structure_300", 300, 50, "app"),
        ("structure_2000", 2000, 400, "app"),
        ("fixed_900_0", 900, 0, "fixed"),
        ("fixed_900_150", 900, 150, "fixed"),
    ]
    for name, size, overlap, kind in specs:
        configs[name] = []
        for doc, text in texts.items():
            if kind == "fixed":
                add(name, doc, [(c, []) for c in fixed_windows(text, size, overlap)])
            elif kind == "corrected":
                add(name, doc, section_pairs(doc, text, size, overlap, True))
            else:
                add(name, doc, section_pairs(doc, text, size, overlap, False))
    return configs


# --------------------------------------------------------------------------- #
# Embeddings / legs
# --------------------------------------------------------------------------- #

class STEmbedder:
    def __init__(self, model_name: str, doc_prefix: str = "", query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    def encode(self, texts: list[str], kind: str = "doc", normalize: bool = False) -> np.ndarray:
        prefix = self.doc_prefix if kind == "doc" else self.query_prefix
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )


class CrossEncoderRanker:
    def __init__(self):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cache: dict[tuple, float] = {}

    def rank(self, key: tuple, query: str, pairs: list[tuple[str, str]]) -> list[str]:
        missing = [(cid, content) for cid, content in pairs if (*key, cid) not in self.cache]
        if missing:
            scores = self.model.predict(
                [(query, content) for _, content in missing],
                batch_size=32,
                show_progress_bar=False,
            )
            for (cid, _), score in zip(missing, scores):
                self.cache[(*key, cid)] = float(score)
        return [cid for cid, _ in sorted(pairs, key=lambda p: self.cache[(*key, p[0])], reverse=True)]


class Legs:
    """All precomputed retrieval legs for one chunk configuration."""

    def __init__(self, name: str, chunks: list[Chunk], questions: list[dict]):
        self.name = name
        self.chunks = chunks
        self.by_id = {c.cid: c for c in chunks}
        self.queries = {q["id"]: q["question"] for q in questions}
        self.dense: dict[str, dict[str, list[tuple[str, float]]]] = {}
        self.sparse_hits: dict[str, list[tuple[str, float]]] = {}
        self._build()

    def _build(self) -> None:
        contents = [c.content for c in self.chunks]
        docs_e5 = E5.encode(contents, "doc", normalize=False)
        docs_e5_norm = docs_e5 / (np.linalg.norm(docs_e5, axis=1, keepdims=True) + 1e-12)
        docs_minilm = MINILM.encode(contents, "doc", normalize=True)
        docs_bge = BGE.encode(contents, "doc", normalize=True)

        qids = list(self.queries)
        qtexts = [self.queries[q] for q in qids]
        q_e5 = E5.encode(qtexts, "query", normalize=False)
        q_e5_norm = q_e5 / (np.linalg.norm(q_e5, axis=1, keepdims=True) + 1e-12)
        q_minilm = MINILM.encode(qtexts, "query", normalize=True)
        q_bge = BGE.encode(qtexts, "query", normalize=True)

        variants = {
            "e5_l2": ("l2", q_e5, docs_e5),
            "e5_cos": ("cos", q_e5_norm, docs_e5_norm),
            "minilm": ("cos", q_minilm, docs_minilm),
            "bge_small": ("cos", q_bge, docs_bge),
        }
        self.dense = {v: {} for v in variants}
        for variant, (mode, qmat, dmat) in variants.items():
            for row, qid in enumerate(qids):
                if mode == "l2":
                    dist = ((dmat - qmat[row]) ** 2).sum(axis=1)
                    scores = 1.0 / (1.0 + dist)
                else:
                    scores = dmat @ qmat[row]
                order = np.argsort(-scores)[:TOP_N]
                self.dense[variant][qid] = [
                    (self.chunks[i].cid, float(scores[i])) for i in order
                ]

        self.e5_docs_raw = docs_e5
        self.e5_query_raw = {qid: q_e5[i] for i, qid in enumerate(qids)}

        sparse = SparseIndex()
        grouped: dict[str, dict] = {}
        for c in self.chunks:
            row = grouped.setdefault(
                c.doc,
                {"document_id": c.doc, "document_name": c.doc, "collection_id": None, "chunks": []},
            )
            row["chunks"].append(
                {
                    "id": c.cid,
                    "content": c.content,
                    "chunk_index": c.index,
                    "page_start": None,
                    "page_end": None,
                    "section_path": c.section_path,
                }
            )
        sparse.refresh(list(grouped.values()))
        for qid in qids:
            hits = sparse.search(self.queries[qid], TOP_N)
            self.sparse_hits[qid] = [(h.chunk_id, h.score) for h in hits]
        self.sparse = sparse

    # ---- helpers -------------------------------------------------------- #

    def candidates(self, cids: list[str]) -> list[RetrievalCandidate]:
        out = []
        for cid in cids:
            c = self.by_id[cid]
            out.append(
                RetrievalCandidate(
                    chunk_id=c.cid,
                    document_id=c.doc,
                    document_name=c.doc,
                    content=c.content,
                    chunk_index=c.index,
                    section_path=c.section_path,
                )
            )
        return out

    def rrf(self, qid: str, k: int = 60, wd: float = 1.0, ws: float = 1.0,
            dense_variant: str = "e5_l2") -> list[str]:
        scores: dict[str, float] = {}
        for rank, (cid, _) in enumerate(self.dense[dense_variant][qid]):
            scores[cid] = scores.get(cid, 0.0) + wd / (k + rank + 1)
        for rank, (cid, _) in enumerate(self.sparse_hits[qid]):
            scores[cid] = scores.get(cid, 0.0) + ws / (k + rank + 1)
        return sorted(scores, key=lambda c: scores[c], reverse=True)

    def linear(self, qid: str, wd: float, ws: float) -> list[str]:
        def norm(pairs):
            d = dict(pairs)
            if not d:
                return {}
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-12:
                return {k: 1.0 for k in d}
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}

        dn = norm(self.dense["e5_l2"][qid])
        sn = norm(self.sparse_hits[qid])
        keys = set(dn) | set(sn)
        return sorted(keys, key=lambda c: wd * dn.get(c, 0.0) + ws * sn.get(c, 0.0), reverse=True)

    def lexical(self, qid: str, cids: list[str]) -> list[str]:
        query = self.queries[qid]
        scored = [
            (LEX.score(query, self.by_id[cid].content), -i, cid) for i, cid in enumerate(cids)
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [cid for _, _, cid in scored]

    def crossencoder(self, qid: str, cids: list[str]) -> list[str]:
        pairs = [(cid, self.by_id[cid].content) for cid in cids]
        return CE.rank((self.name, qid), self.queries[qid], pairs)

    def prf_terms(self, qid: str, k: int = 5) -> list[str]:
        query_terms = set(tokenize(self.queries[qid]))
        counts: Counter = Counter()
        for cid, _ in self.dense["e5_l2"][qid][:3]:
            for term in tokenize(self.by_id[cid].content):
                if term in _stopwords() or term in query_terms or len(term) < 3:
                    continue
                counts[term] += 1
        return [term for term, _ in counts.most_common(k)]

    def dense_prf(self, qid: str) -> list[str]:
        expanded = self.queries[qid] + " " + " ".join(self.prf_terms(qid))
        qvec = E5.encode([expanded], "query", normalize=False)[0]
        dist = ((self.e5_docs_raw - qvec) ** 2).sum(axis=1)
        order = np.argsort(-(1.0 / (1.0 + dist)))[:TOP_N]
        return [self.chunks[i].cid for i in order]

    def sparse_prf(self, qid: str) -> list[str]:
        expanded = self.queries[qid] + " " + " ".join(self.prf_terms(qid))
        hits = self.sparse.search(expanded, TOP_N)
        return [h.chunk_id for h in hits]

    def mmr(self, qid: str, lam: float = 0.7) -> list[str]:
        pool = [cid for cid, _ in self.dense["e5_l2"][qid]]
        vecs = {cid: self.e5_docs_raw[[c.cid for c in self.chunks].index(cid)] for cid in pool}
        qvec = self.e5_query_raw[qid]
        norms = {cid: v / (np.linalg.norm(v) + 1e-12) for cid, v in vecs.items()}
        qn = qvec / (np.linalg.norm(qvec) + 1e-12)
        rel = {cid: float(norms[cid] @ qn) for cid in pool}
        selected: list[str] = []
        while pool and len(selected) < TOP_N:
            best_cid, best_score = None, -1e9
            for cid in pool:
                penalty = max((float(norms[cid] @ norms[s]) for s in selected), default=0.0)
                score = lam * rel[cid] - (1 - lam) * penalty
                if score > best_score:
                    best_cid, best_score = cid, score
            selected.append(best_cid)
            pool.remove(best_cid)
        return selected


E5 = STEmbedder("intfloat/e5-base-v2", doc_prefix="passage: ", query_prefix="query: ")
MINILM = STEmbedder("all-MiniLM-L6-v2")
BGE = STEmbedder(
    "BAAI/bge-small-en-v1.5",
    query_prefix="Represent this sentence for searching relevant passages: ",
)
LEX = LexicalReranker()
CE = CrossEncoderRanker()


# --------------------------------------------------------------------------- #
# Strategy registry (all executed)
# --------------------------------------------------------------------------- #

CORE_STRATEGIES: list[tuple[str, str, callable]] = [
    ("dense e5 (raw L2)", "dense", lambda L, q: [c for c, _ in L.dense["e5_l2"][q]]),
    ("dense e5 (normalized cos)", "dense", lambda L, q: [c for c, _ in L.dense["e5_cos"][q]]),
    ("dense all-MiniLM-L6-v2", "dense", lambda L, q: [c for c, _ in L.dense["minilm"][q]]),
    ("dense bge-small-en-v1.5", "dense", lambda L, q: [c for c, _ in L.dense["bge_small"][q]]),
    ("sparse BM25", "sparse", lambda L, q: [c for c, _ in L.sparse_hits[q]]),
    ("sparse BM25 + PRF", "sparse", lambda L, q: L.sparse_prf(q)),
    ("dense e5 + PRF", "dense", lambda L, q: L.dense_prf(q)),
    ("dense e5 + MMR (l=0.7)", "dense", lambda L, q: L.mmr(q)),
    ("RRF k=10", "fusion", lambda L, q: L.rrf(q, k=10)),
    ("RRF k=60", "fusion", lambda L, q: L.rrf(q, k=60)),
    ("RRF k=100", "fusion", lambda L, q: L.rrf(q, k=100)),
    ("RRF dense x2", "fusion", lambda L, q: L.rrf(q, wd=2.0)),
    ("RRF sparse x2", "fusion", lambda L, q: L.rrf(q, ws=2.0)),
    ("linear minmax 50/50", "fusion", lambda L, q: L.linear(q, 0.5, 0.5)),
    ("linear minmax 70/30", "fusion", lambda L, q: L.linear(q, 0.7, 0.3)),
    ("linear minmax 30/70", "fusion", lambda L, q: L.linear(q, 0.3, 0.7)),
    ("union dense+sparse", "fusion", lambda L, q: _union(L, q)),
    ("RRF + lexical rerank", "rerank", lambda L, q: L.lexical(q, L.rrf(q, k=60))),
    ("RRF + cross-encoder", "rerank", lambda L, q: L.crossencoder(q, L.rrf(q, k=60))),
    ("dense + cross-encoder", "rerank", lambda L, q: L.crossencoder(q, [c for c, _ in L.dense["e5_l2"][q]])),
    ("RRF + lexical + cross-encoder", "rerank", lambda L, q: L.crossencoder(q, L.lexical(q, L.rrf(q, k=60)))),
    ("RRF bge+BM25", "fusion", lambda L, q: L.rrf(q, k=60, dense_variant="bge_small")),
    ("RRF MiniLM+BM25", "fusion", lambda L, q: L.rrf(q, k=60, dense_variant="minilm")),
    ("RRF e5norm+BM25", "fusion", lambda L, q: L.rrf(q, k=60, dense_variant="e5_cos")),
]

CHUNK_PIPELINES: list[tuple[str, str, callable]] = [
    ("RRF k=60", "fusion", lambda L, q: L.rrf(q, k=60)),
    ("RRF + lexical rerank", "rerank", lambda L, q: L.lexical(q, L.rrf(q, k=60))),
    ("RRF + cross-encoder", "rerank", lambda L, q: L.crossencoder(q, L.rrf(q, k=60))),
]


def _union(L: Legs, qid: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for (d, _), (s, _) in zip(L.dense["e5_l2"][qid], L.sparse_hits[qid]):
        for cid in (d, s):
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
    for cid, _ in L.dense["e5_l2"][qid] + L.sparse_hits[qid]:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def ndcg(flags: list[bool], k: int, total_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags[:k]) if f)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, total_relevant)))
    return dcg / ideal if ideal else 0.0


def question_metrics(ranked: list[str], gt: set[str]) -> dict:
    flags = [cid in gt for cid in ranked]
    n = len(gt)
    if not n:
        return {}
    mrr = 0.0
    for i, f in enumerate(flags, 1):
        if f:
            mrr = 1.0 / i
            break
    out = {"mrr": mrr, "ndcg@8": ndcg(flags, 8, n), "precision@1": 1.0 if flags[:1] == [True] else 0.0}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for f in flags[:k] if f) / n
    out["hit@5"] = 1.0 if any(flags[:5]) else 0.0
    return out


def mean_metrics(per_q: dict[str, dict], qids: list[str]) -> dict:
    keys = ["mrr", "ndcg@8", "precision@1", "recall@1", "recall@3", "recall@8", "recall@25", "hit@5"]
    out = {}
    for key in keys:
        vals = [per_q[q][key] for q in qids if q in per_q and key in per_q[q]]
        out[key] = float(np.mean(vals)) if vals else None
    out["judged"] = sum(1 for q in qids if q in per_q)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def main() -> None:
    started = time.perf_counter()
    official = [json.loads(line) for line in (EVALS_DIR / "datasets" / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    extended = [json.loads(line) for line in (EVALS_DIR / "datasets" / "golden_extended.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = official + extended
    official_ids = [q["id"] for q in official]
    answerable_ids = [q["id"] for q in questions if q["answerable"]]
    trap_ids = [q["id"] for q in questions if not q["answerable"]]

    configs = build_configs()
    print("chunk configs:", {k: len(v) for k, v in configs.items()})

    settings = Settings(groq_api_key=None, enable_memory_extraction=False)
    grader = EvidenceGrader(LLMClient(settings), settings)

    legs_by_config: dict[str, Legs] = {}
    gt_by_config: dict[str, dict[str, set[str]]] = {}
    for name, chunks in configs.items():
        legs = Legs(name, chunks, questions)
        legs_by_config[name] = legs
        gt_by_config[name] = {
            q["id"]: build_ground_truth(legs.candidates([c.cid for c in chunks]), q["expected_documents"], q["evidence_markers"])
            for q in questions
        }
        print(f"legs built: {name} ({len(chunks)} chunks)")

    # Verification 1: in-memory e5 L2 ranking == Chroma VectorIndex ranking.
    chroma_match = _verify_chroma(legs_by_config[DEFAULT_CONFIG], questions)

    # Verification 2: local RRF == app rrf_fuse.
    rrf_match = _verify_rrf(legs_by_config[DEFAULT_CONFIG], questions)

    runs: list[dict] = []

    async def run_strategy(legs: Legs, name: str, family: str, fn) -> None:
        per_q: dict[str, dict] = {}
        ranked_by_q: dict[str, list[str]] = {}
        verdicts: dict[str, str] = {}
        for q in questions:
            qid = q["id"]
            ranked = fn(legs, qid)
            ranked_by_q[qid] = ranked
            gm = question_metrics(ranked, gt_by_config[legs.name][qid])
            if gm:
                per_q[qid] = gm
            assessment = await grader.grade(q["question"], legs.candidates(ranked[:8]))
            verdicts[qid] = assessment.verdict.value
        all_m = mean_metrics(per_q, [q["id"] for q in questions])
        official_m = mean_metrics(per_q, official_ids)
        answerable_with_evidence = [
            q["id"] for q in questions
            if q["answerable"] and any(cid in gt_by_config[legs.name][q["id"]] for cid in ranked_by_q[q["id"]][:8])
        ]
        sufficient = [qid for qid in answerable_with_evidence if verdicts[qid] == "sufficient"]
        trap_safe = sum(1 for q in trap_ids if verdicts[q] in {"insufficient", "ambiguous"})
        run = {
            "config": legs.name,
            "strategy": name,
            "family": family,
            "metrics": all_m,
            "official": official_m,
            "verdicts": {
                "sufficient_rate": len(sufficient) / len(answerable_with_evidence) if answerable_with_evidence else None,
                "answerable_with_evidence": len(answerable_with_evidence),
                "trap_safe": trap_safe,
                "trap_total": len(trap_ids),
            },
            "ndcg_per_q": {qid: per_q[qid]["ndcg@8"] for qid in per_q},
            "judged_ids": list(per_q),
        }
        runs.append(run)
        print(f"  ran {legs.name} | {name} | nDCG@8={all_m['ndcg@8']:.3f} MRR={all_m['mrr']:.3f} R@1={all_m['recall@1']:.3f}")

    for name, family, fn in CORE_STRATEGIES:
        await run_strategy(legs_by_config[DEFAULT_CONFIG], name, family, fn)

    for config_name, legs in legs_by_config.items():
        if config_name == DEFAULT_CONFIG:
            continue
        for name, family, fn in CHUNK_PIPELINES:
            await run_strategy(legs, f"chunk:{config_name} | {name}", f"chunking+{family}", fn)

    _bootstrap_and_tier(runs, answerable_ids)

    report = _render(
        runs, configs, official_ids, len(questions), len(answerable_ids),
        chroma_match, rrf_match, time.perf_counter() - started,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "tournament.md").write_text(report, encoding="utf-8")
    payload = {"configs": {k: len(v) for k, v in configs.items()}, "runs": runs,
               "chroma_match": chroma_match, "rrf_match": rrf_match}
    (REPORTS_DIR / "tournament.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\n" + report)


def _verify_chroma(legs: Legs, questions: list[dict]) -> dict:
    from langchain_core.embeddings import Embeddings

    from app.ingestion.indexer import VectorIndex, vector_key

    class RawE5(Embeddings):
        def embed_documents(self, texts):
            return E5.model.encode(texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True).tolist()

        def embed_query(self, text):
            return E5.model.encode([text], convert_to_numpy=True)[0].tolist()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(chroma_dir=tmp, collection_name="verify", groq_api_key=None)
        index = VectorIndex(settings, embeddings=RawE5())
        rows = [
            {
                "vector_key": vector_key(c.doc, f"h{c.index}"),
                "id": c.cid,
                "document_id": c.doc,
                "document_name": c.doc,
                "collection_id": None,
                "content": c.content,
                "chunk_index": c.index,
                "page_start": None,
                "page_end": None,
                "section_path": c.section_path,
            }
            for c in legs.chunks
        ]
        index.upsert_chunks(rows)
        matches = 0
        for q in questions:
            chroma_ids = [c.chunk_id for c, _ in index.query(q["question"], 8)]
            memory_ids = [c for c, _ in legs.dense["e5_l2"][q["id"]][:8]]
            matches += chroma_ids == memory_ids
        return {"queries": len(questions), "top8_exact_order_matches": matches}


def _verify_rrf(legs: Legs, questions: list[dict]) -> dict:
    matches = 0
    for q in questions:
        qid = q["id"]
        fused = rrf_fuse(
            {
                "dense": legs.candidates([c for c, _ in legs.dense["e5_l2"][qid]]),
                "sparse": legs.candidates([c for c, _ in legs.sparse_hits[qid]]),
            },
            k=60,
            limit=40,
        )
        app_ids = [c.chunk_id for c in fused[:8]]
        local_ids = legs.rrf(qid, k=60)[:8]
        matches += app_ids == local_ids
    return {"queries": len(questions), "top8_exact_order_matches": matches}


def _bootstrap_and_tier(runs: list[dict], answerable_ids: list[str]) -> None:
    """Attach paired-bootstrap CIs vs the overall leader and a tier label."""
    eligible = [
        r for r in runs
        if r["ndcg_per_q"] and len(r["judged_ids"]) >= 0.9 * len(answerable_ids)
    ]
    leader = max(eligible, key=lambda r: r["metrics"]["ndcg@8"])
    rng = np.random.default_rng(7)
    for run in runs:
        base = np.array([leader["ndcg_per_q"][q] for q in leader["judged_ids"]])
        common = [q for q in run["judged_ids"] if q in leader["ndcg_per_q"]]
        cand = np.array([run["ndcg_per_q"][q] for q in common])
        basem = np.array([leader["ndcg_per_q"][q] for q in common])
        diffs = cand - basem
        if len(diffs):
            sample = rng.choice(diffs, size=(2000, len(diffs)), replace=True).mean(axis=1)
            lo, hi = np.percentile(sample, [2.5, 97.5]).tolist()
        else:
            lo, hi = -1.0, -1.0
        run["vs_leader"] = {"delta": float(np.mean(diffs)) if len(diffs) else None, "ci_low": lo, "ci_high": hi}
        flags = []
        if run["metrics"]["recall@8"] is not None and run["metrics"]["recall@8"] < 0.90:
            flags.append("fragile-recall")
        tier = "S" if lo > -0.005 else "A" if lo > -0.03 else "B" if lo > -0.10 else "C"
        if flags and tier == "S":
            tier = "A"
        run["tier"] = tier
        run["flags"] = flags


def _pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _render(runs, configs, official_ids, total_questions, answerable_count, chroma_match, rrf_match, elapsed) -> str:
    counts = {k: (len(v) if isinstance(v, list) else v) for k, v in configs.items()}
    ordered = sorted(runs, key=lambda r: (-(r["metrics"]["ndcg@8"] or 0), -(r["metrics"]["mrr"] or 0)))
    lines = [
        "# Retrieval Strategy Tournament",
        "",
        f"Questions: 27 official + 27 extended = {total_questions} total ({answerable_count} answerable, 3 traps); "
        "judged per question via the repo's marker ground truth (expected document + marker substring).",
        f"Runtime: {elapsed:.0f}s | chunk configs: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
        "",
        f"Dense-fidelity check (in-memory e5 L2 vs Chroma): {chroma_match['top8_exact_order_matches']}/{chroma_match['queries']} top-8 exact.",
        f"RRF-fidelity check (local vs app rrf_fuse): {rrf_match['top8_exact_order_matches']}/{rrf_match['queries']} top-8 exact.",
        "",
        "Tiers are relative to the overall leader, paired bootstrap over judged questions (2000 resamples): "
        "S = CI lower bound > -0.005 (statistically indistinguishable from leader), A > -0.03, B > -0.10, C otherwise. "
        "'fragile-recall' flags recall@8 < 90%.",
        "",
        "## Leaderboard (all questions, judged)",
        "",
        "| tier | strategy | family | nDCG@8 | MRR | R@1 | R@3 | R@8 | R@25 | P@1 | sufficient | trap-safe | official nDCG@8 | judged | delta vs leader [95% CI] |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in ordered:
        m = r["metrics"]
        delta = r.get("vs_leader", {})
        ci = (
            f"{delta['delta']:+.3f} [{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}]"
            if delta.get("delta") is not None else "—"
        )
        lines.append(
            f"| {r['tier']} | {r['strategy']} | {r['family']} | "
            f"{m['ndcg@8']:.3f} | {m['mrr']:.3f} | {_pct(m['recall@1'])} | {_pct(m['recall@3'])} | "
            f"{_pct(m['recall@8'])} | {_pct(m['recall@25'])} | {_pct(m['precision@1'])} | "
            f"{_pct(r['verdicts']['sufficient_rate'])} | {r['verdicts']['trap_safe']}/{r['verdicts']['trap_total']} | "
            f"{_pct(r['official']['ndcg@8'])} | {m['judged']} | {ci} |"
        )

    lines += ["", "## Chunking ablation (pipeline held constant)", ""]
    lines.append("| chunk config | chunks | pipeline | nDCG@8 | MRR | R@1 | R@8 | judged |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    chunk_runs = [r for r in runs if r["config"] != DEFAULT_CONFIG]
    for r in sorted(chunk_runs, key=lambda r: (r["config"], -(r["metrics"]["ndcg@8"] or 0))):
        m = r["metrics"]
        lines.append(
            f"| {r['config']} | {counts[r['config']]} | {r['strategy']} | {m['ndcg@8']:.3f} | "
            f"{m['mrr']:.3f} | {_pct(m['recall@1'])} | {_pct(m['recall@8'])} | {m['judged']} |"
        )

    unstable = [r for r in ordered if r["metrics"]["recall@25"] is not None and r["metrics"]["recall@25"] < 0.999]
    if unstable:
        lines += ["", "## Strategies with recall@25 < 100%", ""]
        for r in unstable:
            lines.append(
                f"- {r['config']} | {r['strategy']}: recall@25={_pct(r['metrics']['recall@25'])}, "
                f"recall@8={_pct(r['metrics']['recall@8'])}, judged={r['metrics']['judged']}"
            )
    return "\n".join(lines) + "\n"


def render_from_json() -> str:
    payload = json.loads((REPORTS_DIR / "tournament.json").read_text(encoding="utf-8"))
    official = [
        json.loads(line)
        for line in (EVALS_DIR / "datasets" / "golden.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    extended = [
        json.loads(line)
        for line in (EVALS_DIR / "datasets" / "golden_extended.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = official + extended
    report = _render(
        payload["runs"], payload["configs"], [q["id"] for q in official],
        len(questions), sum(1 for q in questions if q["answerable"]),
        payload["chroma_match"], payload["rrf_match"], 0.0,
    )
    (REPORTS_DIR / "tournament.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval strategy tournament")
    parser.add_argument("--from-json", action="store_true", help="Re-render the markdown from the last run")
    args = parser.parse_args()
    if args.from_json:
        print(render_from_json())
    else:
        asyncio.run(main())
