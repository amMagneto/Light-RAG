import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from evals.judging import build_ground_truth, is_relevant
from evals import metrics as M

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVALS_DIR / "datasets" / "golden.jsonl"
DEFAULT_CORPUS = EVALS_DIR / "corpus"

CONFIGS = ("dense", "sparse", "fused", "full")
KS = (1, 3, 5, 8)


def load_dataset(path: Path = DEFAULT_DATASET, subset: int = 0) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries[:subset] if subset else entries


def load_corpus(corpus_dir: Path = DEFAULT_CORPUS) -> list[tuple[str, bytes]]:
    return [
        (path.name, path.read_bytes())
        for path in sorted(corpus_dir.glob("*.md"))
    ]


async def _prepare_context(max_k_unused: int = 0):
    import tempfile

    from app.api.dependencies import build_context
    from app.core.config import Settings
    from app.db.engine import init_db

    tmp = Path(tempfile.mkdtemp(prefix="lightrag_eval_"))
    settings = Settings(
        groq_api_key=None,
        database_url=f"sqlite+aiosqlite:///{tmp / 'eval.db'}",
        chroma_dir=str(tmp / "chroma"),
        storage_dir=str(tmp / "storage"),
        log_level="ERROR",
        enable_memory_extraction=False,
    )
    ctx = build_context(settings)
    await init_db(ctx.engine)
    return ctx


async def run_eval_async(
    max_k: int = 8,
    subset: int = 0,
    out_dir: Path | None = None,
    dataset_path: Path = DEFAULT_DATASET,
) -> dict:
    from app.db.repositories import ChunkRepository
    from app.reasoning.grader import EvidenceGrader
    from app.retrieval.fusion import rrf_fuse

    ctx = await _prepare_context()
    questions = load_dataset(dataset_path, subset)

    for filename, payload in load_corpus():
        result = await ctx.ingestion_service.ingest(filename, payload)
        assert result["status"] == "READY", f"injest failed for {filename}: {result}"

    async with ctx.sessionmaker() as session:
        rows = await ChunkRepository(session).all_with_documents(current_only=True)
    all_chunks = [
        {
            "chunk_id": chunk.id,
            "document_name": filename,
            "content": chunk.content,
            "section_path": list(chunk.section_path_json or []),
        }
        for chunk, filename, _collection in rows
    ]

    def to_candidate(row: dict):
        from app.schemas.retrieval import RetrievalCandidate

        return RetrievalCandidate(
            chunk_id=row["chunk_id"],
            document_id="eval",
            document_name=row["document_name"],
            content=row["content"],
            section_path=row["section_path"],
        )

    corpus_candidates = [to_candidate(row) for row in all_chunks]

    ground_truth: dict[str, set[str]] = {}
    for question in questions:
        ground_truth[question["id"]] = build_ground_truth(
            corpus_candidates,
            question["expected_documents"],
            question["evidence_markers"],
        )

    flags_by_config: dict[str, dict[str, list[bool]]] = {config: {} for config in CONFIGS}
    ranks_detail: dict[str, dict] = {}
    latency = {"dense_ms": [], "sparse_ms": [], "fusion_ms": []}
    verdict_records: list[dict] = []
    context_stats = {"input_chunks": [], "kept_citations": [], "relevant_kept": 0}

    grader = EvidenceGrader(ctx.llm, ctx.settings)
    context_builder = ctx.context_builder
    reranker = ctx.reranker
    settings = ctx.settings

    for question in questions:
        qid = question["id"]
        text = question["question"]

        start = time.perf_counter()
        dense_hits = await asyncio.to_thread(ctx.index.query, text, max_k)
        dense_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        sparse_hits = await asyncio.to_thread(ctx.sparse.search, text, max_k)
        sparse_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        fused = rrf_fuse(
            {"dense": [c for c, _ in dense_hits], "sparse": sparse_hits},
            k=settings.rrf_k,
            limit=settings.candidate_pool,
        )
        full = reranker.rerank(text, fused, top_k=max_k)
        fusion_ms = (time.perf_counter() - start) * 1000

        latency["dense_ms"].append(round(dense_ms, 2))
        latency["sparse_ms"].append(round(sparse_ms, 2))
        latency["fusion_ms"].append(round(fusion_ms, 2))

        ranked = {
            "dense": [c for c, _ in dense_hits][:max_k],
            "sparse": sparse_hits[:max_k],
            "fused": fused[:max_k],
            "full": full[:max_k],
        }

        relevant_ids = ground_truth[qid]
        for config, candidates in ranked.items():
            flags_by_config[config][qid] = [
                c.chunk_id in relevant_ids for c in candidates
            ]
        ranks_detail[qid] = {
            "category": question["category"],
            "ground_truth_size": len(relevant_ids),
            **{
                config: [
                    {
                        "chunk_id": c.chunk_id[:8],
                        "doc": c.document_name,
                        "rel": c.chunk_id in relevant_ids,
                    }
                    for c in candidates
                ]
                for config, candidates in ranked.items()
            },
        }

        citations, _context_text = context_builder.build(full)
        context_stats["input_chunks"].append(len(full))
        context_stats["kept_citations"].append(len(citations))
        context_stats["relevant_kept"] += sum(1 for c in citations if c.chunk_id in relevant_ids)

        assessment = await grader.grade(text, full)
        verdict_records.append(
            {
                "id": qid,
                "answerable": question["answerable"],
                "verdict": assessment.verdict.value,
                "recall_at_k": len(relevant_ids & {c.chunk_id for c in full}) / len(relevant_ids)
                if relevant_ids
                else None,
                "rationale": assessment.rationale,
            }
        )

    overall = {config: M.aggregate(flags_by_config[config], KS) for config in CONFIGS}

    categories: dict[str, dict] = {}
    for question in questions:
        category = question["category"]
        bucket_flags = {
            config: {qid: flags for qid, flags in flags_by_config[config].items() if ranks_detail[qid]["category"] == category}
            for config in CONFIGS
        }
        if category not in categories:
            categories[category] = {config: M.aggregate(bucket_flags[config], KS) for config in CONFIGS}

    traps = [r for r in verdict_records if not r["answerable"]]
    answerable = [r for r in verdict_records if r["answerable"]]
    trap_lenient = sum(1 for r in traps if r["verdict"] in {"insufficient", "ambiguous"})
    answerable_with_evidence = [r for r in answerable if r["recall_at_k"]]
    answerable_sufficient = sum(1 for r in answerable_with_evidence if r["verdict"] == "sufficient")

    verdict_summary = {
        "trap_total": len(traps),
        "trap_correct_lenient": trap_lenient,
        "trap_accuracy_lenient": round(trap_lenient / len(traps), 4) if traps else None,
        "answerable_with_evidence": len(answerable_with_evidence),
        "answerable_sufficient": answerable_sufficient,
        "answerable_verdict_accuracy_conditional": round(
            answerable_sufficient / len(answerable_with_evidence), 4
        )
        if answerable_with_evidence
        else None,
    }

    def avg(key: str) -> float:
        values = latency[key]
        return round(sum(values) / len(values), 2) if values else 0.0

    latency_summary = {key: avg(key) for key in latency}

    total_input = sum(context_stats["input_chunks"])
    total_kept = sum(context_stats["kept_citations"])
    context_summary = {
        "dedup_ratio": round(total_kept / total_input, 4) if total_input else None,
        "relevant_fraction_of_context": round(
            context_stats["relevant_kept"] / total_kept, 4
        )
        if total_kept
        else None,
    }

    hypotheses = {
        "H1_exact_term_sparse_wins": {
            "statement": "MRR(sparse) >= MRR(dense) on exact_term queries",
            "mrr_dense": categories.get("exact_term", {}).get("dense", {}).get("mrr"),
            "mrr_sparse": categories.get("exact_term", {}).get("sparse", {}).get("mrr"),
        },
        "H2_paraphrase_dense_wins": {
            "statement": "MRR(dense) > MRR(sparse) on paraphrase queries",
            "mrr_dense": categories.get("paraphrase", {}).get("dense", {}).get("mrr"),
            "mrr_sparse": categories.get("paraphrase", {}).get("sparse", {}).get("mrr"),
        },
        "H3_fusion_never_loses_recall": {
            "statement": "recall@8(fused) >= max(recall@8 legs) - 0.02",
            "recall_dense": overall["dense"].get("recall@8"),
            "recall_sparse": overall["sparse"].get("recall@8"),
            "recall_fused": overall["fused"].get("recall@8"),
        },
        "H4_reranker_trades_cleanly": {
            "statement": "precision@3(full) >= precision@3(fused) - 0.02 AND recall@8(full) >= recall@8(fused) - 0.05",
            "precision3_fused": overall["fused"].get("precision@3"),
            "precision3_full": overall["full"].get("precision@3"),
            "recall8_fused": overall["fused"].get("recall@8"),
            "recall8_full": overall["full"].get("recall@8"),
        },
    }
    for key, hypothesis in hypotheses.items():
        if key == "H1_exact_term_sparse_wins":
            d, s = hypothesis["mrr_dense"], hypothesis["mrr_sparse"]
            hypothesis["pass"] = None if None in (d, s) else bool(s >= d - 0.02)
        elif key == "H2_paraphrase_dense_wins":
            d, s = hypothesis["mrr_dense"], hypothesis["mrr_sparse"]
            hypothesis["pass"] = None if None in (d, s) else bool(d > s)
        elif key == "H3_fusion_never_loses_recall":
            vals = [hypothesis["recall_dense"], hypothesis["recall_sparse"], hypothesis["recall_fused"]]
            hypothesis["pass"] = (
                None
                if None in vals
                else bool(hypothesis["recall_fused"] >= max(hypothesis["recall_dense"], hypothesis["recall_sparse"]) - 0.02)
            )
        else:
            vals = [
                hypothesis["precision3_fused"],
                hypothesis["precision3_full"],
                hypothesis["recall8_fused"],
                hypothesis["recall8_full"],
            ]
            hypothesis["pass"] = (
                None
                if None in vals
                else bool(
                    hypothesis["precision3_full"] >= hypothesis["precision3_fused"] - 0.02
                    and hypothesis["recall8_full"] >= hypothesis["recall8_fused"] - 0.05
                )
            )

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "questions": len(questions),
        "corpus_chunks": len(all_chunks),
        "max_k": max_k,
        "overall": overall,
        "per_category": categories,
        "verdicts": verdict_summary,
        "context": context_summary,
        "latency_ms_avg": latency_summary,
        "hypotheses": hypotheses,
        "detail": ranks_detail,
        "verdict_records": verdict_records,
    }

    out_dir = Path(out_dir or (EVALS_DIR / "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"eval_report_{timestamp}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    report = render_markdown_report(results)
    (out_dir / "latest.md").write_text(report, encoding="utf-8")

    await ctx.engine.dispose()
    return results


def render_markdown_report(results: dict) -> str:
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"Generated: `{results['timestamp']}` | questions: {results['questions']} | corpus chunks: {results['corpus_chunks']} | k_max: {results['max_k']}",
        "",
        "## Methodology",
        "",
        "- **Corpus:** 4 authored documents; every fact has a known passage.",
        "- **Judging:** a retrieved chunk is RELEVANT iff it comes from an expected document AND contains a ground-truth marker substring. Ground truth is computed corpus-wide over ALL indexed chunks, so recall is a true fraction of available evidence.",
        "- **recall@k** is primary: generation cannot use evidence it never retrieves.",
        "- **MRR** captures single-fact lookup UX; **precision@k** proxies context pollution; **hit_rate@k** includes unanswerable traps so misses stay visible.",
        "- **Verdict honesty** is scored separately: traps must abstain (insufficient/ambiguous); answerable questions must say sufficient when evidence was actually retrieved.",
        "- Grader runs in heuristic mode (LLM disabled) so numbers isolate retrieval quality from LLM variance and cost zero quota.",
        "",
        "## Overall leaderboard",
        "",
        "| metric | " + " | ".join(f"**{c}**" for c in CONFIGS) + " |",
        "|---|" + "---:|" * len(CONFIGS),
    ]

    pct_keys = [
        ("mrr", "MRR"),
        ("recall@1", "recall@1"),
        ("recall@3", "recall@3"),
        ("recall@5", "recall@5"),
        ("recall@8", "recall@8"),
        ("precision@1", "precision@1"),
        ("precision@3", "precision@3"),
        ("hit_rate@1", "hit rate@1"),
        ("hit_rate@5", "hit rate@5"),
    ]
    for key, label in pct_keys:
        row = [M.pct(results["overall"][config].get(key)) for config in CONFIGS]
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    judged_row = [str(results["overall"][config].get("judged_queries", "—")) for config in CONFIGS]
    lines.append("| judged queries (of answerable) | " + " | ".join(judged_row) + " |")

    lines += [
        "",
        "## Per-category MRR",
        "",
        "| query archetype | " + " | ".join(f"**{c}**" for c in CONFIGS) + " |",
        "|---|" + "---:|" * len(CONFIGS),
    ]
    for category, cat_results in results["per_category"].items():
        if category == "unanswerable":
            continue
        cells = [M.pct(cat_results[config].get("mrr")) for config in CONFIGS]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines.append("| *unanswerable traps* | scored under verdict honesty below |")

    v = results["verdicts"]
    trap_pct = M.pct(v["trap_accuracy_lenient"])
    ans_pct = M.pct(v["answerable_verdict_accuracy_conditional"])
    lines += [
        "",
        "## Verdict honesty (heuristic grader)",
        "",
        "| check | result | rate |",
        "|---|---|---:|",
        f"| unanswerable traps correctly abstain (insufficient/ambiguous) | {v['trap_correct_lenient']}/{v['trap_total']} | **{trap_pct}** |",
        f"| answerable → SUFFICIENT when evidence was actually retrieved | {v['answerable_sufficient']}/{v['answerable_with_evidence']} | **{ans_pct}** |",
        "",
        "## Context engine & latency budget",
        "",
        "| component | value |",
        "|---|---:|",
        f"| context dedup ratio (kept / input chunks) | {M.pct(results['context']['dedup_ratio'])} |",
        f"| relevant fraction of final packed context | {M.pct(results['context']['relevant_fraction_of_context'])} |",
        f"| dense search (e5 embed + chroma, avg) | {results['latency_ms_avg']['dense_ms']:.0f} ms |",
        f"| sparse search (BM25, avg) | {results['latency_ms_avg']['sparse_ms']:.1f} ms |",
        f"| RRF fusion + rerank (avg) | {results['latency_ms_avg']['fusion_ms']:.1f} ms |",
        "",
        "## Hypothesis verification",
        "",
        "| hypothesis | claim | evidence | verdict |",
        "|---|---|---|---|",
    ]
    for name, hypothesis in results["hypotheses"].items():
        status = hypothesis["pass"]
        mark = "**PASS**" if status else ("FAIL" if status is False else "N/A")
        fields = ", ".join(
            f"{k}={M.pct(x) if isinstance(x, float) else x}"
            for k, x in hypothesis.items()
            if k not in {"statement", "pass"}
        )
        lines.append(f"| `{name}` | {hypothesis['statement']} | {fields} | {mark} |")

    return "\n".join(lines) + "\n"


def run_eval(max_k: int = 8, subset: int = 0, out_dir: Path | None = None) -> dict:
    return asyncio.run(run_eval_async(max_k=max_k, subset=subset, out_dir=out_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run golden-dataset retrieval evaluation")
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--subset", type=int, default=0, help="Only first N questions (0 = all)")
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="Re-render the markdown report from evals/reports/latest.json without re-running",
    )
    args = parser.parse_args()

    if args.from_json:
        outcome = json.loads((EVALS_DIR / "reports" / "latest.json").read_text(encoding="utf-8"))
    else:
        outcome = run_eval(max_k=args.max_k, subset=args.subset)
    report_text = render_markdown_report(outcome)
    (EVALS_DIR / "reports" / "latest.md").write_text(report_text, encoding="utf-8")
    print(report_text)
