import pytest

from evals.runner import run_eval

FLOORS = {
    "recall_at_8_full": 0.80,
    "mrr_full": 0.55,
    "hit_rate_5_full": 0.70,
    "trap_accuracy_lenient": 0.60,
}


@pytest.fixture(scope="module")
def eval_results():
    return run_eval(max_k=8)


def test_overall_retrieval_quality_floors(eval_results):
    full = eval_results["overall"]["full"]
    assert full["recall@8"] >= FLOORS["recall_at_8_full"], f"recall@8={full['recall@8']}"
    assert full["mrr"] >= FLOORS["mrr_full"], f"mrr={full['mrr']}"
    assert full["hit_rate@5"] >= FLOORS["hit_rate_5_full"], f"hit@5={full['hit_rate@5']}"


def test_fusion_beats_or_matches_best_single_leg(eval_results):
    overall = eval_results["overall"]
    best_single = max(overall["dense"].get("recall@8", 0), overall["sparse"].get("recall@8", 0))
    assert overall["fused"]["recall@8"] >= best_single - 0.05


def test_trap_abstention_works(eval_results):
    verdicts = eval_results["verdicts"]
    assert verdicts["trap_total"] >= 3
    assert verdicts["trap_accuracy_lenient"] >= FLOORS["trap_accuracy_lenient"]


def test_grader_recognizes_sufficient_evidence(eval_results):
    """Rank-1 evidence must yield SUFFICIENT verdicts from the heuristic grader."""
    verdicts = eval_results["verdicts"]
    assert verdicts["answerable_with_evidence"] >= 20
    assert verdicts["answerable_verdict_accuracy_conditional"] >= 0.75


def test_context_engine_keeps_relevant_evidence(eval_results):
    """Expected context fraction = mean ground-truth size / k (single-fact corpus).
    Actual must stay within 2x of that theoretical ceiling."""
    detail = eval_results["detail"]
    answerable_gt = [v["ground_truth_size"] for v in detail.values() if v["ground_truth_size"]]
    mean_gt = sum(answerable_gt) / len(answerable_gt)
    expected = mean_gt / eval_results["max_k"]
    actual = eval_results["context"]["relevant_fraction_of_context"]
    assert actual is not None
    assert actual >= 0.5 * expected, f"actual={actual} expected_ceiling={expected:.3f}"
