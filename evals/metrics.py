"""Aggregate metrics across queries.

Queries whose corpus-wide ground truth is empty (unanswerable traps) carry no
IR signal: their correct outcome is an honest INSUFFICIENT verdict, measured
separately by the runner. They ARE included in hit_rate denominators so that
retrieval misses stay visible in headline numbers.
"""


def recall_at_k(flags: list[bool], k: int) -> float | None:
    relevant_total = sum(1 for f in flags if f)
    if relevant_total == 0:
        return None
    retrieved_relevant = sum(1 for f in flags[:k] if f)
    return retrieved_relevant / relevant_total


def precision_at_k(flags: list[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for f in flags[:k] if f) / k


def mrr_from_flags(flags: list[bool]) -> float:
    for position, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / position
    return 0.0


def hit_at_k(flags: list[bool], k: int) -> bool:
    return any(flags[:k])


def mean(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def pct(value: float | None, digits: int = 1) -> str:
    """Format a [0,1] ratio as a percentage string for human-facing tables."""
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def aggregate(
    per_query_flags: dict[str, list[bool]],
    ks: tuple[int, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    judged = {qid: flags for qid, flags in per_query_flags.items() if any(flags)}
    result["judged_queries"] = len(judged)
    result["missed_queries"] = len(per_query_flags) - len(judged)

    for k in ks:
        recalls = [recall_at_k(flags, k) for flags in judged.values()]
        precs = [precision_at_k(flags, k) for flags in judged.values()]
        r = mean(recalls)
        p = mean(precs)
        if r is not None:
            result[f"recall@{k}"] = round(r, 4)
        if p is not None:
            result[f"precision@{k}"] = round(p, 4)
        if per_query_flags:
            hits = [hit_at_k(flags, k) for flags in per_query_flags.values()]
            result[f"hit_rate@{k}"] = round(sum(hits) / len(hits), 4)

    rr = [mrr_from_flags(flags) for flags in judged.values()]
    result["mrr"] = round(sum(rr) / len(rr), 4) if rr else 0.0
    return result
