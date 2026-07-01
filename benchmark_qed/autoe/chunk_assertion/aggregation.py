# Copyright (c) 2025 Microsoft Corporation.
"""Aggregation logic for chunk-level assertion evaluation."""

from __future__ import annotations

from typing import Any

from benchmark_qed.autoe.data_model.chunk_assertion import EvalSummary


def summarize_at_k(
    *,
    k: int | None,
    question_set: Any,
    per_chunk_grades: dict[tuple[int, int], list[tuple[int, float, str]]],
    retrieved_chunk_counts: list[int],
    n_assertions_total: int,
    pass_threshold: float,
    total_calls: int,
    successful_calls: int,
    failed_calls: int,
) -> EvalSummary:
    """Aggregate per-(q, a) chunk grades at *k*; k=None means no truncation.

    For each (question, assertion) pair, identifies the best-scoring chunk
    within top-k and uses that to compute pass/fail metrics.

    Args:
        k: Truncation value (None = no truncation, full list)
        question_set: Question set object with .assertions field
        per_chunk_grades: Dict of (q_idx, a_idx) -> [(rank, score, grade), ...]
        retrieved_chunk_counts: Chunk count per question
        n_assertions_total: Total assertions evaluated
        pass_threshold: Score threshold for pass (0.5 = partial+ passes)
        total_calls: Total LLM calls made
        successful_calls: Successful LLM calls
        failed_calls: Failed LLM calls

    Returns
    -------
        EvalSummary with coverage, strict_coverage, mean_score metrics
    """
    # Aggregate scores: for each (q, a), find best chunk in top-k
    agg_assertion_scores: dict[tuple[int, int], float] = {}
    agg_assertion_grades: dict[tuple[int, int], str] = {}

    for q_key, rows in per_chunk_grades.items():
        if not rows:
            continue
        if k is not None:
            kept = [(score, grade) for (rank, score, grade) in rows if rank < k]
        else:
            kept = [(score, grade) for (_rank, score, grade) in rows]
        if not kept:
            continue
        # Best chunk is one with highest score
        best_idx = max(range(len(kept)), key=lambda i: kept[i][0])
        agg_assertion_scores[q_key] = kept[best_idx][0]
        agg_assertion_grades[q_key] = kept[best_idx][1]

    scored_assertions = len(agg_assertion_scores)
    passed = sum(1 for s in agg_assertion_scores.values() if s >= pass_threshold)
    mean_score = (
        sum(agg_assertion_scores.values()) / scored_assertions
        if scored_assertions
        else 0.0
    )

    # Per-question aggregation
    q_pass_rates_overall: dict[int, float] = {}
    q_pass_rates_strict: dict[int, float] = {}
    per_question_strength: dict[int, float] = {}

    for q_idx, q_row in enumerate(question_set.get("assertions", [])):
        assertions = q_row.get("assertions", [])
        if not assertions:
            continue

        q_passed_overall = 0
        q_passed_strict = 0
        q_score_sum = 0.0
        q_scored = 0

        for a_idx in range(len(assertions)):
            q_key = (q_idx, a_idx)
            if q_key not in agg_assertion_scores:
                continue
            q_scored += 1
            score = agg_assertion_scores[q_key]
            grade = agg_assertion_grades[q_key]
            if score >= pass_threshold:
                q_passed_overall += 1
            if grade == "full_support":
                q_passed_strict += 1
            q_score_sum += score

        if q_scored == 0:
            continue
        q_pass_rates_overall[q_idx] = q_passed_overall / q_scored
        q_pass_rates_strict[q_idx] = q_passed_strict / q_scored
        per_question_strength[q_idx] = q_score_sum / q_scored

    coverage = (
        sum(q_pass_rates_overall.values()) / len(q_pass_rates_overall)
        if q_pass_rates_overall
        else 0.0
    )
    strict_coverage = (
        sum(q_pass_rates_strict.values()) / len(q_pass_rates_strict)
        if q_pass_rates_strict
        else 0.0
    )

    per_query_metrics: dict[str, dict[str, float]] = {}
    for q_idx, cov_q in q_pass_rates_overall.items():
        per_query_metrics[f"{q_idx:04d}"] = {
            "coverage": float(cov_q),
            "strict_coverage": float(q_pass_rates_strict.get(q_idx, 0.0)),
            "mean_score": float(per_question_strength.get(q_idx, 0.0)),
        }

    mean_retrieved = (
        sum(min(c, k) if k is not None else c for c in retrieved_chunk_counts)
        / len(retrieved_chunk_counts)
        if retrieved_chunk_counts
        else 0.0
    )

    return EvalSummary(
        k=k,
        n_questions=len(q_pass_rates_overall),
        n_assertions=scored_assertions,
        n_questions_total=len(question_set.get("assertions", [])),
        n_assertions_total=n_assertions_total,
        coverage=coverage,
        strict_coverage=strict_coverage,
        mean_score=mean_score,
        mean_retrieved_chunks=mean_retrieved,
        pass_rate=passed / scored_assertions if scored_assertions else 0.0,
        total_calls=total_calls,
        successful_calls=successful_calls,
        failed_calls=failed_calls,
        eval_mode="chunk",
        per_query_metrics=per_query_metrics,
    )
