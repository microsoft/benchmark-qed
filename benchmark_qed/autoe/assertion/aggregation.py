# Copyright (c) 2025 Microsoft Corporation.
"""Aggregation functions for assertion scores.

This module provides functions for aggregating assertion scores across trials
and summarizing results by question.
"""

from typing import Any

import pandas as pd


def aggregate_hierarchical_scores(
    scores_df: pd.DataFrame,
    pass_threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate hierarchical assertion scores across trials.

    Computes aggregated metrics for each unique assertion:
    - global_score: Binary (0/1) based on mean of global_passed across trials
    - global_score_mean: Mean of global_passed across trials
    - n_supporting: Total count of supporting assertions for this global
        assertion
    - n_supporting_passed: Count of supporting assertions that passed
    - support_level: Ratio of supporting assertions that passed
        (per-global-assertion)
    - has_discovery: Majority vote across trials (only when global_score=1)
    - discovery_reasoning: From first trial with discovery (if any)
    - global_score_overridden: True if global_score was forced to 0 due to
        having no support and no discovery

    Note: has_discovery is only meaningful when the global assertion passes.
    If global_score=0, has_discovery is automatically set to False since there
    is no "discovery" when the answer doesn't satisfy the main claim.

    Additionally, if a global assertion passes but has support_level=0 and
    has_discovery=False, the global_score is overridden to 0 (fail) since
    a pass without any supporting evidence or discovery is suspicious.

    Args:
        scores_df: DataFrame from get_hierarchical_assertion_scores() with
            per-trial results including global_passed, supporting_passed,
            has_discovery columns.
        pass_threshold: Threshold for determining if an assertion passed.
            Default 0.5.

    Returns:
        DataFrame with aggregated scores per assertion containing:
            - question: The question text
            - assertion: The global assertion text
            - global_score: Binary score (1 if mean > threshold, else 0)
            - global_score_mean: Mean global_passed across trials
            - global_score_overridden: Whether global_score was forced to 0
            - n_supporting: Total number of supporting assertions
            - n_supporting_passed: Number of supporting assertions that passed
            - support_level: Ratio of passed supporting assertions (per-global)
            - support_results: List of dicts with id, passed, pass_rate for each
            - has_discovery: Whether discovery was detected (only if global
                passed)
            - discovery_reasoning: Explanation from first discovery trial
    """

    def _aggregate_supporting(group: pd.DataFrame) -> dict[str, Any]:
        """Aggregate supporting assertion scores for a group of trials.

        Returns metrics at two levels:
        - Per-global-assertion: support_level (fraction of supporting passed)
        - Direct counts: n_supporting, n_supporting_passed for raw tallies
        """
        # Get all supporting_results lists (one per trial)
        all_results = group["supporting_results"].tolist()

        if not all_results or not all_results[0]:
            return {
                "n_supporting": 0,
                "n_supporting_passed": 0,
                "support_level": 0.0,
                "support_results": [],
            }

        n_supporting = len(all_results[0])
        n_trials = len(all_results)

        # Build aggregated results per supporting assertion
        aggregated_results = []
        passed_count = 0

        for i in range(n_supporting):
            # Get ID from first trial (should be consistent)
            sa_id = all_results[0][i]["id"]

            # Calculate pass rate across trials
            trials_passed = sum(
                1 for trial_results in all_results if trial_results[i]["passed"]
            )
            pass_rate = trials_passed / n_trials

            # Determine if this supporting assertion passed (using threshold)
            sa_passed = pass_rate > pass_threshold
            if sa_passed:
                passed_count += 1

            # Get representative reasoning (from first trial that matches
            # majority)
            reasoning = ""
            for trial_results in all_results:
                if trial_results[i]["passed"] == sa_passed:
                    reasoning = trial_results[i]["reasoning"]
                    break

            aggregated_results.append(
                {
                    "id": sa_id,
                    "passed": sa_passed,
                    "pass_rate": pass_rate,
                    "reasoning": reasoning,
                }
            )

        level = passed_count / n_supporting if n_supporting > 0 else 0.0

        return {
            "n_supporting": n_supporting,
            "n_supporting_passed": passed_count,
            "support_level": level,
            "support_results": aggregated_results,
        }

    def _aggregate_discovery(group: pd.DataFrame) -> dict[str, Any]:
        """Aggregate discovery detection across trials using majority vote."""
        discoveries = group["has_discovery"].tolist()
        majority_threshold = len(discoveries) / 2
        has_discovery = sum(discoveries) > majority_threshold

        # Get reasoning from first trial with discovery
        discovery_reasoning = ""
        if has_discovery:
            for _, row in group.iterrows():
                if row["has_discovery"] and row["discovery_reasoning"]:
                    discovery_reasoning = row["discovery_reasoning"]
                    break

        return {
            "has_discovery": has_discovery,
            "discovery_reasoning": discovery_reasoning,
        }

    # Group by assertion (question + assertion text)
    grouped = scores_df.groupby(["question", "assertion"])

    results = []
    for (question, assertion), group in grouped:
        # Aggregate global score
        global_passed_mean = group["global_passed"].mean()
        global_score = 1 if global_passed_mean > pass_threshold else 0

        # Aggregate supporting assertions
        support_metrics = _aggregate_supporting(group)

        # Aggregate discovery
        discovery_metrics = _aggregate_discovery(group)

        # Check for suspicious pass: global passed but no support and no
        # discovery Override global_score to 0 in this case
        global_score_overridden = False
        if (
            global_score == 1
            and support_metrics["support_level"] == 0.0
            and not discovery_metrics["has_discovery"]
        ):
            global_score = 0
            global_score_overridden = True

        # Get supporting_assertions from the first row (should be the same for
        # all trials)
        supporting_assertions = (
            group["supporting_assertions"].iloc[0]
            if "supporting_assertions" in group.columns
            else []
        )

        # Get reasoning from first row
        reasoning = group["reasoning"].iloc[0] if "reasoning" in group.columns else ""

        results.append(
            {
                "question": question,
                "assertion": assertion,
                "global_score": global_score,
                "global_score_mean": global_passed_mean,
                "global_score_overridden": global_score_overridden,
                "reasoning": reasoning,
                "supporting_assertions": supporting_assertions,
                **support_metrics,
                **discovery_metrics,
            }
        )

    return pd.DataFrame(results)


def summarize_hierarchical_by_question(
    aggregated_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize hierarchical assertion results by question.

    Args:
        aggregated_df: DataFrame from aggregate_hierarchical_scores() with
            per-assertion aggregated metrics.

    Returns:
        DataFrame with per-question summary containing:
            - question: The question text
            - assertions_passed: Number of global assertions that passed
            - assertions_failed: Number of global assertions that failed
            - assertions_total: Total number of assertions
            - global_pass_rate: Ratio of passed assertions
            - avg_support_level: Average support level across assertions
            - n_supporting_unique: Total unique supporting assertions
                (deduplicated)
            - n_supporting_passed_unique: Unique supporting assertions that
                passed
            - supporting_pass_rate: Aggregate pass rate for unique supporting
                assertions
            - n_with_discovery: Number of assertions with discovery detected
            - n_with_support: Number of assertions with at least one supporting
                passed
            - n_with_both: Number of assertions with both support and discovery
            - n_support_only: Number of assertions with support but no discovery
            - pct_with_discovery: Percentage of assertions with discovery
            - pct_with_support: Percentage of assertions with support
            - pct_with_both: Percentage with both
    """

    def _compute_unique_supporting_stats(group: pd.DataFrame) -> pd.Series:
        """Compute deduplicated supporting assertion statistics.

        Supporting assertions may appear in multiple global assertions.
        This function deduplicates by assertion ID before computing stats.
        """
        # Collect all supporting results from all assertions in this question
        unique_supporting: dict[str, bool] = {}  # id -> passed (latest wins)

        for support_results in group["support_results"]:
            if support_results:
                for sr in support_results:
                    # Use the assertion ID as key for deduplication
                    unique_supporting[sr["id"]] = sr["passed"]

        n_unique = len(unique_supporting)
        n_passed = sum(unique_supporting.values())
        pass_rate = n_passed / n_unique if n_unique > 0 else 0.0

        return pd.Series(
            {
                "n_supporting_unique": n_unique,
                "n_supporting_passed_unique": n_passed,
                "supporting_pass_rate": pass_rate,
            }
        )

    # Basic aggregations
    summary = (
        aggregated_df.groupby("question")
        .agg(
            assertions_passed=("global_score", lambda x: (x == 1).sum()),
            assertions_failed=("global_score", lambda x: (x == 0).sum()),
            assertions_total=("global_score", "count"),
            avg_support_level=("support_level", "mean"),
            n_with_discovery=("has_discovery", "sum"),
            n_with_support=("n_supporting_passed", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    summary["global_pass_rate"] = (
        summary["assertions_passed"] / summary["assertions_total"]
    )

    # Compute discovery breakdown metrics
    # n_with_both: has_discovery=True AND n_supporting_passed > 0
    # n_support_only: n_supporting_passed > 0 AND has_discovery=False
    both_counts = (
        aggregated_df.groupby("question")
        .apply(  # type: ignore[call-overload]
            lambda g: pd.Series(
                {
                    "n_with_both": (
                        (g["has_discovery"]) & (g["n_supporting_passed"] > 0)
                    ).sum(),
                    "n_support_only": (
                        (~g["has_discovery"]) & (g["n_supporting_passed"] > 0)
                    ).sum(),
                }
            ),
            include_groups=False,  # type: ignore[arg-type]
        )
        .reset_index()
    )
    summary = summary.merge(both_counts, on="question", how="left")

    # Compute percentages
    summary["pct_with_discovery"] = (
        summary["n_with_discovery"] / summary["assertions_total"] * 100
    )
    summary["pct_with_support"] = (
        summary["n_with_support"] / summary["assertions_total"] * 100
    )
    summary["pct_with_both"] = (
        summary["n_with_both"] / summary["assertions_total"] * 100
    )

    # Compute deduplicated supporting assertion stats
    unique_stats = (
        aggregated_df.groupby("question")
        .apply(  # type: ignore[call-overload]
            _compute_unique_supporting_stats, include_groups=False  # type: ignore[arg-type]
        )
        .reset_index()
    )

    # Merge the unique supporting stats
    summary = summary.merge(unique_stats, on="question", how="left")

    return summary
