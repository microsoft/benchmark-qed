# Copyright (c) 2025 Microsoft Corporation.
"""Statistical significance testing for assertion scores.

This module provides functions for comparing assertion scores across RAG methods
using statistical significance tests (ANOVA, Kruskal-Wallis, Friedman, etc.).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as rich_print

from benchmark_qed.autoe.utils.stats import GroupComparisonResult, compare_groups


def compare_assertion_scores_significance(
    output_dir: Path,
    generated_rags: list[str],
    question_sets: list[str],
    alpha: float = 0.05,
    correction_method: str = "holm",
) -> dict[str, GroupComparisonResult]:
    """Compare assertion scores across RAG methods using statistical tests.

    For each question set, loads per-question accuracy scores for each RAG method
    and performs omnibus and pairwise post-hoc tests to determine if differences
    are statistically significant.

    Args:
        output_dir: Directory containing assertion score results.
        generated_rags: List of RAG method names to compare.
        question_sets: List of question set names to analyze.
        alpha: Significance level (default 0.05).
        correction_method: P-value correction method
            ("holm", "bonferroni", "fdr_bh").

    Returns:
        Dictionary mapping question_set names to GroupComparisonResult.
    """
    results: dict[str, GroupComparisonResult] = {}

    for question_set in question_sets:
        rich_print(f"\n[bold]Statistical significance test for {question_set}[/bold]")

        # Collect per-question accuracy for each RAG method
        groups: dict[str, list[float]] = {}
        question_set_dir = output_dir / question_set

        for rag_method in generated_rags:
            summary_file = question_set_dir / f"{rag_method}_summary_by_question.csv"
            if not summary_file.exists():
                rich_print(
                    f"  [yellow]Warning: {summary_file} not found, "
                    f"skipping {rag_method}[/yellow]"
                )
                continue

            # Load per-question summary and calculate accuracy
            summary_df = pd.read_csv(summary_file)
            if (
                "success" not in summary_df.columns
                or "fail" not in summary_df.columns
            ):
                rich_print(
                    f"  [yellow]Warning: {summary_file} missing required "
                    f"columns[/yellow]"
                )
                continue

            # Calculate per-question accuracy: success / (success + fail)
            total = summary_df["success"] + summary_df["fail"]
            accuracy = summary_df["success"] / total.replace(0, np.nan)
            accuracy = accuracy.dropna().tolist()

            if len(accuracy) > 0:
                groups[rag_method] = accuracy

        # Need at least 2 groups to compare
        if len(groups) < 2:
            rich_print(
                "  [yellow]Insufficient data for comparison "
                "(need at least 2 RAG methods with data)[/yellow]"
            )
            continue

        # Run statistical comparison (paired=True since same questions across
        # RAG methods)
        comparison_result = compare_groups(
            groups, alpha=alpha, correction=correction_method, paired=True
        )
        results[question_set] = comparison_result

        # Print summary
        rich_print(f"  {comparison_result.omnibus.summary()}")
        if comparison_result.posthoc:
            rich_print(f"  {comparison_result.posthoc.summary()}")

        # Save detailed results to CSV
        _save_significance_results(
            comparison_result, question_set, question_set_dir, groups
        )

    return results


def compare_hierarchical_assertion_scores_significance(
    aggregated_scores: dict[str, pd.DataFrame],
    alpha: float = 0.05,
    correction_method: str = "holm",
    output_dir: Path | None = None,
) -> dict[str, GroupComparisonResult]:
    """Compare hierarchical assertion scores across RAG methods.

    Performs statistical significance tests for four metrics:
    - **global_pass_rate**: Per-question global assertion pass rate
    - **support_level**: Per-question average support level
    - **supporting_pass_rate**: Per-question supporting assertion pass rate
        (deduplicated aggregate)
    - **discovery_rate**: Per-question discovery rate (fraction with discovery)

    For each metric, runs omnibus tests (ANOVA/Friedman) and
    pairwise post-hoc tests with multiple comparison correction.

    Args:
        aggregated_scores: Dictionary mapping RAG method names to their
            aggregated hierarchical score DataFrames (from
            aggregate_hierarchical_scores). Each DataFrame must have columns:
            question, global_score, support_coverage, has_discovery.
        alpha: Significance level for hypothesis tests (default 0.05).
        correction_method: P-value correction method for post-hoc tests.
            Options: "holm" (default), "bonferroni", "fdr_bh".
        output_dir: Optional directory to save significance test results as CSV
            files.

    Returns:
        Dictionary mapping metric names to GroupComparisonResult objects
        containing omnibus test results and pairwise post-hoc comparisons.

    Raises:
        ValueError: If fewer than 2 RAG methods are provided.

    Example:
        >>> results = compare_hierarchical_assertion_scores_significance(
        ...     aggregated_scores={
        ...         "graphrag": graphrag_aggregated,
        ...         "vectorrag": vectorrag_aggregated,
        ...     },
        ...     alpha=0.05,
        ...     output_dir=Path("./output"),
        ... )
        >>> print(results["global_pass_rate"].omnibus.summary())
    """
    if len(aggregated_scores) < 2:
        msg = "Need at least 2 RAG methods to compare"
        raise ValueError(msg)

    rag_methods = list(aggregated_scores.keys())
    results: dict[str, GroupComparisonResult] = {}

    # Metrics to compare with their column names and aggregation methods
    # Note: support_level is per-global-assertion, supporting_pass_rate is
    # aggregate
    metrics_config = {
        "global_pass_rate": {
            "column": "global_score",
            "agg_func": "mean",  # Mean of binary scores = pass rate
            "description": "Global Assertion Pass Rate",
        },
        "support_level": {
            "column": "support_level",
            "agg_func": "mean",  # Average support level per question
            "description": "Support Level (per-global avg)",
        },
        "supporting_pass_rate": {
            "column": "n_supporting_passed",
            "agg_func": "_supporting_pass_rate",  # Special: compute from counts
            "description": "Supporting Pass Rate (aggregate)",
        },
        "discovery_rate": {
            "column": "has_discovery",
            "agg_func": "mean",  # Mean of binary = discovery rate
            "description": "Discovery Rate",
        },
    }

    for metric_name, config in metrics_config.items():
        rich_print(f"\n[bold]Significance test for {config['description']}[/bold]")

        # Collect per-question metric values for each RAG method
        groups: dict[str, list[float]] = {}

        for rag_method in rag_methods:
            df = aggregated_scores[rag_method]

            # Special handling for supporting_pass_rate
            if config["agg_func"] == "_supporting_pass_rate":
                # Compute per-question supporting pass rate
                # Count all supporting assertion evaluations (no deduplication)
                # because the same supporting assertion can have different results
                # under different global assertions
                per_question_values = []
                for _, group in df.groupby("question"):
                    total_supporting = 0
                    total_passed = 0
                    for support_results in group["support_results"]:
                        if support_results:
                            total_supporting += len(support_results)
                            total_passed += sum(1 for sr in support_results if sr["passed"])
                    if total_supporting > 0:
                        pass_rate = total_passed / total_supporting
                        per_question_values.append(pass_rate)

                if per_question_values:
                    groups[rag_method] = per_question_values
                    rich_print(
                        f"  {rag_method}: n={len(per_question_values)}, "
                        f"mean={np.mean(per_question_values):.3f}, "
                        f"std={np.std(per_question_values):.3f}"
                    )
                continue

            # Validate required column exists
            if config["column"] not in df.columns:
                rich_print(
                    f"  [yellow]Warning: {rag_method} missing column "
                    f"'{config['column']}', skipping[/yellow]"
                )
                continue

            # Aggregate per question
            per_question = df.groupby("question")[config["column"]].agg(
                config["agg_func"]
            )
            values = per_question.dropna().tolist()

            if len(values) > 0:
                groups[rag_method] = values
                rich_print(
                    f"  {rag_method}: n={len(values)}, "
                    f"mean={np.mean(values):.3f}, std={np.std(values):.3f}"
                )

        # Need at least 2 groups to compare
        if len(groups) < 2:
            rich_print(
                "  [yellow]Insufficient data for comparison "
                "(need at least 2 RAG methods with data)[/yellow]"
            )
            continue

        # Run statistical comparison (paired=True since same questions across
        # methods)
        comparison_result = compare_groups(
            groups, alpha=alpha, correction=correction_method, paired=True
        )
        results[metric_name] = comparison_result

        # Print summary
        rich_print(f"  {comparison_result.omnibus.summary()}")
        if comparison_result.posthoc:
            rich_print(f"  {comparison_result.posthoc.summary()}")

        # Save results if output_dir provided
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _save_hierarchical_significance_results(
                comparison_result,
                metric_name,
                config["description"],
                output_dir,
                groups,
            )

    return results


def _save_hierarchical_significance_results(
    result: GroupComparisonResult,
    metric_name: str,
    metric_description: str,
    output_dir: Path,
    groups: dict[str, list[float]],
) -> None:
    """Save hierarchical significance test results to CSV files.

    Args:
        result: GroupComparisonResult from statistical comparison.
        metric_name: Short name of the metric (e.g., "global_pass_rate").
        metric_description: Human-readable description of the metric.
        output_dir: Directory to save CSV files.
        groups: Dictionary of RAG method names to their per-question values.
    """
    # Save group statistics for this metric
    group_stats = []
    for name, data in groups.items():
        group_stats.append(
            {
                "rag_method": name,
                "metric": metric_name,
                "n_questions": len(data),
                "mean": np.mean(data),
                "std": np.std(data),
                "median": np.median(data),
                "min": np.min(data),
                "max": np.max(data),
            }
        )
    stats_df = pd.DataFrame(group_stats)
    stats_df.to_csv(
        output_dir / f"significance_{metric_name}_group_stats.csv", index=False
    )

    # Save omnibus test result
    omnibus_df = pd.DataFrame(
        [
            {
                "metric": metric_name,
                "description": metric_description,
                "test_name": result.omnibus.test_name,
                "statistic": result.omnibus.statistic,
                "p_value": result.omnibus.p_value,
                "is_significant": result.omnibus.is_significant,
                "alpha": result.omnibus.alpha,
                "is_normal": result.omnibus.is_normal,
            }
        ]
    )
    omnibus_df.to_csv(
        output_dir / f"significance_{metric_name}_omnibus.csv", index=False
    )

    # Save pairwise comparisons if available
    if result.posthoc and result.posthoc.comparisons:
        pairwise_data = [
            {
                "metric": metric_name,
                "group1": c.group1,
                "group2": c.group2,
                "statistic": c.statistic,
                "p_value_raw": c.p_value_raw,
                "p_value_corrected": c.p_value_corrected,
                "is_significant": c.is_significant,
            }
            for c in result.posthoc.comparisons
        ]
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df.to_csv(
            output_dir / f"significance_{metric_name}_pairwise.csv", index=False
        )


def _save_significance_results(
    result: GroupComparisonResult,
    question_set: str,
    output_dir: Path,
    groups: dict[str, list[float]],
) -> None:
    """Save significance test results to CSV files.

    Args:
        result: GroupComparisonResult from statistical comparison.
        question_set: Name of the question set being analyzed.
        output_dir: Directory to save CSV files.
        groups: Dictionary of RAG method names to their per-question accuracy
            values.
    """
    # Save group statistics
    group_stats = []
    for name, data in groups.items():
        group_stats.append(
            {
                "rag_method": name,
                "n_questions": len(data),
                "mean_accuracy": np.mean(data),
                "std_accuracy": np.std(data),
                "median_accuracy": np.median(data),
            }
        )
    stats_df = pd.DataFrame(group_stats)
    stats_df.to_csv(output_dir / "significance_group_stats.csv", index=False)

    # Save omnibus test result
    omnibus_df = pd.DataFrame(
        [
            {
                "question_set": question_set,
                "test_name": result.omnibus.test_name,
                "statistic": result.omnibus.statistic,
                "p_value": result.omnibus.p_value,
                "is_significant": result.omnibus.is_significant,
                "alpha": result.omnibus.alpha,
                "is_normal": result.omnibus.is_normal,
            }
        ]
    )
    omnibus_df.to_csv(output_dir / "significance_omnibus.csv", index=False)

    # Save pairwise comparisons if available
    if result.posthoc and result.posthoc.comparisons:
        pairwise_data = [
            {
                "group1": c.group1,
                "group2": c.group2,
                "statistic": c.statistic,
                "p_value_raw": c.p_value_raw,
                "p_value_corrected": c.p_value_corrected,
                "is_significant": c.is_significant,
            }
            for c in result.posthoc.comparisons
        ]
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df.to_csv(output_dir / "significance_pairwise.csv", index=False)
