# Copyright (c) 2025 Microsoft Corporation.
"""Statistical significance testing for assertion scores.

This module provides functions for comparing assertion scores across RAG methods
using statistical significance tests (ANOVA, Kruskal-Wallis, Friedman, etc.).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich import print as rich_print

from benchmark_qed.autoe.utils.stats import (
    GroupComparisonResult,
    compare_groups,
    run_clustered_permutation_test,
)


def summarize_significance_results(
    results: dict[str, GroupComparisonResult],
    groups: dict[str, dict[str, list[float]]] | None = None,
) -> pd.DataFrame:
    """Create a summary DataFrame from significance test results.

    Produces a table with one row per metric for omnibus results, plus
    one row per pairwise comparison for each metric that has post-hoc
    tests.  When ``groups`` is provided, per-method mean values are
    included in pairwise rows.

    Args:
        results: Dictionary mapping metric names to
            GroupComparisonResult objects (as returned by
            ``compare_hierarchical_assertion_scores_significance``).
        groups: Optional nested dict ``{metric: {rag_method: values}}``
            for embedding per-method means in the pairwise rows.

    Returns:
        DataFrame with columns: metric, level, test_name, group1,
        group2, statistic, p_value, is_significant, group1_mean,
        group2_mean.  ``level`` is ``"omnibus"`` or ``"pairwise"``.
    """
    rows: list[dict[str, object]] = []

    for metric, result in results.items():
        # Omnibus row
        rows.append({
            "metric": metric,
            "level": "omnibus",
            "test_name": result.omnibus.test_name,
            "group1": "",
            "group2": "",
            "statistic": round(result.omnibus.statistic, 4),
            "p_value": result.omnibus.p_value,
            "is_significant": result.omnibus.is_significant,
            "group1_mean": None,
            "group2_mean": None,
        })

        # Pairwise rows
        if result.posthoc:
            for comp in result.posthoc.comparisons:
                g1_mean = None
                g2_mean = None
                if groups and metric in groups:
                    metric_groups = groups[metric]
                    if comp.group1 in metric_groups:
                        g1_mean = round(
                            float(np.mean(metric_groups[comp.group1])),
                            4,
                        )
                    if comp.group2 in metric_groups:
                        g2_mean = round(
                            float(np.mean(metric_groups[comp.group2])),
                            4,
                        )

                rows.append({
                    "metric": metric,
                    "level": "pairwise",
                    "test_name": result.posthoc.test_name,
                    "group1": comp.group1,
                    "group2": comp.group2,
                    "statistic": round(comp.statistic, 4),
                    "p_value": comp.p_value_corrected,
                    "is_significant": comp.is_significant,
                    "group1_mean": g1_mean,
                    "group2_mean": g2_mean,
                })

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df["formatted_p_value"] = summary_df["p_value"].apply(
            lambda p: f"{p:.4f}" if p >= 0.001 else "< 0.001"
        )

    return summary_df


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
    run_clustered_permutation: bool = False,
    n_permutations: int = 10_000,
    permutation_seed: int | None = None,
) -> dict[str, GroupComparisonResult]:
    """Compare hierarchical assertion scores across RAG methods.

    Performs statistical significance tests for four metrics:
    - **global_pass_rate**: Per-question global assertion pass rate
    - **support_level**: Per-question average support level
    - **supporting_pass_rate**: Per-question supporting assertion
        pass rate
    - **discovery_rate**: Per-question discovery rate

    For each metric, runs omnibus tests (ANOVA/Friedman) and
    pairwise post-hoc tests with multiple comparison correction.

    Optionally runs a secondary clustered permutation test at the assertion
    level, which accounts for within-question correlation by permuting
    RAG method labels at the question (cluster) level. Results are stored
    with "_clustered" suffix keys.

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
        run_clustered_permutation: Whether to run assertion-level clustered
            permutation tests as secondary analysis (default False).
        n_permutations: Number of permutations for the clustered permutation
            test (default 10,000).
        permutation_seed: Random seed for reproducibility of permutation
            tests (default None).

    Returns:
        Dictionary mapping metric names to GroupComparisonResult objects
        containing omnibus test results and pairwise post-hoc comparisons.
        If run_clustered_permutation is True, additional keys with
        "_clustered" suffix contain assertion-level results.

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
        ...     run_clustered_permutation=True,
        ... )
        >>> print(results["global_pass_rate"].omnibus.summary())
        >>> print(results["global_pass_rate_clustered"].omnibus.summary())
    """
    if len(aggregated_scores) < 2:
        msg = "Need at least 2 RAG methods to compare"
        raise ValueError(msg)

    rag_methods = list(aggregated_scores.keys())
    results: dict[str, GroupComparisonResult] = {}

    # Metrics to compare with their column names and aggregation methods
    # Note: support_level is per-global-assertion, supporting_pass_rate is
    # aggregate
    # Metrics with "filter" only include rows matching the filter condition.
    metrics_config = {
        "global_pass_rate": {
            "column": "global_score",
            "agg_func": "mean",
            "description": "Global Assertion Pass Rate",
        },
        "support_level": {
            "column": "support_level",
            "agg_func": "mean",
            "description": "Support Level",
        },
        "supporting_pass_rate": {
            "column": "n_supporting_passed",
            "agg_func": "_supporting_pass_rate",
            "description": "Supporting Pass Rate",
        },
        "discovery_rate": {
            "column": "has_discovery",
            "agg_func": "mean",
            "description": "Discovery Rate",
        },
    }

    for metric_name, config in metrics_config.items():
        rich_print(
            f"\n[bold]Significance test for "
            f"{config['description']}[/bold]"
        )

        # Collect per-question metric values for each RAG method
        groups: dict[str, list[float]] = {}

        for rag_method in rag_methods:
            scores_df = aggregated_scores[rag_method]

            # Special handling for supporting_pass_rate
            if config["agg_func"] == "_supporting_pass_rate":
                # Compute per-question supporting pass rate.
                # Count all supporting assertion evaluations
                # (no deduplication) because the same supporting
                # assertion can have different results under
                # different global assertions.
                per_question_values: list[float] = []
                for _, group in scores_df.groupby(
                    "question"
                ):
                    total_supporting = 0
                    total_passed = 0
                    for sr in group["support_results"]:
                        if sr:
                            total_supporting += len(sr)
                            total_passed += sum(
                                1
                                for s in sr
                                if s["passed"]
                            )
                    if total_supporting > 0:
                        per_question_values.append(
                            total_passed / total_supporting
                        )

                if per_question_values:
                    groups[rag_method] = per_question_values
                    rich_print(
                        f"  {rag_method}: "
                        f"n={len(per_question_values)}, "
                        f"mean="
                        f"{np.mean(per_question_values):.3f}"
                        f", std="
                        f"{np.std(per_question_values):.3f}"
                    )
                continue

            # Validate required column exists
            if config["column"] not in scores_df.columns:
                rich_print(
                    f"  [yellow]Warning: {rag_method} "
                    f"missing column "
                    f"'{config['column']}', "
                    f"skipping[/yellow]"
                )
                continue

            # Aggregate per question
            per_question = (
                scores_df.groupby("question")[
                    config["column"]
                ].agg(config["agg_func"])
            )
            values = per_question.dropna().tolist()

            if len(values) > 0:
                groups[rag_method] = values
                rich_print(
                    f"  {rag_method}: n={len(values)}, "
                    f"mean={np.mean(values):.3f}, "
                    f"std={np.std(values):.3f}"
                )

        # Need at least 2 groups to compare
        if len(groups) < 2:
            rich_print(
                "  [yellow]Insufficient data for comparison "
                "(need at least 2 RAG methods with "
                "data)[/yellow]"
            )
            continue

        # Run statistical comparison (paired=True since same
        # questions across methods)
        comparison_result = compare_groups(
            groups,
            alpha=alpha,
            correction=correction_method,
            paired=True,
        )
        results[metric_name] = comparison_result

        # Print summary
        rich_print(f"  {comparison_result.omnibus.summary()}")
        if comparison_result.posthoc:
            rich_print(
                f"  {comparison_result.posthoc.summary()}"
            )

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

    # Secondary analysis: assertion-level clustered permutation tests
    if run_clustered_permutation:
        _run_clustered_permutation_analysis(
            aggregated_scores=aggregated_scores,
            rag_methods=rag_methods,
            results=results,
            alpha=alpha,
            correction_method=correction_method,
            n_permutations=n_permutations,
            permutation_seed=permutation_seed,
            output_dir=output_dir,
        )

    return results


def _run_clustered_permutation_analysis(
    aggregated_scores: dict[str, pd.DataFrame],
    rag_methods: list[str],
    results: dict[str, GroupComparisonResult],
    alpha: float,
    correction_method: str,
    n_permutations: int,
    permutation_seed: int | None,
    output_dir: Path | None,
) -> None:
    """Run assertion-level clustered permutation tests for all metrics.

    Adds results to the provided results dictionary with "_clustered"
    suffix keys. For global_pass_rate, support_level, and discovery_rate,
    each assertion row is an observation clustered by question. For
    supporting_pass_rate, individual supporting assertions are exploded
    as observations clustered by question.

    Args:
        aggregated_scores: RAG method name to aggregated DataFrame.
        rag_methods: List of RAG method names.
        results: Dictionary to add clustered permutation results to
            (mutated in place).
        alpha: Significance level.
        correction_method: P-value correction method.
        n_permutations: Number of permutations.
        permutation_seed: Random seed for reproducibility.
        output_dir: Optional directory to save results.
    """
    rich_print(
        "\n[bold]Running clustered permutation tests "
        "(assertion-level)...[/bold]"
    )

    # Metrics that map directly to assertion-level columns
    direct_metrics = {
        "global_pass_rate": {
            "column": "global_score",
            "description": "Global Pass Rate (clustered permutation)",
        },
        "support_level": {
            "column": "support_level",
            "description": "Support Level (clustered permutation)",
        },
        "discovery_rate": {
            "column": "has_discovery",
            "description": "Discovery Rate (clustered permutation)",
        },
    }

    for metric_name, config in direct_metrics.items():
        result_key = f"{metric_name}_clustered"
        rich_print(
            f"\n[bold]Clustered permutation: "
            f"{config['description']}[/bold]"
        )

        groups: dict[str, list[float]] = {}
        cluster_id_map: dict[str, list[str]] = {}

        for rag_method in rag_methods:
            scores_df = aggregated_scores[rag_method]
            col = config["column"]
            if col not in scores_df.columns:
                rich_print(
                    f"  [yellow]Warning: {rag_method} missing "
                    f"column '{col}', skipping[/yellow]"
                )
                continue

            valid = scores_df[["question", col]].dropna(
                subset=[col]
            )
            groups[rag_method] = valid[col].astype(float).tolist()
            cluster_id_map[rag_method] = (
                valid["question"].astype(str).tolist()
            )
            rich_print(
                f"  {rag_method}: n={len(groups[rag_method])}, "
                f"mean={np.mean(groups[rag_method]):.3f}"
            )

        if len(groups) < 2:
            rich_print(
                "  [yellow]Insufficient data for comparison[/yellow]"
            )
            continue

        comparison_result = run_clustered_permutation_test(
            groups=groups,
            cluster_ids=cluster_id_map,
            n_permutations=n_permutations,
            alpha=alpha,
            correction=correction_method,
            seed=permutation_seed,
        )
        results[result_key] = comparison_result

        rich_print(f"  {comparison_result.omnibus.summary()}")
        if comparison_result.posthoc:
            rich_print(f"  {comparison_result.posthoc.summary()}")

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _save_hierarchical_significance_results(
                comparison_result,
                result_key,
                config["description"],
                output_dir,
                groups,
            )

    # Special handling: supporting_pass_rate at supporting assertion
    # level. Explode support_results so each individual supporting
    # assertion is an observation, clustered by question.
    _run_supporting_pass_rate_clustered(
        aggregated_scores=aggregated_scores,
        rag_methods=rag_methods,
        results=results,
        result_key="supporting_pass_rate_clustered",
        description="Supporting Pass Rate (clustered permutation)",
        alpha=alpha,
        correction_method=correction_method,
        n_permutations=n_permutations,
        permutation_seed=permutation_seed,
        output_dir=output_dir,
    )


def _run_supporting_pass_rate_clustered(
    aggregated_scores: dict[str, pd.DataFrame],
    rag_methods: list[str],
    results: dict[str, GroupComparisonResult],
    result_key: str,
    description: str,
    alpha: float,
    correction_method: str,
    n_permutations: int,
    permutation_seed: int | None,
    output_dir: Path | None,
) -> None:
    """Run clustered permutation test for supporting pass rate.

    Explodes support_results so each individual supporting assertion
    becomes an observation, clustered by question.

    Args:
        aggregated_scores: RAG method name to aggregated DataFrame.
        rag_methods: List of RAG method names.
        results: Dictionary to add result to (mutated in place).
        result_key: Key name for storing the result.
        description: Human-readable metric description.
        alpha: Significance level.
        correction_method: P-value correction method.
        n_permutations: Number of permutations.
        permutation_seed: Random seed for reproducibility.
        output_dir: Optional directory to save results.
    """
    rich_print(
        f"\n[bold]Clustered permutation: {description}[/bold]"
    )
    supp_groups: dict[str, list[float]] = {}
    supp_clusters: dict[str, list[str]] = {}

    for rag_method in rag_methods:
        scores_df = aggregated_scores[rag_method]

        values: list[float] = []
        clusters_list: list[str] = []

        for _, row in scores_df.iterrows():
            question = str(row["question"])
            support_results = row.get("support_results")
            if not support_results:
                continue
            for sr in support_results:
                values.append(float(sr["passed"]))
                clusters_list.append(question)

        if values:
            supp_groups[rag_method] = values
            supp_clusters[rag_method] = clusters_list
            rich_print(
                f"  {rag_method}: n={len(values)}, "
                f"mean={np.mean(values):.3f}"
            )

    if len(supp_groups) >= 2:
        supp_result = run_clustered_permutation_test(
            groups=supp_groups,
            cluster_ids=supp_clusters,
            n_permutations=n_permutations,
            alpha=alpha,
            correction=correction_method,
            seed=permutation_seed,
        )
        results[result_key] = supp_result

        rich_print(f"  {supp_result.omnibus.summary()}")
        if supp_result.posthoc:
            rich_print(f"  {supp_result.posthoc.summary()}")

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            _save_hierarchical_significance_results(
                supp_result,
                result_key,
                description,
                output_dir,
                supp_groups,
            )
    else:
        rich_print(
            "  [yellow]Insufficient data for comparison[/yellow]"
        )


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
