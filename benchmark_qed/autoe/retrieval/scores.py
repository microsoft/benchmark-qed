# Copyright (c) 2025 Microsoft Corporation.
"""Retrieval metrics evaluation: precision, recall, and fidelity.

This module provides comprehensive functions for evaluating retrieval-augmented
generation (RAG) methods, including:
- Loading and managing cluster data
- Reference result handling
- Relevance assessment
- Metric calculation (precision, recall, fidelity)
- Statistical significance testing
- Result persistence
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from rich import print as rich_print

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.clustering.base import (
    create_text_unit_to_cluster_mapping,
)
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
from benchmark_qed.autoe.data_model.retrieval_result import (
    RetrievalResult,
    load_retrieval_results_from_dicts,
)

# Re-export from retrieval_metrics subpackage (will be deprecated, use retrieval/)
from benchmark_qed.autoe.retrieval_metrics.reference_gen.cluster_relevance import (
    QueryClusterReferenceResult,
)
from benchmark_qed.autoe.retrieval_metrics.scoring.fidelity import (
    FidelityMetric,
    calculate_fidelity,
)
from benchmark_qed.autoe.retrieval_metrics.scoring.precision import (
    calculate_binary_precision,
    calculate_graded_precision,
)
from benchmark_qed.autoe.retrieval_metrics.scoring.recall import calculate_recall
from benchmark_qed.autoe.retrieval_metrics.scoring.retrieval_relevance import (
    BatchRelevanceResult,
    assess_batch_relevance,
)
from benchmark_qed.autoe.utils.stats import GroupComparisonResult, compare_groups
from benchmark_qed.cli.utils import print_df

if TYPE_CHECKING:
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.base import (
        RelevanceRater,
    )

log: logging.Logger = logging.getLogger(__name__)


def load_clusters_from_json(
    clusters_path: Path,
    text_units_path: Path | None = None,
) -> list[TextCluster]:
    """Load clusters from a JSON file.

    Supports two formats:
    1. text_units: array of objects with {id, short_id, text}
    2. text_unit_ids: array of string IDs (requires text_units_path to load text)

    Args:
        clusters_path: Path to clusters JSON file.
        text_units_path: Optional path to text units file (parquet/json/csv).
            Required if clusters use text_unit_ids format.

    Returns
    -------
        List of TextCluster objects with full TextUnit data.
    """
    with clusters_path.open(encoding="utf-8") as f:
        data = json.load(f)

    # Check if we need to load text units separately
    needs_text_units = any(
        "text_unit_ids" in cluster_data and "text_units" not in cluster_data
        for cluster_data in data
    )

    text_unit_map: dict[str, str] = {}
    if needs_text_units:
        if text_units_path is None:
            log.warning(
                "Clusters use text_unit_ids format but no text_units_path provided. "
                "Text content will be empty, which may break cluster mapping."
            )
        else:
            # Load text units and create ID -> text mapping
            suffix = text_units_path.suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(text_units_path)
            elif suffix == ".csv":
                df = pd.read_csv(text_units_path)
            elif suffix in (".json", ".jsonl"):
                df = pd.read_json(text_units_path, lines=(suffix == ".jsonl"))
            else:
                log.warning("Unknown text units file format: %s", suffix)
                df = None

            if df is not None:
                # Create mapping from ID to text
                id_col = "id" if "id" in df.columns else df.columns[0]
                text_col = (
                    "text"
                    if "text" in df.columns
                    else "chunk"
                    if "chunk" in df.columns
                    else df.columns[1]
                )
                text_unit_map = dict(
                    zip(df[id_col].astype(str), df[text_col].astype(str), strict=False)
                )
                log.info(
                    "Loaded %s text units from %s",
                    len(text_unit_map),
                    text_units_path,
                )

    clusters = []
    for cluster_data in data:
        # Handle both formats: text_units (full objects) or text_unit_ids (just IDs)
        if "text_units" in cluster_data:
            text_units = [
                TextUnit(
                    id=tu.get("id", ""),
                    short_id=tu.get("short_id", tu.get("id", "")),
                    text=tu.get("text", ""),
                )
                for tu in cluster_data["text_units"]
            ]
        elif "text_unit_ids" in cluster_data:
            # Create TextUnit objects from IDs, looking up text from map
            text_units = [
                TextUnit(
                    id=tu_id,
                    short_id=tu_id,
                    text=text_unit_map.get(tu_id, ""),
                )
                for tu_id in cluster_data["text_unit_ids"]
            ]
        else:
            text_units = []

        cluster = TextCluster(
            id=cluster_data["cluster_id"],
            text_units=text_units,
        )
        clusters.append(cluster)

    return clusters


def load_reference_results(
    reference_dir: Path,
    question_set: str,  # noqa: ARG001
    reference_filename: str = "reference.json",
) -> list[QueryClusterReferenceResult]:
    """Load reference cluster relevance results from JSON files.

    Args:
        reference_dir: Directory containing reference data.
        question_set: Name of the question set (unused when reference_filename is set).
        reference_filename: Filename for reference data within reference_dir.

    Returns
    -------
        List of QueryClusterReferenceResult objects.
    """
    reference_file = reference_dir / reference_filename
    if not reference_file.exists():
        msg = f"Reference file not found: {reference_file}"
        raise FileNotFoundError(msg)

    with reference_file.open(encoding="utf-8") as f:
        data = json.load(f)

    # Handle wrapped format from generate-retrieval-reference
    if isinstance(data, dict) and "references" in data:
        data = data["references"]

    return [QueryClusterReferenceResult.model_validate(item) for item in data]


def load_retrieval_results(
    retrieval_path: Path,
    context_id_key: str = "chunk_id",
    context_text_key: str = "text",
) -> list[RetrievalResult]:
    """Load retrieval results from a JSON file.

    Args:
        retrieval_path: Path to the retrieval results JSON file.
        context_id_key: Key name for chunk ID in retrieval results.
        context_text_key: Key name for chunk text in retrieval results.

    Returns
    -------
        List of RetrievalResult objects.
    """
    with retrieval_path.open(encoding="utf-8") as f:
        data = json.load(f)

    return load_retrieval_results_from_dicts(
        data,
        context_id_key=context_id_key,
        context_text_key=context_text_key,
        question_id_key="question_id",
        question_text_key="text",
        context_key="context",
    )


async def assess_rag_method_relevance(
    retrieval_results: list[RetrievalResult],
    relevance_rater: RelevanceRater,
    max_concurrent: int = 8,
) -> BatchRelevanceResult:
    """Assess relevance of retrieved chunks for a RAG method.

    Args:
        retrieval_results: List of retrieval results to assess.
        relevance_rater: RelevanceRater instance for assessing chunk relevance.
        max_concurrent: Maximum number of concurrent relevance assessments.

    Returns
    -------
        BatchRelevanceResult containing all relevance assessments.
    """
    return await assess_batch_relevance(
        retrieval_results=retrieval_results,
        relevance_rater=relevance_rater,
        max_concurrent=max_concurrent,
    )


def calculate_retrieval_metrics(
    batch_relevance: BatchRelevanceResult,
    reference_results: list[QueryClusterReferenceResult],
    clusters: list[TextCluster],
    relevance_threshold: int = 2,
    text_unit_to_cluster_mapping: dict[str, str] | None = None,
    fidelity_metric: FidelityMetric = FidelityMetric.JENSEN_SHANNON,
    cluster_match_by: str = "text",
) -> dict[str, Any]:
    """Calculate all retrieval metrics for a RAG method.

    Computes precision, recall, and fidelity metrics based on relevance assessments
    and reference cluster data.

    Args:
        batch_relevance: BatchRelevanceResult with relevance assessments.
        reference_results: Reference cluster relevance results.
        clusters: List of TextCluster objects.
        relevance_threshold: Minimum score to consider relevant.
        text_unit_to_cluster_mapping: Pre-computed mapping (optional).
        fidelity_metric: Fidelity metric to use (JS or TVD).
        cluster_match_by: How to match text units to clusters.

    Returns
    -------
        Dictionary with all metrics including precision, recall, fidelity, and summary.
    """
    # Create mapping if not provided
    if text_unit_to_cluster_mapping is None:
        text_unit_to_cluster_mapping = create_text_unit_to_cluster_mapping(
            clusters, match_by=cluster_match_by
        )

    # Calculate precision metrics
    binary_precision = calculate_binary_precision(
        batch_relevance, relevance_threshold=relevance_threshold
    )
    graded_precision = calculate_graded_precision(batch_relevance)

    # Calculate recall metrics
    recall_metrics = calculate_recall(
        query_relevance_results=batch_relevance.results,
        retrieval_references=reference_results,
        relevance_threshold=relevance_threshold,
        text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
        clusters=clusters,
        match_by=cluster_match_by,
    )

    # Calculate fidelity metrics (using selected metric)
    fidelity = calculate_fidelity(
        query_relevance_results=batch_relevance.results,
        retrieval_references=reference_results,
        relevance_threshold=relevance_threshold,
        text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
        clusters=clusters,
        metric=fidelity_metric,
        match_by=cluster_match_by,
    )

    return {
        "precision": {
            "binary": binary_precision,
            "graded": graded_precision,
        },
        "recall": recall_metrics,
        "fidelity": fidelity,
        "summary": {
            "binary_precision": binary_precision["macro_averaged_precision"],
            "graded_precision": graded_precision["macro_averaged_precision"],
            "recall": recall_metrics["macro_averaged_recall"],
            "fidelity": fidelity["macro_averaged_fidelity"],
            "total_queries": batch_relevance.total_queries,
        },
    }


def extract_per_query_metrics(
    batch_relevance: BatchRelevanceResult,
    reference_results: list[QueryClusterReferenceResult],
    clusters: list[TextCluster],
    relevance_threshold: int = 2,
    text_unit_to_cluster_mapping: dict[str, str] | None = None,
    fidelity_metric: FidelityMetric = FidelityMetric.JENSEN_SHANNON,
    cluster_match_by: str = "text",
) -> pd.DataFrame:
    """Extract per-query metrics for statistical analysis.

    Args:
        batch_relevance: BatchRelevanceResult with relevance assessments.
        reference_results: Reference cluster relevance results.
        clusters: List of TextCluster objects.
        relevance_threshold: Minimum score to consider relevant.
        text_unit_to_cluster_mapping: Pre-computed mapping (optional).
        fidelity_metric: Fidelity metric to use (JS or TVD).
        cluster_match_by: How to match text units to clusters.

    Returns
    -------
        DataFrame with one row per query and columns for each metric.
    """
    if text_unit_to_cluster_mapping is None:
        text_unit_to_cluster_mapping = create_text_unit_to_cluster_mapping(
            clusters, match_by=cluster_match_by
        )

    # Calculate all metrics to get per-query details
    recall_metrics = calculate_recall(
        query_relevance_results=batch_relevance.results,
        retrieval_references=reference_results,
        relevance_threshold=relevance_threshold,
        text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
        clusters=clusters,
        match_by=cluster_match_by,
    )

    fidelity = calculate_fidelity(
        query_relevance_results=batch_relevance.results,
        retrieval_references=reference_results,
        relevance_threshold=relevance_threshold,
        text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
        clusters=clusters,
        metric=fidelity_metric,
        match_by=cluster_match_by,
    )

    # Determine fidelity key based on metric
    fidelity_key = (
        "js_fidelity"
        if fidelity_metric == FidelityMetric.JENSEN_SHANNON
        else "tvd_fidelity"
    )

    # Build per-query DataFrame
    rows = []
    for result in batch_relevance.results:
        question_id = result.question_id

        # Calculate per-query precision
        relevant_count = result.get_relevant_count(relevance_threshold)
        precision = (
            relevant_count / result.total_chunks if result.total_chunks > 0 else 0.0
        )

        # Get recall and fidelity from query_details
        recall_detail = next(
            (
                d
                for d in recall_metrics.get("query_details", [])
                if d["question_id"] == question_id
            ),
            {"recall": 0.0},
        )
        fidelity_detail = next(
            (
                d
                for d in fidelity.get("query_details", [])
                if d["question_id"] == question_id
            ),
            {fidelity_key: 0.0},
        )

        rows.append({
            "question_id": question_id,
            "precision": precision,
            "recall": recall_detail.get("recall", 0.0),
            "fidelity": fidelity_detail.get(fidelity_key, 0.0),
            "total_chunks": result.total_chunks,
            "relevant_chunks": relevant_count,
        })

    return pd.DataFrame(rows)


def compare_retrieval_metrics_significance(
    rag_metrics: dict[str, pd.DataFrame],
    alpha: float = 0.05,
    correction_method: str = "holm",
) -> dict[str, GroupComparisonResult]:
    """Compare retrieval metrics across RAG methods using statistical tests.

    Performs significance testing to determine if differences in retrieval
    metrics between RAG methods are statistically significant.

    Args:
        rag_metrics: Dictionary mapping RAG method names to per-query DataFrames.
        alpha: Significance level.
        correction_method: P-value correction method.

    Returns
    -------
        Dictionary mapping metric names to GroupComparisonResult.
    """
    results: dict[str, GroupComparisonResult] = {}
    metrics_to_compare = ["precision", "recall", "fidelity"]

    for metric_name in metrics_to_compare:
        rich_print(f"\n[bold]Significance test for {metric_name}[/bold]")

        # Collect per-query values for each RAG method
        groups: dict[str, list[float]] = {}
        for rag_name, df in rag_metrics.items():
            if metric_name in df.columns:
                values = df[metric_name].dropna().tolist()
                if len(values) > 0:
                    groups[rag_name] = values

        if len(groups) < 2:
            rich_print(
                "  [yellow]Insufficient data for comparison "
                "(need at least 2 RAG methods)[/yellow]"
            )
            continue

        # Run statistical comparison (paired=True since same questions)
        comparison_result = compare_groups(
            groups, alpha=alpha, correction=correction_method, paired=True
        )
        results[metric_name] = comparison_result

        # Print summary
        rich_print(f"  {comparison_result.omnibus.summary()}")
        if comparison_result.posthoc:
            rich_print(f"  {comparison_result.posthoc.summary()}")

    return results


def save_retrieval_results(
    output_dir: Path,
    rag_name: str,
    question_set: str,
    metrics: dict[str, Any],
    per_query_df: pd.DataFrame,
    batch_relevance: BatchRelevanceResult,
) -> None:
    """Save retrieval evaluation results to files.

    Args:
        output_dir: Directory to save results.
        rag_name: Name of the RAG method.
        question_set: Name of the question set.
        metrics: Dictionary with all metrics.
        per_query_df: DataFrame with per-query metrics.
        batch_relevance: BatchRelevanceResult with relevance assessments.
    """
    question_set_dir = output_dir / question_set
    question_set_dir.mkdir(parents=True, exist_ok=True)

    # Save summary metrics
    summary = metrics["summary"]
    summary["rag_method"] = rag_name
    summary["question_set"] = question_set
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(question_set_dir / f"{rag_name}_summary.csv", index=False)

    # Save per-query metrics
    per_query_df["rag_method"] = rag_name
    per_query_df.to_csv(question_set_dir / f"{rag_name}_per_query.csv", index=False)

    # Save full relevance results
    batch_relevance.save_to_json(
        question_set_dir / f"{rag_name}_relevance_results.json"
    )


def save_significance_results(
    output_dir: Path,
    question_set: str,
    significance_results: dict[str, GroupComparisonResult],
    rag_metrics: dict[str, pd.DataFrame],
) -> None:
    """Save significance test results to files.

    Args:
        output_dir: Directory to save results.
        question_set: Name of the question set.
        significance_results: Dictionary mapping metric names to results.
        rag_metrics: Dictionary mapping RAG method names to DataFrames.
    """
    question_set_dir = output_dir / question_set
    question_set_dir.mkdir(parents=True, exist_ok=True)

    # Save group statistics per metric
    for metric_name, result in significance_results.items():
        # Collect group stats
        group_stats = []
        for rag_name, df in rag_metrics.items():
            if metric_name in df.columns:
                values = df[metric_name].dropna()
                group_stats.append({
                    "rag_method": rag_name,
                    "n_queries": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                })

        stats_df = pd.DataFrame(group_stats)
        stats_df.to_csv(
            question_set_dir / f"significance_{metric_name}_stats.csv", index=False
        )

        # Save omnibus result
        omnibus_df = pd.DataFrame([
            {
                "metric": metric_name,
                "test_name": result.omnibus.test_name,
                "statistic": result.omnibus.statistic,
                "p_value": result.omnibus.p_value,
                "is_significant": result.omnibus.is_significant,
                "alpha": result.omnibus.alpha,
                "is_normal": result.omnibus.is_normal,
            }
        ])
        omnibus_df.to_csv(
            question_set_dir / f"significance_{metric_name}_omnibus.csv", index=False
        )

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
            pairwise_df.to_csv(
                question_set_dir / f"significance_{metric_name}_pairwise.csv",
                index=False,
            )


async def run_retrieval_evaluation(
    relevance_rater: RelevanceRater,
    rag_methods: list[dict[str, Any]],
    question_sets: list[str],
    reference_dir: Path,
    clusters: list[TextCluster],
    output_dir: Path,
    relevance_threshold: int = 2,
    context_id_key: str = "chunk_id",
    context_text_key: str = "text",
    run_significance_test: bool = True,
    significance_alpha: float = 0.05,
    significance_correction: str = "holm",
    fidelity_metric: FidelityMetric = FidelityMetric.JENSEN_SHANNON,
    max_concurrent: int = 8,
    reference_filename: str = "reference.json",
    cluster_match_by: str = "text",
) -> pd.DataFrame:
    """Run retrieval evaluation for multiple RAG methods and question sets.

    Args:
        relevance_rater: RelevanceRater instance for assessing chunk relevance.
        rag_methods: List of dicts with 'name' and 'retrieval_results_path'.
        question_sets: List of question set names to evaluate.
        reference_dir: Directory containing reference cluster relevance data.
        clusters: List of TextCluster objects.
        output_dir: Directory to save results.
        relevance_threshold: Minimum score to consider relevant.
        context_id_key: Key name for chunk ID in retrieval results.
        context_text_key: Key name for chunk text in retrieval results.
        run_significance_test: Whether to run statistical significance tests.
        significance_alpha: Alpha level for significance tests.
        significance_correction: P-value correction method.
        fidelity_metric: Fidelity metric to use (JS or TVD).
        max_concurrent: Maximum concurrent relevance assessments.
        reference_filename: Filename for reference data.
        cluster_match_by: How to match text units to clusters.

    Returns
    -------
        DataFrame with overall summary of retrieval metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute text unit to cluster mapping
    text_unit_to_cluster_mapping = create_text_unit_to_cluster_mapping(
        clusters, match_by=cluster_match_by
    )

    overall_results = []

    for question_set in question_sets:
        rich_print(f"\n[bold]Processing question set: {question_set}[/bold]")

        # Load reference results for this question set
        try:
            reference_results = load_reference_results(
                reference_dir, question_set, reference_filename
            )
        except FileNotFoundError as e:
            rich_print(f"  [red]Error: {e}[/red]")
            continue

        # Collect per-query metrics for significance testing
        rag_per_query_metrics: dict[str, pd.DataFrame] = {}

        for rag_method in rag_methods:
            rag_name = rag_method["name"]
            retrieval_path = Path(rag_method["retrieval_results_path"])

            # Check if path includes question_set placeholder
            if "{question_set}" in str(retrieval_path):  # noqa: RUF027
                retrieval_path = Path(
                    str(retrieval_path).format(question_set=question_set)
                )

            rich_print(f"  Processing RAG method: {rag_name}")

            if not retrieval_path.exists():
                rich_print(
                    f"    [yellow]Warning: {retrieval_path} not found, skipping"
                    "[/yellow]"
                )
                continue

            # Load retrieval results
            try:
                retrieval_results = load_retrieval_results(
                    retrieval_path,
                    context_id_key=context_id_key,
                    context_text_key=context_text_key,
                )
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                rich_print(f"    [red]Error loading retrieval results: {e}[/red]")
                continue

            # Assess relevance
            rich_print(
                f"    Assessing relevance for {len(retrieval_results)} queries..."
            )
            batch_relevance = await assess_rag_method_relevance(
                retrieval_results, relevance_rater, max_concurrent
            )

            # Calculate metrics
            metrics = calculate_retrieval_metrics(
                batch_relevance=batch_relevance,
                reference_results=reference_results,
                clusters=clusters,
                relevance_threshold=relevance_threshold,
                text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
                fidelity_metric=fidelity_metric,
                cluster_match_by=cluster_match_by,
            )

            # Extract per-query metrics for significance testing
            per_query_df = extract_per_query_metrics(
                batch_relevance=batch_relevance,
                reference_results=reference_results,
                clusters=clusters,
                relevance_threshold=relevance_threshold,
                text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
                fidelity_metric=fidelity_metric,
                cluster_match_by=cluster_match_by,
            )
            rag_per_query_metrics[rag_name] = per_query_df

            # Save results
            save_retrieval_results(
                output_dir=output_dir,
                rag_name=rag_name,
                question_set=question_set,
                metrics=metrics,
                per_query_df=per_query_df,
                batch_relevance=batch_relevance,
            )

            # Print summary
            fidelity_label = (
                "JS Fidelity"
                if fidelity_metric == FidelityMetric.JENSEN_SHANNON
                else "TVD Fidelity"
            )
            summary = metrics["summary"]
            rich_print(f"    Binary Precision: {summary['binary_precision']:.3f}")
            rich_print(f"    Recall: {summary['recall']:.3f}")
            rich_print(f"    {fidelity_label}: {summary['fidelity']:.3f}")

            # Add to overall results
            overall_results.append({
                "question_set": question_set,
                "rag_method": rag_name,
                **summary,
            })

        # Run significance tests for this question set
        if run_significance_test and len(rag_per_query_metrics) >= 2:
            significance_results = compare_retrieval_metrics_significance(
                rag_metrics=rag_per_query_metrics,
                alpha=significance_alpha,
                correction_method=significance_correction,
            )
            save_significance_results(
                output_dir=output_dir,
                question_set=question_set,
                significance_results=significance_results,
                rag_metrics=rag_per_query_metrics,
            )

    # Create and save overall summary
    if overall_results:
        overall_df = pd.DataFrame(overall_results)
        overall_df = overall_df.sort_values(
            ["question_set", "recall"], ascending=[True, False]
        )
        overall_df.to_csv(output_dir / "retrieval_scores_summary.csv", index=False)

        print_df(overall_df, "Retrieval Metrics Summary")

        # Create pivot tables for each metric
        for metric in ["binary_precision", "recall", "fidelity"]:
            if metric in overall_df.columns:
                pivot = overall_df.pivot_table(
                    index="rag_method", columns="question_set", values=metric
                )
                pivot.to_csv(output_dir / f"retrieval_{metric}_pivot.csv")
                print_df(
                    pivot.reset_index(), f"{metric.replace('_', ' ').title()} by RAG"
                )

        return overall_df

    return pd.DataFrame()
