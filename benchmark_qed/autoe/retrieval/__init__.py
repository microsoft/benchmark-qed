# Copyright (c) 2025 Microsoft Corporation.
"""Retrieval evaluation module for autoe.

This module provides comprehensive retrieval metric evaluation capabilities,
including precision, recall, fidelity metrics, and statistical significance testing.

Main scoring functions are in scores.py. Sub-modules provide:
- reference_gen: Reference generation for cluster relevance
- relevance_assessment: Relevance rating implementations
- scoring: Individual metric calculations (precision, recall, fidelity)
"""

# Main scoring functions
from benchmark_qed.autoe.retrieval.scores import (
    assess_rag_method_relevance,
    calculate_retrieval_metrics,
    compare_retrieval_metrics_significance,
    extract_per_query_metrics,
    load_clusters_from_json,
    load_reference_results,
    load_retrieval_results,
    run_retrieval_evaluation,
    save_retrieval_results,
    save_significance_results,
)

# Re-export commonly used items from retrieval_metrics subpackages
from benchmark_qed.autoe.retrieval_metrics.reference_gen import (
    ClusterRelevanceRater,
    ClusterRelevanceResult,
    QueryClusterReferenceResult,
    get_relevant_clusters,
    get_relevant_units_per_cluster,
    load_cluster_references_from_json,
    save_cluster_references_to_json,
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
    QueryRelevanceResult,
    assess_batch_relevance,
)

__all__ = [
    "BatchRelevanceResult",
    "ClusterRelevanceRater",
    "ClusterRelevanceResult",
    "FidelityMetric",
    "QueryClusterReferenceResult",
    "QueryRelevanceResult",
    "assess_batch_relevance",
    "assess_rag_method_relevance",
    "calculate_binary_precision",
    "calculate_fidelity",
    "calculate_graded_precision",
    "calculate_recall",
    "calculate_retrieval_metrics",
    "compare_retrieval_metrics_significance",
    "extract_per_query_metrics",
    "get_relevant_clusters",
    "get_relevant_units_per_cluster",
    "load_cluster_references_from_json",
    "load_clusters_from_json",
    "load_reference_results",
    "load_retrieval_results",
    "run_retrieval_evaluation",
    "save_cluster_references_to_json",
    "save_retrieval_results",
    "save_significance_results",
]
