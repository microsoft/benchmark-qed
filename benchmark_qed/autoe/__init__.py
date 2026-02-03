# Copyright (c) 2025 Microsoft Corporation.
"""Relative measure module for evaluating the performance of models.

This module provides comprehensive evaluation capabilities including:
- Assertion-based scoring (standard and hierarchical)
- Reference-based scoring
- Pairwise comparison scoring
- Retrieval metrics evaluation
"""

# Assertion scoring
from benchmark_qed.autoe.assertion import (
    HierarchicalMode,
    aggregate_hierarchical_scores,
    compare_hierarchical_assertion_scores_significance,
    evaluate_assertion,
    get_assertion_scores,
    get_hierarchical_assertion_scores,
    summarize_hierarchical_by_question,
)

# Pairwise scoring
from benchmark_qed.autoe.pairwise import (
    SCORE_MAPPING,
    analyze_criteria,
    get_pairwise_score,
    get_pairwise_scores,
)

# Reference scoring
from benchmark_qed.autoe.reference import (
    get_reference_score,
    get_reference_scores,
    summarize_reference_scores,
)

# Retrieval scoring
from benchmark_qed.autoe.retrieval import (
    FidelityMetric,
    calculate_retrieval_metrics,
    load_clusters_from_json,
    load_reference_results,
    load_retrieval_results,
    run_retrieval_evaluation,
)

# Utilities
from benchmark_qed.autoe.utils import (
    GroupComparisonResult,
    NormalityResult,
    OmnibusTestResult,
    PairwiseComparison,
    PostHocResult,
    check_normality,
    compare_groups,
    run_omnibus_test,
    run_posthoc_pairwise,
)

# Visualization
from benchmark_qed.autoe.visualization import (
    get_available_question_sets,
    get_available_rag_methods,
    plot_assertion_accuracy_by_rag_method,
    plot_assertion_score_distribution,
    prepare_assertion_summary_data,
)

__all__ = [
    # Assertion scoring
    "HierarchicalMode",
    "aggregate_hierarchical_scores",
    "compare_hierarchical_assertion_scores_significance",
    "evaluate_assertion",
    "get_assertion_scores",
    "get_hierarchical_assertion_scores",
    "summarize_hierarchical_by_question",
    # Pairwise scoring
    "SCORE_MAPPING",
    "analyze_criteria",
    "get_pairwise_score",
    "get_pairwise_scores",
    # Reference scoring
    "get_reference_score",
    "get_reference_scores",
    "summarize_reference_scores",
    # Retrieval scoring
    "FidelityMetric",
    "calculate_retrieval_metrics",
    "load_clusters_from_json",
    "load_reference_results",
    "load_retrieval_results",
    "run_retrieval_evaluation",
    # Utilities
    "GroupComparisonResult",
    "NormalityResult",
    "OmnibusTestResult",
    "PairwiseComparison",
    "PostHocResult",
    "check_normality",
    "compare_groups",
    "run_omnibus_test",
    "run_posthoc_pairwise",
    # Visualization
    "get_available_question_sets",
    "get_available_rag_methods",
    "plot_assertion_accuracy_by_rag_method",
    "plot_assertion_score_distribution",
    "prepare_assertion_summary_data",
]
