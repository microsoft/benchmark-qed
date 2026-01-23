# Copyright (c) 2025 Microsoft Corporation.
"""Relative measure module for evaluating the performance of models."""

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
from benchmark_qed.autoe.visualization import (
    get_available_question_sets,
    get_available_rag_methods,
    plot_assertion_accuracy_by_rag_method,
    plot_assertion_score_distribution,
    prepare_assertion_summary_data,
)

__all__ = [
    "GroupComparisonResult",
    "NormalityResult",
    "OmnibusTestResult",
    "PairwiseComparison",
    "PostHocResult",
    "check_normality",
    "compare_groups",
    "get_available_question_sets",
    "get_available_rag_methods",
    "plot_assertion_accuracy_by_rag_method",
    "plot_assertion_score_distribution",
    "prepare_assertion_summary_data",
    "run_omnibus_test",
    "run_posthoc_pairwise",
]
