# Copyright (c) 2025 Microsoft Corporation.
"""Assertion scoring module for evaluating RAG system responses.

This module provides functions for:
- Standard assertion scoring
- Hierarchical assertion scoring (with supporting assertions)
- Score aggregation and summarization
- Statistical significance testing
- Evaluation pipelines and orchestration
"""

from benchmark_qed.autoe.assertion.aggregation import (
    aggregate_hierarchical_scores,
    summarize_hierarchical_by_question,
)
from benchmark_qed.autoe.assertion.hierarchical import (
    HierarchicalMode,
    evaluate_hierarchical_assertion,
    evaluate_supporting_discovery,
    get_hierarchical_assertion_scores,
)
from benchmark_qed.autoe.assertion.pipeline import (
    evaluate_rag_method,
    load_and_normalize_assertions,
    load_and_normalize_hierarchical_assertions,
    run_assertion_evaluation,
    run_hierarchical_assertion_evaluation,
)
from benchmark_qed.autoe.assertion.significance import (
    compare_assertion_scores_significance,
    compare_hierarchical_assertion_scores_significance,
    summarize_significance_results,
)
from benchmark_qed.autoe.assertion.standard import (
    evaluate_assertion,
    get_assertion_scores,
)

__all__ = [
    # Hierarchical scoring
    "HierarchicalMode",
    # Aggregation
    "aggregate_hierarchical_scores",
    # Significance testing
    "compare_assertion_scores_significance",
    "compare_hierarchical_assertion_scores_significance",
    # Standard scoring
    "evaluate_assertion",
    "evaluate_hierarchical_assertion",
    # Pipeline
    "evaluate_rag_method",
    "evaluate_supporting_discovery",
    "get_assertion_scores",
    "get_hierarchical_assertion_scores",
    "load_and_normalize_assertions",
    "load_and_normalize_hierarchical_assertions",
    "run_assertion_evaluation",
    "run_hierarchical_assertion_evaluation",
    "summarize_hierarchical_by_question",
    "summarize_significance_results",
]
