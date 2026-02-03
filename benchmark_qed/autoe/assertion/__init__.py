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
    run_assertion_evaluation,
)
from benchmark_qed.autoe.assertion.significance import (
    compare_assertion_scores_significance,
    compare_hierarchical_assertion_scores_significance,
)
from benchmark_qed.autoe.assertion.standard import (
    evaluate_assertion,
    get_assertion_scores,
)

__all__ = [
    # Standard scoring
    "evaluate_assertion",
    "get_assertion_scores",
    # Hierarchical scoring
    "HierarchicalMode",
    "evaluate_hierarchical_assertion",
    "evaluate_supporting_discovery",
    "get_hierarchical_assertion_scores",
    # Aggregation
    "aggregate_hierarchical_scores",
    "summarize_hierarchical_by_question",
    # Significance testing
    "compare_assertion_scores_significance",
    "compare_hierarchical_assertion_scores_significance",
    # Pipeline
    "evaluate_rag_method",
    "load_and_normalize_assertions",
    "run_assertion_evaluation",
]
