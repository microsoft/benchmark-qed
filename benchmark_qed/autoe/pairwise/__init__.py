# Copyright (c) 2025 Microsoft Corporation.
"""Pairwise evaluation module for autoe.

This module provides functions for pairwise comparison scoring using LLM-based
evaluation with configurable criteria.
"""

from benchmark_qed.autoe.pairwise.differential import (
    get_differential_pairwise_score,
    get_differential_pairwise_scores,
)
from benchmark_qed.autoe.pairwise.scores import (
    SCORE_MAPPING,
    analyze_criteria,
    get_pairwise_score,
    get_pairwise_scores,
)

__all__ = [
    "SCORE_MAPPING",
    "analyze_criteria",
    "get_differential_pairwise_score",
    "get_differential_pairwise_scores",
    "get_pairwise_score",
    "get_pairwise_scores",
]
