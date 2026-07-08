# Copyright (c) 2025 Microsoft Corporation.
"""Pairwise evaluation module for autoe.

This module provides functions for pairwise comparison scoring using LLM-based
evaluation with configurable criteria.
"""

from benchmark_qed.autoe.pairwise.scores import (
    SCORE_MAPPING,
    analyze_criteria,
    get_pairwise_score,
    get_pairwise_scores,
)
from benchmark_qed.autoe.pairwise.unbiased import (
    UNBIASED_CRITERIA,
    get_unbiased_pairwise_score,
    get_unbiased_pairwise_scores,
)

__all__ = [
    "SCORE_MAPPING",
    "UNBIASED_CRITERIA",
    "analyze_criteria",
    "get_pairwise_score",
    "get_pairwise_scores",
    "get_unbiased_pairwise_score",
    "get_unbiased_pairwise_scores",
]
