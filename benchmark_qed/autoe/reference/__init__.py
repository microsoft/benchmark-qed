# Copyright (c) 2025 Microsoft Corporation.
"""Reference scoring module for comparing generated answers to reference answers.

This module provides LLM-based evaluation of generated answers against
ground truth reference answers using configurable criteria.
"""

from benchmark_qed.autoe.reference.scores import (
    get_reference_score,
    get_reference_scores,
    summarize_reference_scores,
)

__all__ = [
    "get_reference_score",
    "get_reference_scores",
    "summarize_reference_scores",
]
