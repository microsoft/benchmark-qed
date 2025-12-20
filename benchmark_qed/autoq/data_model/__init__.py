# Copyright (c) 2025 Microsoft Corporation.
"""Define data models used in AutoQ."""

from benchmark_qed.autoq.data_model.enums import QuestionType
from benchmark_qed.autoq.data_model.question import Question

__all__ = [
    "Question",
    "QuestionType",
]
