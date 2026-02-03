# Copyright (c) 2025 Microsoft Corporation.
"""Question validation module for quality control."""

from benchmark_qed.autoq.question_gen.data_questions.question_validator.base import (
    BatchQuestionValidator,
)
from benchmark_qed.autoq.question_gen.data_questions.question_validator.global_validator import (
    GlobalQuestionValidator,
)
from benchmark_qed.autoq.question_gen.data_questions.question_validator.link_validator import (
    LinkQuestionValidator,
)

# Backward compatibility alias
QuestionValidator = GlobalQuestionValidator

__all__ = [
    "BatchQuestionValidator",
    "GlobalQuestionValidator",
    "LinkQuestionValidator",
    "QuestionValidator",
]
