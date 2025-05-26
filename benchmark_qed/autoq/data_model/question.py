# Copyright (c) 2025 Microsoft Corporation.
from dataclasses import dataclass
from typing import Any

from benchmark_qed.autoq.data_model.enums import QuestionType


@dataclass
class Question:
    id: str
    """The unique identifier for the question."""

    text: str
    """The text of the question."""

    question_type: QuestionType | None = None
    """The type of the question (optional)."""

    references: list[str] | None = None
    """A list of references used to generate the question (optional)."""

    embedding: list[float] | None = None
    """The embedding of the question (optional)."""

    attributes: dict[str, Any] | None = None
    """A dictionary of additional attributes associated with the question (optional)."""
