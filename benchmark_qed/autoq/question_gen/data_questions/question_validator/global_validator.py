# Copyright (c) 2025 Microsoft Corporation.
"""Question validation for global questions using batch validation with clustering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark_qed.autoq.prompts.data_questions import global_questions
from benchmark_qed.autoq.question_gen.data_questions.question_validator.base import (
    BatchQuestionValidator,
)
from benchmark_qed.config.defaults import LLM_PARAMS, RANDOM_SEED

if TYPE_CHECKING:
    from string import Template

    from benchmark_qed.autoq.data_model.question import Question
    from benchmark_qed.llm.type.base import ChatModel

PROMPTS_PATH = Path(global_questions.__file__).parent


class GlobalQuestionValidator(BatchQuestionValidator):
    """
    Validate global questions using batch validation with KMeans clustering.

    Uses clustering to group similar questions together, enabling duplicate
    detection within batches. Each batch is validated by an LLM that checks
    for quality issues and marks duplicates.

    Validation criteria:
    1. Naturalness - sounds like a human would ask it
    2. Answerability - can be answered by reading documents (no counting/stats)
    3. Clarity - clear and unambiguous
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        validation_prompt: Template | None = None,
        batch_size: int = 15,
        random_seed: int = RANDOM_SEED,
    ) -> None:
        """
        Initialize the GlobalQuestionValidator.

        Parameters
        ----------
        llm : ChatModel
            Language model for validation.
        llm_params : dict[str, Any]
            Parameters for the LLM.
        validation_prompt : Template | None
            Custom prompt for batch validation.
        batch_size : int
            Target number of questions per batch. Default 15.
        random_seed : int
            Random seed for KMeans clustering. Default from config.
        """
        super().__init__(
            llm=llm,
            llm_params=llm_params,
            validation_prompt=validation_prompt,
            batch_size=batch_size,
            random_seed=random_seed,
        )

    def _get_default_prompt_path(self) -> Path:
        """Return the path to the global validation prompt."""
        return PROMPTS_PATH / "batch_validation_prompt.txt"

    def _format_question_for_validation(
        self, idx: int, question: Question
    ) -> dict[str, Any]:
        """
        Format a global question for validation.

        Global questions have simpler formatting - just id and text.

        Parameters
        ----------
        idx : int
            Index of the question in the batch.
        question : Question
            The question to format.

        Returns
        -------
        dict[str, Any]
            Dictionary with id and text.
        """
        return {
            "id": idx,
            "text": question.text,
        }
