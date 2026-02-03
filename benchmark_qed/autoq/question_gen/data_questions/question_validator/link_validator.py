# Copyright (c) 2025 Microsoft Corporation.
"""Question validation for link questions using batch validation with clustering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark_qed.autoq.prompts.data_questions import link_questions
from benchmark_qed.autoq.question_gen.data_questions.question_validator.base import (
    BatchQuestionValidator,
)
from benchmark_qed.config.defaults import LLM_PARAMS, RANDOM_SEED

if TYPE_CHECKING:
    from string import Template

    from benchmark_qed.autoq.data_model.question import Question
    from benchmark_qed.llm.type.base import ChatModel

PROMPTS_PATH = Path(link_questions.__file__).parent


class LinkQuestionValidator(BatchQuestionValidator):
    """
    Validate link questions using batch validation with KMeans clustering.

    Uses clustering to group similar questions together, enabling duplicate
    detection within batches. Each batch is validated by an LLM that checks
    for quality issues and marks duplicates.

    Link questions include bridge, comparison, intersection, and temporal types
    that require entity-specific validation criteria.
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
        Initialize the LinkQuestionValidator.

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
        """Return the path to the link validation prompt."""
        return PROMPTS_PATH / "batch_validation_prompt.txt"

    def _format_question_for_validation(
        self, idx: int, question: Question
    ) -> dict[str, Any]:
        """
        Format a link question for validation.

        Includes entity, question type, and draft answer for link-specific
        validation.

        Parameters
        ----------
        idx : int
            Index of the question in the batch.
        question : Question
            The question to format.

        Returns
        -------
        dict[str, Any]
            Dictionary with id, text, type, entity, and draft_answer.
        """
        return {
            "id": idx,
            "text": question.text,
            "type": (
                question.attributes.get("question_subtype", "unknown")
                if question.attributes
                else "unknown"
            ),
            "entity": (
                question.attributes.get("entity", "") if question.attributes else ""
            ),
            "draft_answer": (
                question.attributes.get("draft_answer", "")
                if question.attributes
                else ""
            ),
        }
