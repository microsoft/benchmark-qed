# Copyright (c) 2025 Microsoft Corporation.
"""Scoring configuration models."""

from typing import Self

from pydantic import BaseModel, Field, model_validator

from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.model.score import (
    Condition,
    Criteria,
    pairwise_scores_criteria,
    reference_scores_criteria,
)


class BaseAutoEConfig(BaseModel):
    """Base configuration for AutoE scoring."""

    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for scoring.",
    )

    trials: int = Field(
        default=4,
        description="Number of trials to run for each condition.",
    )

    @model_validator(mode="after")
    def check_trials_even(self) -> Self:
        """Check if the number of trials is even."""
        if self.trials % 2 != 0:
            msg = "The number of trials must be even to allow for counterbalancing of conditions."
            raise ValueError(msg)
        return self


class PairwiseConfig(BaseAutoEConfig):
    """Configuration for scoring a set of conditions."""

    base: Condition | None = Field(default=None, description="Base Conditions.")

    others: list[Condition] = Field(
        default_factory=list,
        description="Other Conditions to compare against the base.",
    )

    question_sets: list[str] = Field(
        default_factory=list,
        description="List of question sets to use for scoring.",
    )

    criteria: list[Criteria] = Field(
        default_factory=pairwise_scores_criteria,
        description="List of criteria to use for scoring.",
    )


class ReferenceConfig(BaseAutoEConfig):
    """Configuration for scoring based on reference answers."""

    reference: Condition = Field(
        ..., description="Condition with the reference answers."
    )
    generated: list[Condition] = Field(
        default_factory=list,
        description="Conditions with the generated answers to score.",
    )
    criteria: list[Criteria] = Field(
        default_factory=reference_scores_criteria,
        description="List of criteria to use for scoring.",
    )
    score_min: int = Field(1, description="Minimum score for the criteria.")
    score_max: int = Field(10, description="Maximum score for the criteria.")
