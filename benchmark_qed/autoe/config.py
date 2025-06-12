# Copyright (c) 2025 Microsoft Corporation.
"""Scoring configuration models."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from benchmark_qed.autoe.prompts import pairwise as pairwise_prompts
from benchmark_qed.autoe.prompts import reference as reference_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.model.score import (
    Condition,
    Criteria,
    pairwise_scores_criteria,
    reference_scores_criteria,
)
from benchmark_qed.config.prompt_config import PromptConfig


class AutoEPromptConfig(BaseModel):
    """Configuration for prompts used in AutoE scoring."""

    user_prompt: PromptConfig = Field(
        ...,
        description="User prompt configuration for scoring.",
    )

    system_prompt: PromptConfig = Field(
        ...,
        description="System prompt configuration for scoring.",
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

    prompt_config: AutoEPromptConfig = Field(
        ...,
        description="Configuration for prompts used in scoring.",
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

    prompt_config: AutoEPromptConfig = Field(
        default=AutoEPromptConfig(
            user_prompt=PromptConfig(
                prompt=Path(pairwise_prompts.__file__).parent
                / "pairwise_user_prompt.txt",
            ),
            system_prompt=PromptConfig(
                prompt=Path(pairwise_prompts.__file__).parent
                / "pairwise_system_prompt.txt",
            ),
        ),
        description="Configuration for prompts used in pairwise scoring.",
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

    prompt_config: AutoEPromptConfig = Field(
        default=AutoEPromptConfig(
            user_prompt=PromptConfig(
                prompt=Path(reference_prompts.__file__).parent
                / "reference_user_prompt.txt",
            ),
            system_prompt=PromptConfig(
                prompt=Path(reference_prompts.__file__).parent
                / "reference_system_prompt.txt",
            ),
        ),
        description="Configuration for prompts used in reference scoring.",
    )
