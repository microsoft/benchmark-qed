# Copyright (c) 2025 Microsoft Corporation.
"""Scoring configuration models."""

from pydantic import BaseModel, Field

from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.model.score import (
    Condition,
    Criteria,
    pairwise_scores_criteria,
    reference_scores_criteria,
)


class PairwiseConfig(BaseModel):
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
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for scoring.",
    )


class ReferenceConfig(BaseModel):
    """Configuration for ground truth data."""

    ground_truth: Condition = Field(
        ..., description="Condition with the ground truth answers."
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
    score_max: int = Field(5, description="Maximum score for the criteria.")
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for scoring.",
    )
