# Copyright (c) 2025 Microsoft Corporation.
"""Data models for pairwise scoring."""

from pydantic import BaseModel, Field


class PairwiseLLMResponse(BaseModel):
    """Response from the LLM for pairwise scoring."""

    winner: int = Field(description="The index of the winning answer.")
    reasoning: str = Field(description="The reasoning behind the score.")


class PairwiseExtractionLLMResponse(BaseModel):
    """Response from the LLM for the extraction step of differential pairwise scoring."""

    common: str = Field(description="Summary of the content shared by both answers.")
    unique_answer_1: str = Field(
        description="Content unique to Answer 1, or 'No unique part'."
    )
    unique_answer_2: str = Field(
        description="Content unique to Answer 2, or 'No unique part'."
    )


class CriterionVerdict(BaseModel):
    """Verdict for a single criterion in differential pairwise judging."""

    winner: int = Field(description="The index of the winning answer (1, 2, or 0).")
    reasoning: str = Field(description="The reasoning behind the verdict.")


class DifferentialCriterionVerdict(CriterionVerdict):
    """Named verdict for one criterion in differential pairwise judging.

    Extends ``CriterionVerdict`` with the criterion name so that an arbitrary set of
    criteria (the defaults or user-defined ones) can be judged in a single call
    without hardcoding the criteria in the response schema.
    """

    criteria: str = Field(description="Name of the criterion being judged.")


class DifferentialPairwiseLLMResponse(BaseModel):
    """Response from the LLM for the judging step of differential pairwise scoring.

    Contains one verdict per requested criterion, all judged in a single call based
    only on the unique content extracted from each answer. The criteria are not fixed:
    any set of criteria (default or user-defined) can be scored.
    """

    verdicts: list[DifferentialCriterionVerdict] = Field(
        description="One verdict per criterion, comparing the unique content of the two answers.",
    )
