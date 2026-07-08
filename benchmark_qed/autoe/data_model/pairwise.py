# Copyright (c) 2025 Microsoft Corporation.
"""Data models for pairwise scoring."""

from pydantic import BaseModel, Field


class PairwiseLLMResponse(BaseModel):
    """Response from the LLM for pairwise scoring."""

    winner: int = Field(description="The index of the winning answer.")
    reasoning: str = Field(description="The reasoning behind the score.")


class PairwiseExtractionLLMResponse(BaseModel):
    """Response from the LLM for the extraction step of unbiased pairwise scoring."""

    common: str = Field(description="Summary of the content shared by both answers.")
    unique_answer_1: str = Field(
        description="Content unique to Answer 1, or 'No unique part'."
    )
    unique_answer_2: str = Field(
        description="Content unique to Answer 2, or 'No unique part'."
    )


class CriterionVerdict(BaseModel):
    """Verdict for a single criterion in unbiased pairwise judging."""

    winner: int = Field(description="The index of the winning answer (1, 2, or 0).")
    reasoning: str = Field(description="The reasoning behind the verdict.")


class UnbiasedPairwiseLLMResponse(BaseModel):
    """Response from the LLM for the judging step of unbiased pairwise scoring.

    Both relevance and diversity are judged in a single call, based only on the
    unique content extracted from each answer.
    """

    relevance: CriterionVerdict = Field(
        description="Verdict comparing the unique content on relevance."
    )
    diversity: CriterionVerdict = Field(
        description="Verdict comparing the unique content on diversity."
    )
