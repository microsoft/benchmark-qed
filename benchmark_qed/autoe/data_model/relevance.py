# Copyright (c) 2025 Microsoft Corporation.
"""Data models for retrieval relevance assessment."""

from pydantic import BaseModel, Field

from benchmark_qed.autod.data_model.text_unit import TextUnit


class RelevanceAssessmentItem(BaseModel):
    """Individual relevance assessment for a text chunk."""

    text_unit: TextUnit | None = Field(default=None, description="The text unit being assessed (optional).")
    reasoning: str | None = Field(default=None, description="The reasoning behind the relevance assessment (optional).")
    score: int = Field(description="The relevance score.")


class RelevanceAssessmentResponse(BaseModel):
    """Response from the LLM for relevance assessment."""

    assessment: list[RelevanceAssessmentItem] = Field(
        description="List of relevance assessments for each text chunk."
    )

