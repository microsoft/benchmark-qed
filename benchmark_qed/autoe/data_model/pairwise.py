# Copyright (c) 2025 Microsoft Corporation.
"""Data models for pairwise scoring."""

from pydantic import BaseModel, Field


class PairwiseLLMResponse(BaseModel):
    """Response from the LLM for pairwise scoring."""

    winner: int = Field(description="The index of the winning answer.")
    reasoning: str = Field(description="The reasoning behind the score.")
