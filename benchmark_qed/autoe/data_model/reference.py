# Copyright (c) 2025 Microsoft Corporation.
"""Data models for reference scoring."""

from pydantic import BaseModel, Field


class ReferenceLLMResponse(BaseModel):
    """Response from the LLM for reference scoring."""

    score: int = Field(description="Score.")
    reasoning: str = Field(description="The reasoning behind the score.")
