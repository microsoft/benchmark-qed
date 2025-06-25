# Copyright (c) 2025 Microsoft Corporation.
"""Data models for assertion scoring."""

from typing import NamedTuple

from pydantic import BaseModel, Field


class AssertionLLMResponse(BaseModel):
    """Response from the LLM for assertion scoring."""

    reasoning: str = Field(description="The reasoning behind the assertion score.")
    score: int = Field(description="The assertion score.")


class Assertion(NamedTuple):
    """Assertion data model."""

    question_id: str
    question_text: str
    answer_text: str
    assertion: str
