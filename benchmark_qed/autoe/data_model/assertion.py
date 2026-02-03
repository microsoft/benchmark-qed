# Copyright (c) 2025 Microsoft Corporation.
"""Data models for assertion scoring."""

from typing import NamedTuple

from pydantic import BaseModel, Field


class AssertionLLMResponse(BaseModel):
    """Response from the LLM for assertion scoring."""

    reasoning: str = Field(description="The reasoning behind the assertion score.")
    score: int = Field(description="The assertion score.")


class SupportingAssertionResult(BaseModel):
    """Result for a single supporting assertion evaluation."""

    id: str = Field(description="The ID of the supporting assertion (e.g., 'SA1').")
    passed: bool = Field(description="Whether the supporting assertion was satisfied.")
    reasoning: str = Field(
        description="Brief explanation of why the assertion passed or failed."
    )


class HierarchicalAssertionLLMResponse(BaseModel):
    """Response from LLM for hierarchical assertion scoring with supporting assertions.

    This model is used for global assertions that have supporting (local) assertions.
    It evaluates both the global assertion and all supporting assertions in a single
    LLM call, plus optionally detects if the answer contains information beyond
    what's covered by the supporting assertions (discovery).
    """

    reasoning: str = Field(
        description="Overall reasoning for the global assertion evaluation."
    )
    global_passed: bool = Field(
        description="Whether the answer satisfies the global assertion."
    )
    supporting_results: list[SupportingAssertionResult] = Field(
        description=(
            "List of results for each supporting assertion, including ID, "
            "pass/fail status, and brief reasoning."
        )
    )
    has_discovery: bool = Field(
        default=False,
        description=(
            "Whether the answer contains information that goes beyond "
            "what is covered by the supporting assertions."
        ),
    )
    discovery_reasoning: str = Field(
        default="",
        description=(
            "Explanation of what additional information was found in the answer "
            "that is not covered by the supporting assertions."
        ),
    )


class SupportingDiscoveryLLMResponse(BaseModel):
    """Response from LLM for supporting assertion + discovery evaluation only.

    This model is used in the two-call approach where the global assertion has
    already been evaluated separately. This call only evaluates supporting
    assertions and discovery detection.
    """

    supporting_results: list[SupportingAssertionResult] = Field(
        description=(
            "List of results for each supporting assertion, including ID, "
            "pass/fail status, and brief reasoning."
        )
    )
    has_discovery: bool = Field(
        default=False,
        description=(
            "Whether the answer contains information that goes beyond "
            "what is covered by the supporting assertions."
        ),
    )
    discovery_reasoning: str = Field(
        default="",
        description=(
            "Explanation of what additional information was found in the answer "
            "that is not covered by the supporting assertions."
        ),
    )


class Assertion(NamedTuple):
    """Assertion data model."""

    question_id: str
    question_text: str
    answer_text: str
    assertion: str


class HierarchicalAssertion(NamedTuple):
    """Hierarchical assertion data model with supporting assertions.

    Used for global assertions that have local/supporting assertions
    that were consolidated into them.
    """

    question_id: str
    question_text: str
    answer_text: str
    assertion: str
    supporting_assertions: list[str]
