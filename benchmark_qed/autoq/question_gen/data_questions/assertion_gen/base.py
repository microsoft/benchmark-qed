# Copyright (c) 2025 Microsoft Corporation.
"""Base classes for assertion generation."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator
from tqdm.asyncio import tqdm_asyncio

from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.ranking import (
    calculate_rrf_scores,
)
from benchmark_qed.config.defaults import LLM_PARAMS, MAX_ASSERTIONS

if TYPE_CHECKING:
    from benchmark_qed.autoq.data_model.question import Question
    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.validator import (
        AssertionValidator,
    )
    from benchmark_qed.llm.type.base import ChatModel

ClaimDict = dict[str, Any]  # Individual claim with statement, score, etc.

log: logging.Logger = logging.getLogger(__name__)


class Assertion(BaseModel):
    """Pydantic model representing an assertion for evaluation."""

    statement: str
    """The assertion statement text."""

    score: int = Field(ge=1, le=10)
    """The importance/confidence score (1-10)."""

    sources: list[str] = Field(default_factory=list)
    """List of source text chunks that are associated with the assertion."""

    reasoning: str = ""
    """Explanation of why this assertion is relevant to the question."""

    attributes: dict[str, Any] = Field(default_factory=dict)
    """Additional metadata and attributes."""

    @field_validator("statement")
    @classmethod
    def statement_not_empty(cls, v: str) -> str:
        """Validate that statement is not empty."""
        if not v.strip():
            msg = "Assertion statement cannot be empty"
            raise ValueError(msg)
        return v


class AssertionGenerationResult(BaseModel):
    """Pydantic model for assertion generation results."""

    assertions: list[Assertion]
    """The generated assertions."""

    total_assertions: int
    """Total number of assertions generated."""


class BaseAssertionGenerator(ABC):
    """
    Base class for generating factual assertions for evaluating answer accuracy in question-answering systems.

    This is a general interface that can be implemented by various types of assertion generators
    (claim-based, document-based, context-based, etc.).

    Subclasses should implement specific methods for generating assertions from different input types.

    Supports optional validation of generated assertions using AssertionValidator to filter out
    assertions that are not properly grounded in sources or not relevant to the question.
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        max_assertions: int | None = MAX_ASSERTIONS,
        validator: AssertionValidator | None = None,
        max_concurrent_questions: int | None = None,
    ) -> None:
        """
        Initialize the assertion generator.

        Parameters
        ----------
        llm : ChatModel
            The language model to use for generation.
        llm_params : dict[str, Any]
            Parameters for the LLM.
        json_mode : bool
            Whether to use JSON mode for LLM responses.
        max_assertions : int | None
            Maximum number of assertions to generate. None for no limit.
        validator : AssertionValidator | None
            Optional validator for filtering assertions. If provided, assertions
            will be validated after generation and only valid ones returned.
            If None, validation is skipped.
        max_concurrent_questions : int | None
            Maximum number of questions to process concurrently when generating
            assertions. If None, processes sequentially. If > 1, processes in parallel.
        """
        self.llm = llm
        self.llm_params = llm_params
        self.json_mode = json_mode
        self.max_assertions = max_assertions
        self.validator = validator
        self.max_concurrent_questions = max_concurrent_questions

        if self.json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

    @abstractmethod
    async def agenerate_assertions(
        self, question_text: str, **kwargs: Any
    ) -> AssertionGenerationResult:
        """Generate assertions for evaluating answer accuracy based on a question and additional inputs.

        Args:
            question_text: The question text to generate assertions for
            **kwargs: Additional parameters specific to assertion generator implementation
        """

    async def agenerate_assertions_for_questions(
        self,
        questions: list[Question],
    ) -> None:
        """
        Generate assertions for a list of questions and add them to question attributes.

        This method extracts claims from each question's attributes and generates
        assertions using the configured assertion generator. Results are stored
        as dictionaries in question.attributes["assertions"].

        Args:
            questions: List of Question objects to generate assertions for (modified in place).
                       Each question should have claims in question.attributes["claims"].

        Side Effects:
            Updates each question's attributes with generated assertions:
            - 'assertions': List of assertion dictionaries
            - 'assertion_count': Number of assertions generated
        """
        if self.max_concurrent_questions is None or self.max_concurrent_questions <= 1:
            # Sequential processing
            await self._generate_assertions_sequential(questions)
        else:
            # Parallel processing with semaphore
            await self._generate_assertions_parallel(questions)

    async def _generate_assertions_sequential(self, questions: list[Question]) -> None:
        """Generate assertions sequentially for each question."""
        from tqdm import tqdm

        for question in tqdm(questions, desc="Generating assertions", unit="question"):
            await self._process_single_question(question)

    async def _generate_assertions_parallel(self, questions: list[Question]) -> None:
        """Generate assertions in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self.max_concurrent_questions or 1)

        async def process_with_semaphore(question: Question) -> None:
            async with semaphore:
                await self._process_single_question(question)

        tasks = [process_with_semaphore(q) for q in questions]
        await tqdm_asyncio.gather(*tasks, desc="Generating assertions", unit="question")

    async def _process_single_question(self, question: Question) -> None:
        """Process a single question to generate assertions."""
        try:
            # Get claims from question attributes
            claims = (
                question.attributes.get("claims", []) if question.attributes else []
            )
            if not claims:
                log.warning("No claims found for question: %s", question.text)
                return

            result = await self.agenerate_assertions(
                question_text=question.text,
                claims=claims,
            )

            # Convert assertions to dicts for JSON serialization
            assertions = [assertion.model_dump() for assertion in result.assertions]

            # Initialize attributes if they don't exist
            if question.attributes is None:
                question.attributes = {}

            # Add assertion results to question attributes
            question.attributes["assertions"] = assertions
            question.attributes["assertion_count"] = len(assertions)

            log.info(
                "Generated %s assertions for question: %s",
                len(assertions),
                question.text,
            )

        except Exception as e:  # noqa: BLE001
            log.warning(
                "Failed to generate assertions for question '%s': %s",
                question.text,
                e,
            )
            # Ensure attributes exist even on failure
            if question.attributes is None:
                question.attributes = {}
            question.attributes["assertions"] = []
            question.attributes["assertion_count"] = 0

    async def _validate_assertions(
        self,
        assertions: list[Assertion],
        question_text: str,
    ) -> list[Assertion]:
        """
        Validate assertions using the configured validator.

        If no validator is configured, returns assertions unchanged.

        Args:
            assertions: List of assertions to validate
            question_text: The question text for context in validation

        Returns
        -------
            List of assertions that passed validation, or all assertions if no validator
        """
        if not self.validator or not assertions:
            return assertions

        log.info("Validating %s assertions...", len(assertions))
        summary = await self.validator.validate_assertions(assertions, question_text)

        log.info(
            "Validation complete: %s/%s assertions passed (%.1f%%)",
            summary.valid_count,
            summary.total_count,
            summary.validation_rate * 100,
        )

        return summary.valid_assertions

    def _rank_and_limit_assertions(
        self, assertions: list[Assertion], max_assertions: int | None
    ) -> list[Assertion]:
        """
        Rank assertions using RRF and optionally limit to max_assertions.

        Reciprocal Rank Fusion combines importance score and source count rankings.

        RRF fuses rankings from two criteria:
        1. Importance score (descending: higher scores = better rank)
        2. Source count (descending: more sources = better rank)

        The RRF scores are stored in each assertion's attributes for debugging/analysis.

        Args:
            assertions: List of validated assertions
            max_assertions: Maximum number of assertions to return, or None for no limit

        Returns
        -------
            Top ranked assertions using RRF fusion, optionally limited to max_assertions
        """
        if not assertions:
            return []

        # Calculate RRF scores using the utility function
        rrf_scores = calculate_rrf_scores(
            items=assertions,
            score_key_func=lambda a: a.score,
            source_count_key_func=lambda a: len(a.sources),
        )

        # Store RRF scores in assertion attributes for debugging/analysis
        for assertion in assertions:
            assertion.attributes["rrf_score"] = rrf_scores[id(assertion)]

        # Sort by RRF score (descending - higher RRF scores are better)
        ranked_assertions = sorted(assertions, key=lambda a: -rrf_scores[id(a)])

        # Limit to max_assertions if specified
        if max_assertions is not None:
            return ranked_assertions[:max_assertions]
        return ranked_assertions
