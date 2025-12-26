# Copyright (c) 2025 Microsoft Corporation.
"""Assertion validation for quality control and hallucination prevention."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autoq.prompts import data_questions
from benchmark_qed.config.defaults import LLM_PARAMS, MIN_ASSERTION_VALIDATION_SCORE
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from string import Template

    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.base import (
        Assertion,
    )
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

VALIDATOR_PROMPTS_PATH = Path(data_questions.__file__).parent


@dataclass
class ValidationScores:
    """Scores from LLM validation."""

    grounding: int = 3
    """Grounding score (1-5): how well supported by sources."""

    relevance: int = 3
    """Relevance score (1-5): how useful for evaluating the question."""

    verifiability: int = 3
    """Verifiability score (1-5): how clear and checkable."""

    reasoning: str = ""
    """Explanation of the validation result."""

    def is_valid(self, min_score: int = 3) -> bool:
        """Check if all scores meet the minimum threshold."""
        return all(
            score >= min_score
            for score in [self.grounding, self.relevance, self.verifiability]
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert validation scores to a dictionary for serialization."""
        return {
            "grounding": self.grounding,
            "relevance": self.relevance,
            "verifiability": self.verifiability,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_llm_response(cls, parsed: dict[str, Any]) -> ValidationScores:
        """Create ValidationScores from parsed LLM response."""
        return cls(
            grounding=cls._clamp(int(parsed.get("grounding", 3))),
            relevance=cls._clamp(int(parsed.get("relevance", 3))),
            verifiability=cls._clamp(int(parsed.get("verifiability", 3))),
            reasoning=parsed.get("reasoning", "No reasoning provided"),
        )

    @staticmethod
    def _clamp(value: int, min_val: int = 1, max_val: int = 5) -> int:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))


@dataclass
class ValidationResult:
    """Result of assertion validation."""

    assertion: Assertion
    """The original assertion."""

    is_valid: bool
    """Whether the assertion passed all validation checks."""

    scores: ValidationScores
    """Validation scores from LLM."""

    error: str | None = None
    """Error message if validation failed."""


@dataclass
class ValidationSummary:
    """Summary of validation results for a set of assertions."""

    valid_assertions: list[Assertion] = field(default_factory=list)
    """Assertions that passed all validation checks."""

    invalid_assertions: list[ValidationResult] = field(default_factory=list)
    """Assertions that failed validation with details."""

    @property
    def total_count(self) -> int:
        """Total number of assertions validated."""
        return len(self.valid_assertions) + len(self.invalid_assertions)

    @property
    def valid_count(self) -> int:
        """Number of valid assertions."""
        return len(self.valid_assertions)

    @property
    def validation_rate(self) -> float:
        """Percentage of assertions that passed validation."""
        return self.valid_count / self.total_count if self.total_count > 0 else 0.0


class AssertionValidator:
    """
    Validate assertions for quality control and hallucination prevention.

    Uses LLM to validate assertions against three criteria:
    1. Grounding - verifies assertions are supported by their source texts
    2. Relevance - checks if assertions are useful for evaluating the question
    3. Verifiability - ensures assertions are clear and testable

    Use this validator after assertion generation to filter out potentially
    hallucinated or low-quality assertions before using them for evaluation.
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        min_criterion_score: int = MIN_ASSERTION_VALIDATION_SCORE,
        validation_prompt: Template | None = None,
        concurrent_validations: int = 8,
    ) -> None:
        """
        Initialize the AssertionValidator.

        Parameters
        ----------
        llm : ChatModel
            Language model for validation.
        llm_params : dict[str, Any]
            Parameters for the LLM.
        min_criterion_score : int
            Minimum score (1-5) for grounding, relevance, verifiability. Default 3.
        validation_prompt : Template | None
            Custom prompt for validation.
        concurrent_validations : int
            Number of concurrent validations. Default 8.
        """
        self.llm = llm
        self.llm_params: dict[str, Any] = llm_params.copy()
        self.min_criterion_score = min_criterion_score
        self.concurrent_validations = concurrent_validations
        self._semaphore = asyncio.Semaphore(concurrent_validations)

        # Load validation prompt
        if validation_prompt:
            self.validation_prompt: Template = validation_prompt
        else:
            # Default to local validation prompt for backwards compatibility
            prompt_path = (
                VALIDATOR_PROMPTS_PATH / "assertions" / "local_validation_prompt.txt"
            )
            self.validation_prompt = load_template_file(prompt_path)

    async def validate_assertion(
        self,
        assertion: Assertion,
        question_text: str,
    ) -> ValidationResult:
        """
        Validate a single assertion against the question and its sources.

        Parameters
        ----------
        assertion : Assertion
            The assertion to validate.
        question_text : str
            The question this assertion is meant to evaluate.

        Returns
        -------
        ValidationResult
            Validation result with scores and validity.
        """
        # Use all sources for validation
        sources_text = self._format_sources(assertion.sources)
        if not sources_text:
            return ValidationResult(
                assertion=assertion,
                is_valid=False,
                scores=ValidationScores(
                    grounding=1, reasoning="No valid sources to validate against"
                ),
                error="No valid sources",
            )

        # Get LLM validation
        async with self._semaphore:
            try:
                scores = await self._get_llm_validation(
                    assertion.statement, question_text, sources_text
                )
                is_valid = scores.is_valid(self.min_criterion_score)
                return ValidationResult(
                    assertion=assertion, is_valid=is_valid, scores=scores
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Validation failed for assertion: %s", e)
                return ValidationResult(
                    assertion=assertion,
                    is_valid=True,  # Default to valid on error to avoid losing assertions
                    scores=ValidationScores(reasoning=f"Validation error: {e}"),
                    error=str(e),
                )

    async def _get_llm_validation(
        self, statement: str, question_text: str, sources_text: str
    ) -> ValidationScores:
        """Get validation scores from LLM."""
        prompt = self.validation_prompt.substitute(
            question=question_text,
            assertion=statement,
            sources=sources_text,
        )
        messages = [{"role": "user", "content": prompt}]
        result = await self.llm.chat(messages=messages, **self.llm_params)
        _, parsed = try_parse_json_object(result.output.content)

        if not parsed:
            log.warning(
                "Failed to parse validation response: %s", result.output.content[:200]
            )
            return ValidationScores(reasoning="Could not parse validation response")

        return ValidationScores.from_llm_response(parsed)

    @staticmethod
    def _format_sources(sources: list[str]) -> str:
        """Format and deduplicate sources for the validation prompt."""
        # Filter valid sources and deduplicate while preserving order
        valid_sources = []
        seen = set()
        for s in sources:
            if s and str(s).strip():
                text = str(s).strip()
                if text not in seen:
                    seen.add(text)
                    valid_sources.append(text)

        if not valid_sources:
            return ""
        return "\n\n".join(
            f"Source {i + 1}: {source}" for i, source in enumerate(valid_sources)
        )

    async def validate_assertions(
        self,
        assertions: list[Assertion],
        question_text: str,
    ) -> ValidationSummary:
        """
        Validate a list of assertions and return summary.

        Parameters
        ----------
        assertions : list[Assertion]
            List of assertions to validate.
        question_text : str
            The question these assertions are meant to evaluate.

        Returns
        -------
        ValidationSummary
            Summary with valid/invalid assertions and statistics.
        """
        if not assertions:
            return ValidationSummary()

        log.info(
            "Validating %s assertions for question: %s...",
            len(assertions),
            question_text[:50],
        )

        # Run validations concurrently
        results = await asyncio.gather(
            *[self.validate_assertion(a, question_text) for a in assertions],
            return_exceptions=True,
        )

        summary = ValidationSummary()
        for result in results:
            if isinstance(result, Exception):
                log.error("Validation error: %s", result)
                continue
            if isinstance(result, ValidationResult):
                # Store validation scores in assertion attributes
                result.assertion.attributes["validation"] = {
                    "is_valid": result.is_valid,
                    "scores": result.scores.to_dict(),
                }
                if result.error:
                    result.assertion.attributes["validation"]["error"] = result.error

                if result.is_valid:
                    summary.valid_assertions.append(result.assertion)
                else:
                    summary.invalid_assertions.append(result)

        log.info(
            "Validation complete: %s/%s assertions passed (%.1f%%)",
            summary.valid_count,
            summary.total_count,
            summary.validation_rate * 100,
        )
        return summary
