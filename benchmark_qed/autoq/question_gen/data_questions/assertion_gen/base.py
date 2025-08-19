# Copyright (c) 2025 Microsoft Corporation.
"""Base classes for assertion generation."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from string import Template
from typing import Any

from benchmark_qed.autoq.data_model.question import Question
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.ranking import calculate_rrf_scores
from benchmark_qed.config.defaults import LLM_PARAMS
from benchmark_qed.llm.type.base import ChatModel

ClaimDict = dict[str, Any]  # Individual claim with statement, score, etc.

log = logging.getLogger(__name__)


@dataclass
class Assertion:
    """Data class representing an assertion for evaluation."""

    statement: str
    """The assertion statement text."""

    score: int
    """The importance/confidence score (1-10)."""

    sources: list[str] = field(default_factory=list)
    """List of source text chunks that are associated with the assertion."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Additional metadata and attributes."""

    def __post_init__(self) -> None:
        """Validate assertion fields after creation."""
        if not self.statement.strip():
            raise ValueError("Assertion statement cannot be empty")
        if not (1 <= self.score <= 10):
            raise ValueError("Assertion score must be between 1 and 10")


@dataclass
class AssertionGenerationResult:
    """Data class for assertion generation results."""

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
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        max_assertions: int | None = 10,
    ) -> None:
        self.llm = llm
        self.llm_params = llm_params
        self.json_mode = json_mode
        self.max_assertions = max_assertions
        
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
        pass

    async def agenerate_assertions_for_questions(
        self, questions: list[Question], **kwargs: Any
    ) -> None:
        """
        Convenience method to generate assertions for a list of questions and add them to question attributes.
        
        Args:
            questions: List of Question objects to generate assertions for (modified in place)
            **kwargs: Additional parameters passed to agenerate_assertions
            
        Side Effects:
            Updates each question's attributes with generated assertions:
            - 'assertions': List of generated assertion objects
            - 'assertion_count': Number of assertions generated
        """
        
        async def process_question(question: Question) -> None:
            try:
                result = await self.agenerate_assertions(question.text, **kwargs)
                
                # Initialize attributes if they don't exist
                if question.attributes is None:
                    question.attributes = {}
                
                # Add assertion results to question attributes
                question.attributes.update({
                    "assertions": result.assertions,
                    "assertion_count": result.total_assertions,
                })
                
            except Exception as e:
                log.warning(f"Failed to generate assertions for question '{question.id}': {e}")
                # Add empty assertion data on failure
                if question.attributes is None:
                    question.attributes = {}
                question.attributes.update({
                    "assertions": [],
                    "assertion_count": 0,
                })
        
        # Process all questions concurrently
        await asyncio.gather(
            *[process_question(q) for q in questions],
            return_exceptions=True
        )

    def _rank_and_limit_assertions(
        self, assertions: list[Assertion], max_assertions: int | None
    ) -> list[Assertion]:
        """
        Rank assertions using Reciprocal Rank Fusion (RRF) to combine importance score 
        and source count rankings, and optionally limit to max_assertions.
        
        RRF fuses rankings from two criteria:
        1. Importance score (descending: higher scores = better rank)
        2. Source count (descending: more sources = better rank)
        
        The RRF scores are stored in each assertion's attributes for debugging/analysis.
        
        Args:
            assertions: List of validated assertions
            max_assertions: Maximum number of assertions to return, or None for no limit
            
        Returns:
            Top ranked assertions using RRF fusion, optionally limited to max_assertions
        """
        if not assertions:
            return []
        
        # Calculate RRF scores using the utility function
        rrf_scores = calculate_rrf_scores(
            items=assertions,
            score_key_func=lambda a: a.score,
            source_count_key_func=lambda a: len(a.sources)
        )
        
        # Store RRF scores in assertion attributes for debugging/analysis
        for assertion in assertions:
            assertion.attributes['rrf_score'] = rrf_scores[id(assertion)]
        
        # Sort by RRF score (descending - higher RRF scores are better)
        ranked_assertions = sorted(
            assertions,
            key=lambda a: -rrf_scores[id(a)]
        )
        
        # Limit to max_assertions if specified
        if max_assertions is not None:
            return ranked_assertions[:max_assertions]
        return ranked_assertions
