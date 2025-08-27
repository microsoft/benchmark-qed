# Copyright (c) 2025 Microsoft Corporation.
"""Rationale relevance assessor implementation based on UMBRELA methodology with JSON response."""

import asyncio
import logging
from pathlib import Path
from string import Template

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentItem, RelevanceAssessmentResponse
from benchmark_qed.autoe.prompts import retrieval as retrieval_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel
from benchmark_qed.llm.utils import chat_typed_response
from .base import RelevanceRater

log = logging.getLogger(__name__)

RETRIEVAL_PROMPTS_PATH = Path(retrieval_prompts.__file__).parent


class RationaleRelevanceRater(RelevanceRater):
    """
    Rationale-based relevance assessor using UMBRELA methodology but with structured response.
    
    This assessor uses the same UMBRELA scoring scale (0-3) but prompts the LLM to return
    a structured response with both reasoning and score using Pydantic models.
    """

    def __init__(
        self,
        llm_client: ChatModel,
        llm_config: LLMConfig,
        prompt_template: Template | None = None,
        concurrent_requests: int = 32,
    ) -> None:
        """
        Initialize the RationaleRelevanceAssessor.

        Args:
            llm_client: The language model client to use for relevance assessment.
            llm_config: The LLM configuration containing call arguments and other settings.
            prompt_template: Rationale prompt template. If None, uses default from file.
            concurrent_requests: Maximum number of concurrent requests to the LLM.
        """
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.prompt_template = prompt_template or load_template_file(
            RETRIEVAL_PROMPTS_PATH / "rationale_relevance_assessment_prompt.txt"
        )
        self.semaphore = asyncio.Semaphore(concurrent_requests)

    async def rate_relevance(
        self, query: str, text_units: list[TextUnit]
    ) -> RelevanceAssessmentResponse:
        """
        Rate the relevance of text units to a query using UMBRELA methodology with structured response.

        Args:
            query: The query to assess relevance against.
            text_units: List of text units to assess.

        Returns:
            RelevanceAssessmentResponse containing assessment results with reasoning.
            Score is on UMBRELA's 0-3 scale:
            0 = passage has nothing to do with the query
            1 = passage seems related to the query but does not answer it
            2 = passage has some answer for the query, but may be unclear
            3 = passage is dedicated to the query and contains the exact answer
        """
        if not text_units:
            return RelevanceAssessmentResponse(assessment=[])

        log.info(f"Processing {len(text_units)} text units using Rationale methodology")

        # Process each text unit individually
        tasks = [
            self._assess_unit(query, unit, idx)
            for idx, unit in enumerate(text_units)
        ]

        results = await asyncio.gather(*tasks)

        log.info(f"Completed rationale relevance assessment for {len(results)} text units")
        return RelevanceAssessmentResponse(assessment=results)

    async def _assess_unit(
        self, query: str, unit: TextUnit, unit_idx: int
    ) -> RelevanceAssessmentItem:
        """
        Assess the relevance of a single text unit with detailed reasoning.

        Args:
            query: The query to assess relevance against.
            unit: The text unit to assess.
            unit_idx: Index of the unit for logging.

        Returns:
            RelevanceAssessmentItem containing assessment results with reasoning.
        """
        async with self.semaphore:
            log.debug(f"Processing unit {unit_idx + 1}: {unit.id}")

            try:
                # Create the prompt for this specific text unit
                user_message = self.prompt_template.substitute(
                    query=query, passage=unit.text
                )

                messages = [{"role": "system", "content": user_message}]

                # Use chat_typed_response for structured output
                llm_response = await chat_typed_response(
                    llm=self.llm_client,
                    messages=messages,
                    data_model=RelevanceAssessmentItem,
                    response_format={"type": "json_object"},
                    **self.llm_config.call_args,
                )

                # Create the final response with the text unit included
                return RelevanceAssessmentItem(
                    text_unit=unit,
                    reasoning=llm_response.reasoning,
                    score=llm_response.score,
                )

            except Exception as e:
                log.error(f"Error processing unit {unit.id}: {e}")
                return RelevanceAssessmentItem(
                    text_unit=unit,
                    reasoning=f"Assessment failed - LLM processing error: {e}",
                    score=-1,  # Use -1 to indicate error
                )
