# Copyright (c) 2025 Microsoft Corporation.
"""Bing relevance assessor implementation based on the UMBRELA paper."""

import asyncio
import logging
import re
from pathlib import Path
from string import Template

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentItem, RelevanceAssessmentResponse
from benchmark_qed.autoe.prompts import retrieval as retrieval_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel
from benchmark_qed.autoe.retrieval_scores.relevance_assessment.base import RelevanceRater

log = logging.getLogger(__name__)

RETRIEVAL_PROMPTS_PATH = Path(retrieval_prompts.__file__).parent


class BingRelevanceRater(RelevanceRater):
    """Relevance assessor using the DNA prompt from the UMBRELA paper."""

    def __init__(
        self,
        llm_client: ChatModel,
        llm_config: LLMConfig,
        prompt_template: Template | None = None,
        concurrent_requests: int = 32,
    ) -> None:
        """
        Initialize the BingRelevanceRater.

        Args:
            llm_client: The language model client to use for relevance assessment.
            llm_config: The LLM configuration containing call arguments and other settings.
            prompt_template: UMBRELA prompt template. If None, uses default from file.
            concurrent_requests: Maximum number of concurrent requests to the LLM.
        """
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.prompt_template = prompt_template or load_template_file(
            RETRIEVAL_PROMPTS_PATH / "bing_relevance_assessment_prompt.txt"
        )
        self.semaphore = asyncio.Semaphore(concurrent_requests)

    async def rate_relevance(
        self, query: str, text_units: list[TextUnit]
    ) -> RelevanceAssessmentResponse:
        """
        Rate the relevance of text units to a query using UMBRELA prompt.

        Args:
            query: The query to assess relevance against.
            text_units: List of text units to assess.

        Returns:
            RelevanceAssessmentResponse containing assessment results.
            Score is on UMBRELA's 0-3 scale:
            0 = passage has nothing to do with the query
            1 = passage seems related to the query but does not answer it
            2 = passage has some answer for the query, but may be unclear
            3 = passage is dedicated to the query and contains the exact answer
        """
        if not text_units:
            return RelevanceAssessmentResponse(assessment=[])

        log.info(f"Processing {len(text_units)} text units using Bing assessment prompt")
        tasks = [
            self._assess_unit(query, unit, idx)
            for idx, unit in enumerate(text_units)
        ]

        results = await asyncio.gather(*tasks)

        log.info(f"Completed Bing relevance assessment for {len(results)} text units")
        return RelevanceAssessmentResponse(assessment=results)

    async def _assess_unit(
        self, query: str, unit: TextUnit, unit_idx: int
    ) -> RelevanceAssessmentItem:
        """
        Assess the relevance of a single text unit.

        Args:
            query: The query to assess relevance against.
            unit: The text unit to assess.
            unit_idx: Index of the unit for logging.

        Returns:
            RelevanceAssessmentItem containing assessment results.
        """
        async with self.semaphore:
            log.debug(f"Processing unit {unit_idx + 1}: {unit.id}")

            # Create the prompt using UMBRELA template
            prompt_content = self.prompt_template.substitute(
                query=query,
                passage=unit.text
            )

            messages = [
                {"role": "system", "content": prompt_content}
            ]

            try:
                # Call the LLM directly
                response = await self.llm_client.chat(
                    messages=messages,
                    **self.llm_config.call_args,
                )

                # Extract the score from the response using regex
                score, reasoning = self._parse_response(response.output.content)

                return RelevanceAssessmentItem(
                    text_unit=unit,
                    reasoning=reasoning,
                    score=score,
                )

            except Exception as e:
                log.error(f"Error processing unit {unit.id}: {e}")
                return RelevanceAssessmentItem(
                    text_unit=unit,
                    reasoning=f"Assessment failed - LLM processing error: {e}",
                    score=-1,  # Use -1 to indicate error
                )

    def _parse_response(self, response_content: str) -> tuple[int, str]:
        """
        Parse the response to extract the score and reasoning.

        Args:
            response_content: The raw response content from the LLM.

        Returns:
            Tuple of (score, reasoning).
        """
        # Look for the final score pattern: ##final score: X
        score_pattern = r"##final score:\s*(\d+)"
        score_match = re.search(score_pattern, response_content, re.IGNORECASE)
        
        if score_match:
            try:
                score = int(score_match.group(1))
                # Ensure score is in valid range (0-3)
                if 0 <= score <= 3:
                    # Extract reasoning (everything before the final score)
                    score_start = score_match.start()
                    reasoning = response_content[:score_start].strip()
                    if not reasoning:
                        reasoning = f"Bing assessment resulted in score {score}"
                    return score, reasoning
                else:
                    log.warning(f"Score {score} is out of valid range (0-3)")
                    return -1, f"Invalid score range: {score}. Full response: {response_content}"
            except ValueError:
                log.warning(f"Could not parse score from: {score_match.group(1)}")
        
        # Fallback parsing - look for any number that could be a score
        numbers = re.findall(r'\b([0-3])\b', response_content)
        if numbers:
            # Take the last valid number found
            try:
                score = int(numbers[-1])
                return score, f"Extracted score from response: {response_content.strip()}"
            except ValueError:
                pass
        
        # If no valid score found, return error
        log.warning(f"Could not parse Bing score from response: {response_content}")
        return -1, f"Failed to parse score. Full response: {response_content}"
