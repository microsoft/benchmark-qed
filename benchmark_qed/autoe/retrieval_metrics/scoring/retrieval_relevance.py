# Copyright (c) 2025 Microsoft Corporation.
"""Module for running relevance assessments on a given query's retrieval context."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_serializer

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentResponse
from benchmark_qed.autoe.data_model.retrieval_result import RetrievalResult
from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.base import (
    RelevanceRater,
)

log = logging.getLogger(__name__)


class QueryRelevanceResult(BaseModel):
    """Result of relevance assessment for a single query."""

    model_config = {"frozen": True}

    question_id: str = Field(description="Unique identifier for the question.")
    question_text: str = Field(description="The text of the question.")
    assessments: RelevanceAssessmentResponse = Field(
        description="The relevance assessments for all text units."
    )
    total_chunks: int = Field(description="Total number of chunks assessed.")

    def get_relevant_chunks(self, relevance_threshold: int = 2) -> list[dict[str, Any]]:
        """
        Get chunks that meet or exceed the relevance threshold.

        Args:
            relevance_threshold: Minimum score to consider a chunk relevant (default: 2).

        Returns
        -------
            List of dictionaries containing relevant chunk information.
        """
        return [
            {
                "text_unit": item.text_unit,
                "reasoning": item.reasoning,
                "score": item.score,
            }
            for item in self.assessments.assessment
            if item.score >= relevance_threshold
        ]

    def get_relevant_count(self, relevance_threshold: int = 2) -> int:
        """
        Get the count of chunks that meet or exceed the relevance threshold.

        Args:
            relevance_threshold: Minimum score to consider a chunk relevant (default: 2).

        Returns
        -------
            Number of relevant chunks.
        """
        return len([
            item
            for item in self.assessments.assessment
            if item.score >= relevance_threshold
        ])

    @field_serializer("assessments")
    def serialize_assessments(
        self, response: RelevanceAssessmentResponse
    ) -> dict[str, Any]:
        """Serialize assessments excluding embeddings."""
        return {
            "assessment": [
                {
                    "text_unit": {
                        "id": item.text_unit.id,
                        "short_id": item.text_unit.short_id,
                        "text": item.text_unit.text,
                    },
                    "reasoning": item.reasoning,
                    "score": item.score,
                }
                for item in response.assessment
                if item.text_unit is not None
            ]
        }


class BatchRelevanceResult(BaseModel):
    """Result of batch relevance assessment."""

    model_config = {"frozen": True}

    results: list[QueryRelevanceResult] = Field(description="Individual query results.")

    @property
    def total_queries(self) -> int:
        """Get total number of queries processed."""
        return len(self.results)

    @property
    def mean_chunks_processed(self) -> int:
        """Get mean number of chunks processed across all queries."""
        if not self.results:
            return 0
        return sum(result.total_chunks for result in self.results) // len(self.results)

    def save_to_json(self, filepath: str | Path) -> None:
        """
        Save BatchRelevanceResult to a JSON file.

        Args:
            filepath: Path to the JSON file to save to.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

        log.info("Saved batch relevance results to %s", filepath)

    @classmethod
    def load_from_json(cls, filepath: str | Path) -> "BatchRelevanceResult":
        """
        Load BatchRelevanceResult from a JSON file.

        Args:
            filepath: Path to the JSON file to load from.

        Returns
        -------
            BatchRelevanceResult loaded from the file.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            msg = f"File not found: {filepath}"
            raise FileNotFoundError(msg)

        with filepath.open(encoding="utf-8") as f:
            data = json.load(f)

        result = cls.model_validate(data)
        log.info(
            "Loaded batch relevance results from %s: %d queries",
            filepath,
            result.total_queries,
        )
        return result


async def assess_query_relevance(
    retrieval_result: RetrievalResult,
    relevance_rater: RelevanceRater,
) -> QueryRelevanceResult:
    """
    Run relevance assessment for a single query.

    Args:
        retrieval_result: RetrievalResult containing query and context data.
        relevance_rater: The relevance rater instance to use for assessment.

    Returns
    -------
        QueryRelevanceResult containing all relevance assessments.
    """
    if not retrieval_result.context:
        # Return empty result for queries with no context
        empty_response = RelevanceAssessmentResponse(assessment=[])
        return QueryRelevanceResult(
            question_id=retrieval_result.question_id,
            question_text=retrieval_result.question_text,
            assessments=empty_response,
            total_chunks=0,
        )

    log.info(
        "Running relevance assessment for question %s with %d chunks",
        retrieval_result.question_id,
        len(retrieval_result.context),
    )

    # Convert context to TextUnit objects if needed
    text_units = []
    for item in retrieval_result.context:
        if isinstance(item, TextUnit):
            text_units.append(item)
        elif (
            isinstance(item, dict)
            and retrieval_result.context_id_key in item
            and retrieval_result.context_text_key in item
        ):
            # Convert dict to TextUnit
            text_unit = TextUnit(
                id=retrieval_result.get_context_item_id(item),
                short_id=retrieval_result.get_context_item_id(item),
                text=retrieval_result.get_context_item_text(item),
            )
            text_units.append(text_unit)
        else:
            msg = f"Context item must be TextUnit or dict with '{retrieval_result.context_id_key}' and '{retrieval_result.context_text_key}' keys, got: {type(item)}"
            raise ValueError(msg)

    # Run relevance assessment
    assessments = await relevance_rater.rate_relevance(
        query=retrieval_result.question_text, text_units=text_units
    )

    log.info(
        "Completed relevance assessment for question %s: %d assessments",
        retrieval_result.question_id,
        len(assessments.assessment),
    )

    return QueryRelevanceResult(
        question_id=retrieval_result.question_id,
        question_text=retrieval_result.question_text,
        assessments=assessments,
        total_chunks=len(retrieval_result.context),
    )


async def assess_batch_relevance(
    retrieval_results: list[RetrievalResult],
    relevance_rater: RelevanceRater,
    max_concurrent: int = 8,
) -> BatchRelevanceResult:
    """
    Run relevance assessment for multiple queries concurrently.

    Args:
        retrieval_results: List of RetrievalResult objects to assess.
        relevance_rater: The relevance rater instance to use for assessment.
        max_concurrent: Maximum number of concurrent assessments.

    Returns
    -------
        BatchRelevanceResult containing all query results.
    """
    if not retrieval_results:
        return BatchRelevanceResult(results=[])

    log.info(
        "Starting batch relevance assessment for %d queries with max_concurrent=%d",
        len(retrieval_results),
        max_concurrent,
    )

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def assess_single_with_semaphore(
        retrieval_result: RetrievalResult, index: int
    ) -> QueryRelevanceResult:
        async with semaphore:
            log.debug(
                "Processing query %d/%d: %s",
                index + 1,
                len(retrieval_results),
                retrieval_result.question_id,
            )
            return await assess_query_relevance(retrieval_result, relevance_rater)

    # Run assessments concurrently
    tasks = [
        assess_single_with_semaphore(retrieval_result, i)
        for i, retrieval_result in enumerate(retrieval_results)
    ]

    results = await asyncio.gather(*tasks)

    log.info("Completed batch relevance assessment: %d queries processed", len(results))

    return BatchRelevanceResult(results=results)
