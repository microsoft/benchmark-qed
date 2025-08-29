# Copyright (c) 2025 Microsoft Corporation.
"""Module for running relevance assessments on a given query's retrieval context."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.retrieval_result import RetrievalResult
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentResponse, RelevanceAssessmentItem
from benchmark_qed.autoe.retrieval_scores.relevance_assessment.base import RelevanceRater

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryRelevanceResult:
    """Result of relevance assessment for a single query."""
    
    question_id: str
    """Unique identifier for the question."""
    
    question_text: str
    """The text of the question."""
    
    assessments: RelevanceAssessmentResponse
    """The relevance assessments for all text units."""
    
    total_chunks: int
    """Total number of chunks assessed."""
    
    def get_relevant_chunks(self, relevance_threshold: int = 2) -> list[dict[str, Any]]:
        """
        Get chunks that meet or exceed the relevance threshold.
        
        Args:
            relevance_threshold: Minimum score to consider a chunk relevant (default: 1).
        
        Returns:
            List of dictionaries containing relevant chunk information.
        """
        relevant_chunks = []
        for item in self.assessments.assessment:
            if item.score >= relevance_threshold:
                relevant_chunks.append({
                    "text_unit": item.text_unit,
                    "reasoning": item.reasoning,
                    "score": item.score
                })
        return relevant_chunks
    
    def get_relevant_count(self, relevance_threshold: int = 2) -> int:
        """
        Get the count of chunks that meet or exceed the relevance threshold.
        
        Args:
            relevance_threshold: Minimum score to consider a chunk relevant (default: 2).
        
        Returns:
            Number of relevant chunks.
        """
        return len([item for item in self.assessments.assessment if item.score >= relevance_threshold])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "assessments": {
                "assessment": [
                    {
                        "text_unit": {
                            "id": item.text_unit.id,
                            "short_id": item.text_unit.short_id,
                            "text": item.text_unit.text
                        },
                        "reasoning": item.reasoning,
                        "score": item.score
                    }
                    for item in self.assessments.assessment if item.text_unit is not None
                ]
            },
            "total_chunks": self.total_chunks
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryRelevanceResult":
        """Create from dictionary (JSON deserialization)."""
        assessment_items = []
        for item_data in data["assessments"]["assessment"]:
            text_unit = TextUnit(
                id=item_data["text_unit"]["id"],
                short_id=item_data["text_unit"]["short_id"],
                text=item_data["text_unit"]["text"]
            )
            assessment_items.append(RelevanceAssessmentItem(
                text_unit=text_unit,
                reasoning=item_data["reasoning"],
                score=item_data["score"]
            ))
        
        assessments = RelevanceAssessmentResponse(assessment=assessment_items)
        
        return cls(
            question_id=data["question_id"],
            question_text=data["question_text"],
            assessments=assessments,
            total_chunks=data["total_chunks"]
        )

@dataclass(frozen=True)
class BatchRelevanceResult:
    """Result of batch relevance assessment."""
    
    results: list[QueryRelevanceResult]
    """Individual query results."""
    
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [result.to_dict() for result in self.results]
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchRelevanceResult":
        """Create from dictionary (JSON deserialization)."""
        results = [QueryRelevanceResult.from_dict(result_data) for result_data in data["results"]]
        return cls(results=results)
    
    def save_to_json(self, filepath: str | Path) -> None:
        """
        Save BatchRelevanceResult to a JSON file.
        
        Args:
            filepath: Path to the JSON file to save to.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        log.info(f"Saved batch relevance results to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str | Path) -> "BatchRelevanceResult":
        """
        Load BatchRelevanceResult from a JSON file.
        
        Args:
            filepath: Path to the JSON file to load from.
            
        Returns:
            BatchRelevanceResult loaded from the file.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = cls.from_dict(data)
        log.info(f"Loaded batch relevance results from {filepath}: {result.total_queries} queries")
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

    Returns:
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

    log.info(f"Running relevance assessment for question {retrieval_result.question_id} with {len(retrieval_result.context)} chunks")

    # Convert context to TextUnit objects if needed
    text_units = []
    for item in retrieval_result.context:
        if isinstance(item, TextUnit):
            text_units.append(item)
        elif isinstance(item, dict) and retrieval_result.context_id_key in item and retrieval_result.context_text_key in item:
            # Convert dict to TextUnit
            text_unit = TextUnit(
                id=retrieval_result.get_context_item_id(item),
                short_id=retrieval_result.get_context_item_id(item),
                text=retrieval_result.get_context_item_text(item),
            )
            text_units.append(text_unit)
        else:
            raise ValueError(f"Context item must be TextUnit or dict with '{retrieval_result.context_id_key}' and '{retrieval_result.context_text_key}' keys, got: {type(item)}")

    # Run relevance assessment
    assessments = await relevance_rater.rate_relevance(
        query=retrieval_result.question_text, text_units=text_units
    )

    log.info(f"Completed relevance assessment for question {retrieval_result.question_id}: {len(assessments.assessment)} assessments")

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

    Returns:
        BatchRelevanceResult containing all query results.
    """
    if not retrieval_results:
        return BatchRelevanceResult(results=[])

    log.info(f"Starting batch relevance assessment for {len(retrieval_results)} queries with max_concurrent={max_concurrent}")

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def assess_single_with_semaphore(retrieval_result: RetrievalResult, index: int) -> QueryRelevanceResult:
        async with semaphore:
            log.debug(f"Processing query {index + 1}/{len(retrieval_results)}: {retrieval_result.question_id}")
            return await assess_query_relevance(retrieval_result, relevance_rater)

    # Run assessments concurrently
    tasks = [
        assess_single_with_semaphore(retrieval_result, i)
        for i, retrieval_result in enumerate(retrieval_results)
    ]

    results = await asyncio.gather(*tasks)

    log.info(f"Completed batch relevance assessment: {len(results)} queries processed")

    return BatchRelevanceResult(results=results)
