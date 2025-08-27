# Copyright (c) 2025 Microsoft Corporation.
"""Base classes for relevance assessment."""

from abc import ABC, abstractmethod

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentItem, RelevanceAssessmentResponse


class RelevanceRater(ABC):
    """Abstract base class for rating the relevance of text chunks to queries."""

    @abstractmethod
    async def rate_relevance(
        self, query: str, text_units: list[TextUnit]
    ) -> RelevanceAssessmentResponse:
        """
        Rate the relevance of text units to a query.

        Args:
            query: The query to assess relevance against.
            text_units: List of text units to assess.

        Returns:
            RelevanceAssessmentResponse containing assessment results.
        """
        pass

    def get_relevant_contexts(
        self, 
        result: RelevanceAssessmentResponse, 
        relevance_threshold: int = 1,
    ) -> list[RelevanceAssessmentItem]:
        """
        Filter assessment results to return only items that meet the relevance threshold.

        Args:
            result: The RelevanceAssessmentResponse containing assessment results.
            relevance_threshold: Minimum relevance score threshold (items with score >= threshold are returned).

        Returns:
            List of RelevanceAssessmentItem objects that meet or exceed the threshold.
        """
        return [
            item for item in result.assessment 
            if item.score >= relevance_threshold
        ]
