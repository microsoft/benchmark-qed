# Copyright (c) 2025 Microsoft Corporation.
"""Module for calculating cluster-level recall metrics from retrieval and cluster relevance results."""

import logging
from typing import Any

import numpy as np

from benchmark_qed.autod.sampler.clustering.base import (
    create_text_unit_to_cluster_mapping,
)
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
from benchmark_qed.autoe.retrieval_metrics.reference_gen.cluster_relevance import (
    QueryClusterReferenceResult,
)
from benchmark_qed.autoe.retrieval_metrics.reference_gen.reference_context import (
    get_relevant_clusters,
)
from benchmark_qed.autoe.retrieval_metrics.scoring.retrieval_relevance import (
    QueryRelevanceResult,
)

log: logging.Logger = logging.getLogger(__name__)


def _get_text_unit_key(text_unit: Any, match_by: str = "text") -> str:
    """Get the key for a text unit based on match_by setting."""
    if match_by == "text":
        return text_unit.text.strip().lower()
    if match_by == "id":
        return text_unit.id
    if match_by == "short_id":
        return text_unit.short_id or text_unit.id
    return text_unit.text.strip().lower()


def get_retrieved_clusters(
    query_relevance_result: QueryRelevanceResult,
    text_unit_to_cluster_mapping: dict[str, str],
    relevance_threshold: int = 2,
    match_by: str = "text",
) -> set[str]:
    """
    Get the set of clusters that were retrieved for a query based on relevant text units.

    Args:
        query_relevance_result: QueryRelevanceResult containing relevance assessments.
        text_unit_to_cluster_mapping: Mapping from text unit identifier to cluster ID.
        relevance_threshold: Minimum relevance score to consider a text unit retrieved.
        match_by: How to match text units ('text', 'id', or 'short_id').

    Returns
    -------
        Set of cluster IDs that contain relevant text units.
    """
    retrieved_clusters = set()

    # Get relevant chunks based on the threshold
    relevant_chunks = query_relevance_result.get_relevant_chunks(relevance_threshold)
    log.info("Retrieved %d relevant chunks for query", len(relevant_chunks))

    for chunk_info in relevant_chunks:
        text_unit = chunk_info["text_unit"]
        key = _get_text_unit_key(text_unit, match_by)

        # Map text unit to cluster
        if key in text_unit_to_cluster_mapping:
            cluster_id = text_unit_to_cluster_mapping[key]
            retrieved_clusters.add(cluster_id)
        else:
            log.warning("Text unit key '%s' not found in cluster mapping", key[:100])
    log.info("Retrieved %d clusters for query", len(retrieved_clusters))

    return retrieved_clusters


def calculate_single_query_recall(
    query_relevance_result: QueryRelevanceResult,
    retrieval_reference: QueryClusterReferenceResult,
    text_unit_to_cluster_mapping: dict[str, str],
    relevance_threshold: int = 2,
    match_by: str = "text",
) -> dict[str, Any]:
    """
    Calculate cluster-level recall for a single query.

    Recall = number of retrieved relevant clusters / total number of relevant clusters

    Args:
        query_relevance_result: QueryRelevanceResult containing relevance assessments for retrieved text units.
        retrieval_reference: QueryClusterReferenceResult containing cluster relevance results.
        text_unit_to_cluster_mapping: Mapping from text unit ID to cluster ID.
        relevance_threshold: Minimum relevance score to consider a cluster relevant.
        use_text_unit_short_id: Whether to use the short ID of the text unit for mapping.

    Returns
    -------
        Dictionary containing recall metrics:
        - recall: Recall score (0.0 to 1.0)
        - retrieved_clusters: Number of retrieved clusters
        - relevant_clusters: Number of relevant clusters (ground truth)
        - retrieved_relevant_clusters: Number of retrieved clusters that are relevant
        - retrieved_cluster_ids: Set of retrieved cluster IDs
        - relevant_cluster_ids: Set of relevant cluster IDs (ground truth)
        - retrieved_relevant_cluster_ids: Set of retrieved clusters that are relevant
    """
    # Get ground truth: relevant clusters based on cluster relevance results
    relevant_clusters_dict = get_relevant_clusters(
        cluster_results=retrieval_reference.cluster_results,  # These are ClusterRelevanceResult objects
        relevance_threshold=relevance_threshold,
    )
    relevant_cluster_ids = set(relevant_clusters_dict.keys())

    # Get retrieved clusters
    retrieved_cluster_ids = get_retrieved_clusters(
        query_relevance_result=query_relevance_result,
        text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
        relevance_threshold=relevance_threshold,
        match_by=match_by,
    )

    # Calculate intersection: retrieved clusters that are relevant
    retrieved_relevant_cluster_ids = retrieved_cluster_ids.intersection(
        relevant_cluster_ids
    )

    # Calculate recall
    total_relevant_clusters = len(relevant_cluster_ids)
    retrieved_relevant_clusters = len(retrieved_relevant_cluster_ids)
    log.info(
        "Retrieved %d clusters out of %d clusters",
        len(retrieved_cluster_ids),
        total_relevant_clusters,
    )

    recall = (
        retrieved_relevant_clusters / total_relevant_clusters
        if total_relevant_clusters > 0
        else 0.0
    )

    return {
        "recall": recall,
        "retrieved_clusters": len(retrieved_cluster_ids),
        "relevant_clusters": total_relevant_clusters,
        "retrieved_relevant_clusters": retrieved_relevant_clusters,
        "retrieved_cluster_ids": retrieved_cluster_ids,
        "relevant_cluster_ids": relevant_cluster_ids,
        "retrieved_relevant_cluster_ids": retrieved_relevant_cluster_ids,
        "cluster_classification_error": len([
            id for id in retrieved_cluster_ids if id not in relevant_cluster_ids
        ]),
    }


def calculate_recall(
    query_relevance_results: list[QueryRelevanceResult],
    retrieval_references: list[QueryClusterReferenceResult],
    relevance_threshold: int = 2,
    text_unit_to_cluster_mapping: dict[str, str] | None = None,
    clusters: list[TextCluster] | None = None,
    match_by: str = "text",
) -> dict[str, Any]:
    """
    Calculate cluster-level recall metrics for multiple queries.

    Args:
        query_relevance_results: List of QueryRelevanceResult objects for multiple queries.
        retrieval_references: List of QueryClusterReferenceResult objects from assess_batch.
        relevance_threshold: Minimum relevance score to consider a cluster relevant.
        text_unit_to_cluster_mapping: Mapping from text unit ID to cluster ID.
                                    If None, will be created from clusters parameter.
        clusters: List of TextCluster objects. Required if text_unit_to_cluster_mapping is None.

    Returns
    -------
        Dictionary containing aggregate recall metrics:
        - macro_averaged_recall: Average recall across all queries
        - macro_std_recall: Standard deviation of recall across queries
        - min_recall: Minimum recall score
        - max_recall: Maximum recall score
        - micro_averaged_recall: Overall recall across all queries (total retrieved relevant / total relevant)
        - total_queries: Number of queries processed
        - relevance_threshold: The relevance threshold used
        - total_classification_errors: Total number of irrelevant clusters retrieved across all queries
        - avg_classification_errors_per_query: Average number of classification errors per query
        - classification_error_rate: Rate of classification errors (errors / total retrieved clusters)
        - query_details: List of detailed results for each query
    """
    if not query_relevance_results or not retrieval_references:
        return {
            "macro_averaged_recall": 0.0,
            "macro_std_recall": 0.0,
            "min_recall": 0.0,
            "max_recall": 0.0,
            "micro_averaged_recall": 0.0,
            "total_queries": 0,
            "relevance_threshold": relevance_threshold,
            "total_classification_errors": 0,
            "avg_classification_errors_per_query": 0.0,
            "classification_error_rate": 0.0,
            "query_details": [],
        }

    # Create text unit to cluster mapping if not provided
    if text_unit_to_cluster_mapping is None:
        if clusters is None:
            msg = "Either text_unit_to_cluster_mapping or clusters must be provided"
            raise ValueError(msg)
        text_unit_to_cluster_mapping = create_text_unit_to_cluster_mapping(
            clusters, match_by=match_by
        )

    # Create a mapping from question_id to cluster relevance results
    cluster_references_by_question = {
        references.question_id: references.cluster_results
        for references in retrieval_references
    }

    query_recalls = []
    query_details = []
    total_relevant_clusters = 0
    total_retrieved_relevant_clusters = 0
    total_classification_errors = 0
    classification_errors_per_query = []

    for query_relevance_result in query_relevance_results:
        question_id = query_relevance_result.question_id

        # Get cluster relevance results for this question
        if question_id not in cluster_references_by_question:
            log.warning(
                "No cluster relevance results found for question_id: %s", question_id
            )
            continue

        cluster_reference = cluster_references_by_question[question_id]

        # Create a QueryClusterReferenceResult for this specific question
        single_query_reference = QueryClusterReferenceResult(
            question_id=question_id,
            question_text=query_relevance_result.question_text,
            cluster_results=cluster_reference,
        )

        # Calculate recall for this query
        if clusters is None:
            msg = "Clusters must be provided for calculate_recall"
            raise ValueError(msg)

        recall_result = calculate_single_query_recall(
            query_relevance_result=query_relevance_result,
            retrieval_reference=single_query_reference,
            relevance_threshold=relevance_threshold,
            text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
            match_by=match_by,
        )

        query_recalls.append(recall_result["recall"])
        query_details.append({
            "question_id": question_id,
            "question_text": query_relevance_result.question_text,
            **recall_result,
        })

        # Accumulate for micro-averaged recall and classification error stats
        total_relevant_clusters += recall_result["relevant_clusters"]
        total_retrieved_relevant_clusters += recall_result[
            "retrieved_relevant_clusters"
        ]
        total_classification_errors += recall_result["cluster_classification_error"]
        classification_errors_per_query.append(
            recall_result["cluster_classification_error"]
        )

    if not query_recalls:
        log.warning("No valid queries processed for recall calculation")
        return {
            "macro_averaged_recall": 0.0,
            "macro_std_recall": 0.0,
            "min_recall": 0.0,
            "max_recall": 0.0,
            "micro_averaged_recall": 0.0,
            "query_recalls": [],
            "query_details": [],
            "total_classification_errors": 0,
            "avg_classification_errors_per_query": 0.0,
            "classification_error_rate": 0.0,
        }

    macro_averaged_recall = np.mean(query_recalls)
    macro_std_recall = np.std(query_recalls)
    min_recall = np.min(query_recalls)
    max_recall = np.max(query_recalls)
    micro_averaged_recall = (
        total_retrieved_relevant_clusters / total_relevant_clusters
        if total_relevant_clusters > 0
        else 0.0
    )

    # Calculate cluster classification error statistics
    avg_classification_errors_per_query = np.mean(classification_errors_per_query)
    total_retrieved_clusters = sum(
        len(detail["retrieved_cluster_ids"]) for detail in query_details
    )
    classification_error_rate = (
        total_classification_errors / total_retrieved_clusters
        if total_retrieved_clusters > 0
        else 0.0
    )

    log.info(
        "Calculated recall for %d queries: "
        "macro_avg=%.3f, micro_avg=%.3f, "
        "total_classification_errors=%d, "
        "classification_error_rate=%.3f",
        len(query_recalls),
        macro_averaged_recall,
        micro_averaged_recall,
        total_classification_errors,
        classification_error_rate,
    )

    return {
        "macro_averaged_recall": float(macro_averaged_recall),
        "macro_std_recall": float(macro_std_recall),
        "min_recall": float(min_recall),
        "max_recall": float(max_recall),
        "micro_averaged_recall": float(micro_averaged_recall),
        "relevance_threshold": relevance_threshold,
        "total_queries": len(query_relevance_results),
        "total_classification_errors": total_classification_errors,
        "macro_averaged_classification_error": float(
            avg_classification_errors_per_query
        ),
        "micro_averaged_classification_error": float(classification_error_rate),
        "query_details": query_details,
    }
