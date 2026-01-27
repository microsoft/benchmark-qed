# Copyright (c) 2025 Microsoft Corporation.
"""Module for calculating fidelity metrics using Jensen-Shannon divergence and Total Variation Distance between reference and query distributions."""

import logging
from enum import Enum
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon

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

log = logging.getLogger(__name__)


class FidelityMetric(Enum):
    """Enum for fidelity distance metrics."""

    JENSEN_SHANNON = "js"
    TOTAL_VARIATION = "tvd"


def get_reference_cluster_distribution(
    retrieval_reference: QueryClusterReferenceResult,
    relevance_threshold: int = 2
) -> dict[str, int]:
    """
    Get the distribution of relevant text units across clusters from reference cluster relevance results.

    Args:
        retrieval_reference: QueryClusterReferenceResult containing cluster relevance results.
        text_unit_to_cluster_mapping: Mapping from text unit ID to cluster ID.
        relevance_threshold: Minimum relevance score to consider a cluster relevant.

    Returns
    -------
        Dictionary mapping cluster IDs to count of relevant text units in each cluster.
    """
    # Get relevant clusters based on cluster relevance results
    # This returns dict[str, int] where values are counts of relevant text units
    return get_relevant_clusters(
        cluster_results=retrieval_reference.cluster_results,
        relevance_threshold=relevance_threshold
    )

    # The relevant_clusters_dict already contains cluster_id -> count mapping


def get_query_cluster_distribution(
    query_relevance_result: QueryRelevanceResult,
    text_unit_to_cluster_mapping: dict[str, str],
    relevance_threshold: int = 2,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Get the distribution of relevant text units across clusters from query relevance results.

    Note: Chunk text must exactly match the reference text units (after normalization).
    If your RAG system adds metadata to chunks, preprocess them before evaluation.

    Args:
        query_relevance_result: QueryRelevanceResult containing relevance assessments for retrieved text units.
        text_unit_to_cluster_mapping: Mapping from normalized text to cluster ID.
        relevance_threshold: Minimum relevance score to consider a text unit relevant.

    Returns
    -------
        Tuple of:
        - Dictionary mapping cluster IDs to count of relevant text units in each cluster.
        - Dictionary with match statistics (matched, unmatched counts).
    """
    cluster_distribution: dict[str, int] = {}
    match_stats = {"matched": 0, "unmatched": 0}

    # Get relevant chunks based on the threshold
    relevant_chunks = query_relevance_result.get_relevant_chunks(relevance_threshold)

    for chunk_info in relevant_chunks:
        text_unit = chunk_info["text_unit"]
        normalized_text = text_unit.text.strip().lower()

        if normalized_text in text_unit_to_cluster_mapping:
            cluster_id = text_unit_to_cluster_mapping[normalized_text]
            cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1
            match_stats["matched"] += 1
        else:
            match_stats["unmatched"] += 1
            log.warning(
                "No cluster match for chunk in question '%s'. "
                "Ensure chunk text matches reference exactly (after strip/lowercase). "
                "Chunk preview: '%s...'",
                query_relevance_result.question_id,
                text_unit.text[:100].replace("\n", "\\n"),
            )

    # Log summary if there were unmatched chunks
    if match_stats["unmatched"] > 0:
        total = match_stats["matched"] + match_stats["unmatched"]
        log.warning(
            "Question %s: %d/%d chunks matched to clusters, %d unmatched",
            query_relevance_result.question_id,
            match_stats["matched"],
            total,
            match_stats["unmatched"],
        )

    return cluster_distribution, match_stats


def calculate_js_divergence(
    dist1: dict[str, int],
    dist2: dict[str, int],
    all_cluster_ids: list[str] | None = None,
) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.

    Args:
        dist1: First distribution as dict mapping cluster IDs to counts (reference).
        dist2: Second distribution as dict mapping cluster IDs to counts (query).
        all_cluster_ids: Complete list of all cluster IDs to consider. If None, uses union of dist1 and dist2 keys.

    Returns
    -------
        Jensen-Shannon divergence (0 = identical distributions, 1 = completely different).
    """
    # Use all cluster IDs if provided, otherwise use union of both distributions
    if all_cluster_ids is not None:
        cluster_ids = sorted(all_cluster_ids)
    else:
        cluster_ids = sorted(set(dist1.keys()) | set(dist2.keys()))

    if not cluster_ids:
        return 0.0  # No clusters to compare

    # Convert to probability distributions (total over ALL clusters)
    total1 = sum(dist1.get(cid, 0) for cid in cluster_ids)
    total2 = sum(dist2.get(cid, 0) for cid in cluster_ids)

    if total1 == 0 and total2 == 0:
        return 0.0  # Both distributions are empty

    if total1 == 0 or total2 == 0:
        return 1.0  # One distribution is empty, maximum divergence

    # Create probability distributions (normalized counts)
    prob1 = np.array([dist1.get(cid, 0) / total1 for cid in cluster_ids])
    prob2 = np.array([dist2.get(cid, 0) / total2 for cid in cluster_ids])

    # Calculate Jensen-Shannon divergence
    return float(jensenshannon(prob1, prob2, base=2))


def calculate_total_variation_distance(
    dist1: dict[str, int],
    dist2: dict[str, int],
    all_cluster_ids: list[str] | None = None,
) -> float:
    """
    Calculate Total Variation Distance between two distributions.

    Total Variation Distance is the appropriate metric for categorical distributions
    without spatial relationships. It measures the maximum difference in probability
    that the two distributions assign to any event.

    Formula: TVD = 0.5 * Σ|p1_i - p2_i|

    Args:
        dist1: First distribution as dict mapping cluster IDs to counts (reference).
        dist2: Second distribution as dict mapping cluster IDs to counts (query).
        all_cluster_ids: Complete list of all cluster IDs to consider. If None, uses union of dist1 and dist2 keys.

    Returns
    -------
        Total variation distance (0.0 = identical distributions, 1.0 = no overlap).
    """
    # Use all cluster IDs if provided, otherwise use union of both distributions
    if all_cluster_ids is not None:
        cluster_ids = sorted(all_cluster_ids)
    else:
        cluster_ids = sorted(set(dist1.keys()) | set(dist2.keys()))

    if not cluster_ids:
        return 0.0  # No clusters to compare

    # Convert to probability distributions (total over ALL clusters)
    total1 = sum(dist1.get(cid, 0) for cid in cluster_ids)
    total2 = sum(dist2.get(cid, 0) for cid in cluster_ids)

    if total1 == 0 and total2 == 0:
        return 0.0  # Both distributions are empty

    if total1 == 0 or total2 == 0:
        return 1.0  # One distribution is empty, maximum distance

    # Create probability distributions (normalized counts)
    prob1 = np.array([dist1.get(cid, 0) / total1 for cid in cluster_ids])
    prob2 = np.array([dist2.get(cid, 0) / total2 for cid in cluster_ids])

    # Calculate Total Variation Distance
    # TVD = 0.5 * sum of absolute differences in probabilities
    total_variation = 0.5 * np.sum(np.abs(prob1 - prob2))

    return float(total_variation)  # Always between 0.0 and 1.0


def calculate_single_query_fidelity(
    query_relevance_result: QueryRelevanceResult,
    retrieval_reference: QueryClusterReferenceResult,
    text_unit_to_cluster_mapping: dict[str, str],
    all_cluster_ids: list[str],
    relevance_threshold: int = 2,
    metric: FidelityMetric = FidelityMetric.JENSEN_SHANNON,
) -> dict[str, Any]:
    """
    Calculate fidelity for a single query using the specified distance metric.

    Fidelity measures how well the query's cluster distribution matches the reference.
    Uses all corpus clusters as the distribution space (clusters not present get 0 weight).

    Fidelity = 1 - distance_metric (higher values indicate better fidelity)

    Args:
        query_relevance_result: QueryRelevanceResult containing relevance assessments for retrieved text units.
        retrieval_reference: QueryClusterReferenceResult containing cluster relevance results.
        text_unit_to_cluster_mapping: Mapping from text unit ID to cluster ID.
        all_cluster_ids: Complete list of all cluster IDs in the corpus.
        relevance_threshold: Minimum relevance score to consider relevant.
        metric: Distance metric to use (FidelityMetric.JENSEN_SHANNON or FidelityMetric.TOTAL_VARIATION).

    Returns
    -------
        Dictionary containing fidelity metrics:
        - fidelity: Primary fidelity score using selected metric (0.0 to 1.0, higher is better)
        - distance: Primary distance score using selected metric (0.0 to 1.0, lower is better)
        - metric: The distance metric used
        - js_fidelity: JS fidelity score (always computed for analysis)
        - tvd_fidelity: Total Variation Distance fidelity score (always computed for analysis)
        - js_divergence: Jensen-Shannon divergence (always computed for analysis)
        - tvd_distance: Total Variation Distance (always computed for analysis)
        - reference_distribution: Reference cluster distribution
        - query_distribution: Query cluster distribution
        - all_clusters_count: Total number of clusters in corpus
        - common_clusters: Number of clusters present in both distributions
        - reference_total_units: Total relevant units in reference
        - query_total_units: Total relevant units in query
        - reference_clusters_count: Number of clusters with relevant units in reference
        - query_clusters_count: Number of clusters with relevant units in query
    """
    # Get reference distribution from cluster relevance results
    reference_distribution = get_reference_cluster_distribution(
        retrieval_reference=retrieval_reference,
        relevance_threshold=relevance_threshold
    )

    # Get query distribution from query relevance results
    query_distribution, _match_stats = get_query_cluster_distribution(
        query_relevance_result=query_relevance_result,
        text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
        relevance_threshold=relevance_threshold,
    )

    # Calculate both metrics using ALL cluster IDs as the space
    js_divergence = calculate_js_divergence(reference_distribution, query_distribution, all_cluster_ids)
    tvd_distance = calculate_total_variation_distance(reference_distribution, query_distribution, all_cluster_ids)

    # Fidelity metrics (higher is better for both)
    js_fidelity = 1.0 - js_divergence
    tvd_fidelity = 1.0 - tvd_distance

    # Select primary metric based on parameter
    if metric == FidelityMetric.JENSEN_SHANNON:
        primary_fidelity = js_fidelity
        primary_distance = js_divergence
    else:  # metric == FidelityMetric.TOTAL_VARIATION
        primary_fidelity = tvd_fidelity
        primary_distance = tvd_distance

    # Calculate additional metrics
    common_clusters = len(set(reference_distribution.keys()) & set(query_distribution.keys()))
    reference_total_units = sum(reference_distribution.values())
    query_total_units = sum(query_distribution.values())

    return {
        "fidelity": primary_fidelity,
        "distance": primary_distance,
        "metric": metric.value,  # Return string value instead of enum object
        "js_fidelity": js_fidelity,
        "tvd_fidelity": tvd_fidelity,
        "js_divergence": js_divergence,
        "tvd_distance": tvd_distance,
        "reference_distribution": reference_distribution,
        "query_distribution": query_distribution,
        "all_clusters_count": len(all_cluster_ids),
        "common_clusters": common_clusters,
        "reference_total_units": reference_total_units,
        "query_total_units": query_total_units,
        "reference_clusters_count": len(reference_distribution),
        "query_clusters_count": len(query_distribution)
    }


def calculate_fidelity(
    query_relevance_results: list[QueryRelevanceResult],
    retrieval_references: list[QueryClusterReferenceResult],
    relevance_threshold: int = 2,
    text_unit_to_cluster_mapping: dict[str, str] | None = None,
    clusters: list[TextCluster] | None = None,
    metric: FidelityMetric = FidelityMetric.JENSEN_SHANNON
) -> dict[str, Any]:
    """
    Calculate fidelity metrics for multiple queries using the specified distance metric.

    Args:
        query_relevance_results: List of QueryRelevanceResult objects for multiple queries.
        retrieval_references: List of QueryClusterReferenceResult objects from assess_batch.
        relevance_threshold: Minimum relevance score to consider relevant.
        text_unit_to_cluster_mapping: Mapping from text unit ID to cluster ID.
                                    If None, will be created from clusters parameter.
        clusters: List of TextCluster objects. Required if text_unit_to_cluster_mapping is None.

        metric: Distance metric to use (FidelityMetric.JENSEN_SHANNON for Jensen-Shannon divergence, FidelityMetric.TOTAL_VARIATION for Total Variation Distance).

    Returns
    -------
        Dictionary containing aggregate fidelity metrics:
        - macro_averaged_fidelity: Average fidelity using selected metric across all queries
        - macro_std_fidelity: Standard deviation of fidelity using selected metric across queries
        - min_fidelity: Minimum fidelity score using selected metric
        - max_fidelity: Maximum fidelity score using selected metric
        - macro_averaged_distance: Average distance using selected metric across all queries
        - metric: The distance metric used
        - macro_averaged_js_fidelity: Average JS fidelity across all queries
        - macro_std_js_fidelity: Standard deviation of JS fidelity
        - min_js_fidelity: Minimum JS fidelity score
        - max_js_fidelity: Maximum JS fidelity score
        - macro_averaged_js_divergence: Average JS divergence across all queries
        - macro_averaged_tvd_fidelity: Average Total Variation Distance fidelity across all queries
        - macro_std_tvd_fidelity: Standard deviation of Total Variation Distance fidelity
        - min_tvd_fidelity: Minimum Total Variation Distance fidelity score
        - max_tvd_fidelity: Maximum Total Variation Distance fidelity score
        - macro_averaged_tvd_distance: Average Total Variation Distance across all queries
        - total_queries: Number of queries processed
        - relevance_threshold: The relevance threshold used
        - query_details: List of detailed results for each query
    """
    if not query_relevance_results or not retrieval_references:
        return {
            # Primary metric (selected metric)
            "macro_averaged_fidelity": 0.0,
            "macro_std_fidelity": 0.0,
            "min_fidelity": 0.0,
            "max_fidelity": 0.0,
            "macro_averaged_distance": 1.0,
            "metric": metric.value,  # Return string value instead of enum object

            # Jensen-Shannon specific
            "macro_averaged_js_fidelity": 0.0,
            "macro_std_js_fidelity": 0.0,
            "min_js_fidelity": 0.0,
            "max_js_fidelity": 0.0,
            "macro_averaged_js_divergence": 1.0,

            # Total Variation Distance specific
            "macro_averaged_tvd_fidelity": 0.0,
            "macro_std_tvd_fidelity": 0.0,
            "min_tvd_fidelity": 0.0,
            "max_tvd_fidelity": 0.0,
            "macro_averaged_tvd_distance": 1.0,

            "total_queries": 0,
            "relevance_threshold": relevance_threshold,
            "query_details": []
        }

    # Create text unit to cluster mapping if not provided
    if text_unit_to_cluster_mapping is None:
        if clusters is None:
            msg = "Either text_unit_to_cluster_mapping or clusters must be provided"
            raise ValueError(msg)
        text_unit_to_cluster_mapping = create_text_unit_to_cluster_mapping(
            clusters,
        )

    # Get all cluster IDs from the corpus
    if clusters is None:
        msg = "clusters must be provided to get all cluster IDs for fidelity calculation"
        raise ValueError(msg)
    all_cluster_ids = [cluster.id for cluster in clusters]

    # Create a mapping from question_id to cluster relevance results
    cluster_references_by_question = {
        references.question_id: references.cluster_results
        for references in retrieval_references
    }

    query_fidelities = []
    query_distances = []
    query_js_fidelities = []
    query_tvd_fidelities = []
    query_js_divergences = []
    query_tvd_distances = []
    query_details = []

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
            cluster_results=cluster_reference
        )

        # Calculate fidelity for this query using the selected metric
        fidelity_result = calculate_single_query_fidelity(
            query_relevance_result=query_relevance_result,
            retrieval_reference=single_query_reference,
            text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
            all_cluster_ids=all_cluster_ids,
            relevance_threshold=relevance_threshold,
            metric=metric,
        )

        # Collect primary metrics based on selected metric
        query_fidelities.append(fidelity_result["fidelity"])
        query_distances.append(fidelity_result["distance"])

        # Always collect both metrics for analysis
        query_js_fidelities.append(fidelity_result["js_fidelity"])
        query_tvd_fidelities.append(fidelity_result["tvd_fidelity"])
        query_js_divergences.append(fidelity_result["js_divergence"])
        query_tvd_distances.append(fidelity_result["tvd_distance"])
        query_details.append({
            "question_id": question_id,
            "question_text": query_relevance_result.question_text,
            **fidelity_result
        })

    if not query_fidelities:
        log.warning("No valid queries processed for fidelity calculation")
        return {
            # Primary metric (selected metric)
            "macro_averaged_fidelity": 0.0,
            "macro_std_fidelity": 0.0,
            "min_fidelity": 0.0,
            "max_fidelity": 0.0,
            "macro_averaged_distance": 1.0,
            "metric": metric.value,  # Return string value instead of enum object

            # Jensen-Shannon specific
            "macro_averaged_js_fidelity": 0.0,
            "macro_std_js_fidelity": 0.0,
            "min_js_fidelity": 0.0,
            "max_js_fidelity": 0.0,
            "macro_averaged_js_divergence": 1.0,

            # Total Variation Distance specific
            "macro_averaged_tvd_fidelity": 0.0,
            "macro_std_tvd_fidelity": 0.0,
            "min_tvd_fidelity": 0.0,
            "max_tvd_fidelity": 0.0,
            "macro_averaged_tvd_distance": 1.0,

            "total_queries": 0,
            "relevance_threshold": relevance_threshold,
            "query_details": []
        }

    # Calculate aggregate statistics
    macro_averaged_fidelity = float(np.mean(query_fidelities))
    macro_std_fidelity = float(np.std(query_fidelities))
    min_fidelity = float(np.min(query_fidelities))
    max_fidelity = float(np.max(query_fidelities))
    macro_averaged_distance = float(np.mean(query_distances))

    # Jensen-Shannon specific statistics
    macro_averaged_js_fidelity = float(np.mean(query_js_fidelities))
    macro_std_js_fidelity = float(np.std(query_js_fidelities))
    min_js_fidelity = float(np.min(query_js_fidelities))
    max_js_fidelity = float(np.max(query_js_fidelities))
    macro_averaged_js_divergence = float(np.mean(query_js_divergences))

    # Total Variation Distance specific statistics
    macro_averaged_tvd_fidelity = float(np.mean(query_tvd_fidelities))
    macro_std_tvd_fidelity = float(np.std(query_tvd_fidelities))
    min_tvd_fidelity = float(np.min(query_tvd_fidelities))
    max_tvd_fidelity = float(np.max(query_tvd_fidelities))
    macro_averaged_tvd_distance = float(np.mean(query_tvd_distances))

    return {
        # Primary metric (selected metric)
        "macro_averaged_fidelity": macro_averaged_fidelity,
        "macro_std_fidelity": macro_std_fidelity,
        "min_fidelity": min_fidelity,
        "max_fidelity": max_fidelity,
        "macro_averaged_distance": macro_averaged_distance,
        "metric": metric.value,  # Return string value instead of enum object

        # Jensen-Shannon specific
        "macro_averaged_js_fidelity": macro_averaged_js_fidelity,
        "macro_std_js_fidelity": macro_std_js_fidelity,
        "min_js_fidelity": min_js_fidelity,
        "max_js_fidelity": max_js_fidelity,
        "macro_averaged_js_divergence": macro_averaged_js_divergence,

        # Total Variation Distance specific
        "macro_averaged_tvd_fidelity": macro_averaged_tvd_fidelity,
        "macro_std_tvd_fidelity": macro_std_tvd_fidelity,
        "min_tvd_fidelity": min_tvd_fidelity,
        "max_tvd_fidelity": max_tvd_fidelity,
        "macro_averaged_tvd_distance": macro_averaged_tvd_distance,

        "total_queries": len(query_fidelities),
        "relevance_threshold": relevance_threshold,
        "query_details": query_details
    }
