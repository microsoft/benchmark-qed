# Copyright (c) 2025 Microsoft Corporation.
"""Module for calculating fidelity metrics using Jensen-Shannon divergence between reference and query distributions."""

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.clustering.base import create_text_unit_to_cluster_mapping
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
from benchmark_qed.autoe.retrieval_scores.scoring.retrieval_relevance import QueryRelevanceResult
from benchmark_qed.autoe.retrieval_scores.reference_gen.cluster_relevance import QueryClusterReferenceResult
from benchmark_qed.autoe.retrieval_scores.reference_gen.reference_context import get_relevant_clusters

log = logging.getLogger(__name__)


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
        
    Returns:
        Dictionary mapping cluster IDs to count of relevant text units in each cluster.
    """
    # Get relevant clusters based on cluster relevance results
    # This returns dict[str, int] where values are counts of relevant text units
    relevant_clusters_dict = get_relevant_clusters(
        cluster_results=retrieval_reference.cluster_results,
        relevance_threshold=relevance_threshold
    )
    
    # The relevant_clusters_dict already contains cluster_id -> count mapping
    return relevant_clusters_dict


def get_query_cluster_distribution(
    query_relevance_result: QueryRelevanceResult,
    text_unit_to_cluster_mapping: dict[str, str],
    relevance_threshold: int = 2,
    use_text_unit_short_id: bool = True
) -> dict[str, int]:
    """
    Get the distribution of relevant text units across clusters from query relevance results.
    
    Args:
        query_relevance_result: QueryRelevanceResult containing relevance assessments for retrieved text units.
        text_unit_to_cluster_mapping: Mapping from text unit ID to cluster ID.
        relevance_threshold: Minimum relevance score to consider a text unit relevant.
        use_text_unit_short_id: Whether to use the short ID of the text unit for mapping.
        
    Returns:
        Dictionary mapping cluster IDs to count of relevant text units in each cluster.
    """
    cluster_distribution = {}
    
    # Get relevant chunks based on the threshold
    relevant_chunks = query_relevance_result.get_relevant_chunks(relevance_threshold)
    
    for chunk_info in relevant_chunks:
        text_unit = chunk_info["text_unit"]
        if use_text_unit_short_id:
            text_unit_id = text_unit.short_id
        else:
            text_unit_id = text_unit.id
        
        # Map text unit to cluster
        if text_unit_id in text_unit_to_cluster_mapping:
            cluster_id = text_unit_to_cluster_mapping[text_unit_id]
            cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1
    
    return cluster_distribution


def calculate_js_divergence(dist1: dict[str, int], dist2: dict[str, int]) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.
    Only considers cluster IDs present in dist1 (reference distribution).
    
    Args:
        dist1: First distribution as dict mapping cluster IDs to counts (reference).
        dist2: Second distribution as dict mapping cluster IDs to counts (query).
        
    Returns:
        Jensen-Shannon divergence (0 = identical, 1 = completely different).
    """
    # Only consider cluster IDs from dist1 (reference distribution)
    cluster_ids = set(dist1.keys())
    
    if not cluster_ids:
        return 0.0  # Reference distribution is empty
    
    # Convert to probability distributions
    total1 = sum(dist1.values())
    # Only sum dist2 values for clusters that are present in cluster_ids (from dist1)
    total2 = sum(dist2.get(cluster_id, 0) for cluster_id in cluster_ids)
    
    if total1 == 0 and total2 == 0:
        return 0.0  # Both distributions are empty
    
    if total1 == 0 or total2 == 0:
        return 1.0  # One distribution is empty, maximum divergence
    
    # Create aligned probability vectors
    prob1 = np.array([dist1.get(cluster_id, 0) / total1 for cluster_id in sorted(cluster_ids)])
    prob2 = np.array([dist2.get(cluster_id, 0) / total2 for cluster_id in sorted(cluster_ids)])
    
    # Calculate Jensen-Shannon divergence
    js_divergence = jensenshannon(prob1, prob2)
    
    # Handle NaN case (can occur with zero probabilities)
    if np.isnan(js_divergence):
        return 0.0
        
    return float(js_divergence)


def calculate_single_query_fidelity(
    query_relevance_result: QueryRelevanceResult,
    retrieval_reference: QueryClusterReferenceResult,
    text_unit_to_cluster_mapping: dict[str, str],
    relevance_threshold: int = 2,
    use_text_unit_short_id: bool = True
) -> dict[str, Any]:
    """
    Calculate fidelity for a single query using Jensen-Shannon divergence.
    
    Fidelity = 1 - JS_divergence (higher values indicate better fidelity)
    
    Args:
        query_relevance_result: QueryRelevanceResult containing relevance assessments for retrieved text units.
        retrieval_reference: QueryClusterReferenceResult containing cluster relevance results.
        text_unit_to_cluster_mapping: Mapping from text unit ID to cluster ID.
        relevance_threshold: Minimum relevance score to consider relevant.
        use_text_unit_short_id: Whether to use the short ID of the text unit for mapping.
        
    Returns:
        Dictionary containing fidelity metrics:
        - fidelity: Fidelity score (0.0 to 1.0, higher is better)
        - js_divergence: Jensen-Shannon divergence (0.0 to 1.0, lower is better)
        - reference_distribution: Reference cluster distribution
        - query_distribution: Query cluster distribution
        - common_clusters: Number of clusters present in both distributions
        - reference_total_units: Total relevant units in reference
        - query_total_units: Total relevant units in query
    """
    # Get reference distribution from cluster relevance results
    reference_distribution = get_reference_cluster_distribution(
        retrieval_reference=retrieval_reference,
        relevance_threshold=relevance_threshold
    )
    
    # Get query distribution from query relevance results
    query_distribution = get_query_cluster_distribution(
        query_relevance_result=query_relevance_result,
        text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
        relevance_threshold=relevance_threshold,
        use_text_unit_short_id=use_text_unit_short_id
    )
    
    # Calculate Jensen-Shannon divergence
    js_divergence = calculate_js_divergence(reference_distribution, query_distribution)
    
    # Fidelity is 1 - JS divergence (higher is better)
    fidelity = 1.0 - js_divergence
    
    # Calculate additional metrics
    common_clusters = len(set(reference_distribution.keys()) & set(query_distribution.keys()))
    reference_total_units = sum(reference_distribution.values())
    query_total_units = sum(query_distribution.values())
    
    return {
        "fidelity": fidelity,
        "js_divergence": js_divergence,
        "reference_distribution": reference_distribution,
        "query_distribution": query_distribution,
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
    use_text_unit_short_id: bool = True
) -> dict[str, Any]:
    """
    Calculate fidelity metrics for multiple queries using Jensen-Shannon divergence.
    
    Args:
        query_relevance_results: List of QueryRelevanceResult objects for multiple queries.
        retrieval_references: List of QueryClusterReferenceResult objects from assess_batch.
        relevance_threshold: Minimum relevance score to consider relevant.
        text_unit_to_cluster_mapping: Mapping from text unit ID to cluster ID.
                                    If None, will be created from clusters parameter.
        clusters: List of TextCluster objects. Required if text_unit_to_cluster_mapping is None.
        use_text_unit_short_id: Whether to use the short ID of the text unit for mapping.
        
    Returns:
        Dictionary containing aggregate fidelity metrics:
        - macro_averaged_fidelity: Average fidelity across all queries
        - macro_std_fidelity: Standard deviation of fidelity across queries
        - min_fidelity: Minimum fidelity score
        - max_fidelity: Maximum fidelity score
        - macro_averaged_js_divergence: Average JS divergence across all queries
        - total_queries: Number of queries processed
        - relevance_threshold: The relevance threshold used
        - query_details: List of detailed results for each query
    """
    if not query_relevance_results or not retrieval_references:
        return {
            "macro_averaged_fidelity": 0.0,
            "macro_std_fidelity": 0.0,
            "min_fidelity": 0.0,
            "max_fidelity": 0.0,
            "macro_averaged_js_divergence": 1.0,
            "total_queries": 0,
            "relevance_threshold": relevance_threshold,
            "query_details": []
        }
    
    # Create text unit to cluster mapping if not provided
    if text_unit_to_cluster_mapping is None:
        if clusters is None:
            raise ValueError("Either text_unit_to_cluster_mapping or clusters must be provided")
        text_unit_to_cluster_mapping = create_text_unit_to_cluster_mapping(
            clusters, use_text_unit_short_id=use_text_unit_short_id
        )
    
    # Create a mapping from question_id to cluster relevance results
    cluster_references_by_question = {
        references.question_id: references.cluster_results
        for references in retrieval_references
    }
    
    query_fidelities = []
    query_js_divergences = []
    query_details = []
    
    for query_relevance_result in query_relevance_results:
        question_id = query_relevance_result.question_id
        
        # Get cluster relevance results for this question
        if question_id not in cluster_references_by_question:
            log.warning(f"No cluster relevance results found for question_id: {question_id}")
            continue
        
        cluster_reference = cluster_references_by_question[question_id]
        
        # Create a QueryClusterReferenceResult for this specific question
        single_query_reference = QueryClusterReferenceResult(
            question_id=question_id,
            question_text=query_relevance_result.question_text,
            cluster_results=cluster_reference
        )
        
        # Calculate fidelity for this query
        fidelity_result = calculate_single_query_fidelity(
            query_relevance_result=query_relevance_result,
            retrieval_reference=single_query_reference,
            text_unit_to_cluster_mapping=text_unit_to_cluster_mapping,
            relevance_threshold=relevance_threshold,
            use_text_unit_short_id=use_text_unit_short_id
        )
        
        query_fidelities.append(fidelity_result["fidelity"])
        query_js_divergences.append(fidelity_result["js_divergence"])
        query_details.append({
            "question_id": question_id,
            "question_text": query_relevance_result.question_text,
            **fidelity_result
        })
    
    if not query_fidelities:
        log.warning("No valid queries processed for fidelity calculation")
        return {
            "macro_averaged_fidelity": 0.0,
            "macro_std_fidelity": 0.0,
            "min_fidelity": 0.0,
            "max_fidelity": 0.0,
            "macro_averaged_js_divergence": 1.0,
            "total_queries": 0,
            "relevance_threshold": relevance_threshold,
            "query_details": []
        }
    
    # Calculate aggregate statistics
    macro_averaged_fidelity = np.mean(query_fidelities)
    macro_std_fidelity = np.std(query_fidelities)
    min_fidelity = np.min(query_fidelities)
    max_fidelity = np.max(query_fidelities)
    macro_averaged_js_divergence = np.mean(query_js_divergences)
    
    log.info(f"Calculated fidelity for {len(query_fidelities)} queries: "
             f"macro_avg_fidelity={macro_averaged_fidelity:.3f}, "
             f"macro_avg_js_divergence={macro_averaged_js_divergence:.3f}")
    
    return {
        "macro_averaged_fidelity": float(macro_averaged_fidelity),
        "macro_std_fidelity": float(macro_std_fidelity),
        "min_fidelity": float(min_fidelity),
        "max_fidelity": float(max_fidelity),
        "macro_averaged_js_divergence": float(macro_averaged_js_divergence),
        "total_queries": len(query_relevance_results),
        "relevance_threshold": relevance_threshold,
        "query_details": query_details
    }

