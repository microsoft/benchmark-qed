# Copyright (c) 2025 Microsoft Corporation.
"""Module containing functions that take a list of ClusterRelevanceResult and return a list of relevant clusters with associated relevant units given a relevance threshold."""


from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentItem
from benchmark_qed.autoe.retrieval_scores.reference_gen.cluster_relevance import ClusterRelevanceResult


def get_relevant_units_per_cluster(
    cluster_results: list[ClusterRelevanceResult],
    relevance_threshold: int = 2,
) -> dict[str, list[TextUnit]]:
    """
    Extract relevant text units from each cluster based on relevance threshold.
    
    Args:
        cluster_results: List of ClusterRelevanceResult objects from cluster relevance assessment.
        relevance_threshold: Minimum relevance score threshold (units with score >= threshold are returned).
        
    Returns:
        Dictionary mapping cluster_id to list of relevant TextUnit objects that meet the threshold.
    """
    relevant_units_by_cluster = {}
    
    for cluster_result in cluster_results:
        # Filter assessments that meet the relevance threshold
        relevant_assessments = [
            item for item in cluster_result.all_assessments.assessment
            if item.score >= relevance_threshold and item.text_unit is not None
        ]
        
        # Extract the text units from relevant assessments
        relevant_units = [assessment.text_unit for assessment in relevant_assessments]
        
        relevant_units_by_cluster[cluster_result.cluster_id] = relevant_units
    
    return relevant_units_by_cluster

def get_relevant_clusters(
    cluster_results: list[ClusterRelevanceResult],
    relevance_threshold: int = 2,
) -> dict[str, int]:
    """
    Extract relevant clusters based on relevance threshold.

    Args:
        cluster_results: List of ClusterRelevanceResult objects from cluster relevance assessment.
        relevance_threshold: Minimum relevance score threshold (clusters with score >= threshold are returned).

    Returns:
        Dictionary mapping cluster_id to count of relevant text units that meet the threshold.
    """
    relevant_units_per_clusters = get_relevant_units_per_cluster(cluster_results, relevance_threshold=relevance_threshold)

    relevant_clusters = {}
    for cluster_id, relevant_units in relevant_units_per_clusters.items():
        if len(relevant_units) > 0:
            relevant_clusters[cluster_id] = len(relevant_units)

    return relevant_clusters