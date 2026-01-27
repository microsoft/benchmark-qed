# Copyright (c) 2025 Microsoft Corporation.
"""A module that support data clustering operations in AutoD."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
from benchmark_qed.config.defaults import RANDOM_SEED

log: logging.Logger = logging.getLogger(__name__)


class BaseClustering(ABC):
    """Base class for clustering algorithms."""

    def __init__(self, random_seed: int | None = RANDOM_SEED) -> None:
        self.random_seed = random_seed

    @abstractmethod
    def cluster(
        self, text_units: list[TextUnit], *args: Any, **kwargs: Any
    ) -> list[TextCluster]:
        """Cluster the given text units."""


def print_clusters(clusters: list[TextCluster]) -> None:
    """Print the clusters in a readable format.

    Args:
        clusters (list[TextCluster]): List of clusters to print.
    """
    for cluster in clusters:
        cluster_texts = [f"CLUSTER {cluster.id}:"]
        cluster_texts.extend(f"Text: {unit.text}" for unit in cluster.text_units)
        log.info("\n".join(cluster_texts))

def create_text_unit_to_cluster_mapping(
    clusters: list[TextCluster],
    match_by: str = "text",
) -> dict[str, str]:
    """
    Create a mapping from text unit identifier to cluster ID.
    
    Args:
        clusters: List of TextCluster objects.
        match_by: How to match text units. Options:
            - "text": Match by text content (normalized lowercase, stripped)
            - "id": Match by text unit ID
            - "short_id": Match by text unit short_id

    Returns:
        Dictionary mapping text unit identifier to cluster ID.
    """
    mapping = {}
    for cluster in clusters:
        for text_unit in cluster.text_units:
            if match_by == "text":
                key = text_unit.text.strip().lower()
            elif match_by == "id":
                key = text_unit.id
            elif match_by == "short_id":
                key = text_unit.short_id or text_unit.id
            else:
                msg = f"Invalid match_by value: {match_by}. Must be 'text', 'id', or 'short_id'."
                raise ValueError(msg)
            mapping[key] = cluster.id
    return mapping