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
    use_text_unit_short_id: bool = True
) -> dict[str, str]:
    """
    Create a mapping from text unit ID to cluster ID.
    
    Args:
        clusters: List of TextCluster objects.
        use_text_unit_short_id: Whether to use text unit short ID for mapping. Default to True. If False, use id (the uuid)

    Returns:
        Dictionary mapping text unit ID (short or uuid) to cluster ID.
    """
    mapping = {}
    for cluster in clusters:
        for text_unit in cluster.text_units:
            if use_text_unit_short_id:
                mapping[text_unit.short_id] = cluster.id
            else:
                mapping[text_unit.id] = cluster.id
    return mapping