import logging
from abc import ABC, abstractmethod
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
from benchmark_qed.config.defaults import RANDOM_SEED

log: logging.Logger = logging.getLogger(__name__)


class BaseClustering(ABC):
    def __init__(self, random_seed: int | None = RANDOM_SEED):
        self.random_seed = random_seed

    @abstractmethod
    def cluster(
        self, text_units: list[TextUnit], *args: Any, **kwargs: Any
    ) -> list[TextCluster]:
        """
        Cluster the given text units.
        """


def print_clusters(clusters: list[TextCluster]) -> None:
    for cluster in clusters:
        cluster_texts = [f"CLUSTER {cluster.id}:"]
        for unit in cluster.text_units:
            cluster_texts.append(f"Text: {unit.text}")
        log.info("\n".join(cluster_texts))
