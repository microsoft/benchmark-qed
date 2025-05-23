import logging
import math
from typing import Any, cast
from uuid import uuid4

import numpy as np
import tiktoken
from sklearn.cluster import KMeans

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.text_utils import num_tokens
from benchmark_qed.autod.sampler.clustering.base import BaseClustering
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster

log: logging.Logger = logging.getLogger(__name__)


class ConstraintKmeansClustering(BaseClustering):
    """
    Kmeans clustering while constraining for the text token size of each cluster
    """

    def __init__(self, token_encoder: tiktoken.Encoding | None = None):
        super().__init__()
        self.token_encoder = token_encoder

    def cluster(
        self,
        text_units: list[TextUnit],
        max_cluster_token_size: int = defs.MAX_DATA_TOKENS,
        **_kwargs: Any,
    ) -> list[TextCluster]:
        """
        Cluster the given text units into k clusters using Kmeans with token constraints.
        """
        # estimate the number of clusters based on the token size constraint
        corpus_token_size = sum([
            num_tokens(unit.text, self.token_encoder) for unit in text_units
        ])
        num_clusters = int(math.ceil(corpus_token_size / max_cluster_token_size))
        if num_clusters < 1:
            num_clusters = 1

        # cluster using kmeans
        embeddings = np.array([
            unit.text_embedding
            for unit in text_units
            if unit.text_embedding is not None
        ])
        if len(embeddings) == 0:
            raise ValueError("No valid text embeddings found in the text units.")

        model = KMeans(
            n_clusters=num_clusters, random_state=self.random_seed, n_init="auto"
        ).fit(embeddings)

        cluster_map: dict[int, list[TextUnit]] = {}
        for label, unit in zip(
            cast(np.ndarray, model.labels_), text_units, strict=False
        ):
            if label not in cluster_map:
                cluster_map[label] = [unit]
            else:
                cluster_map[label].append(unit)

        # split clusters that exceed the token size constraint
        text_clusters: list[TextCluster] = []
        for label, cluster in cluster_map.items():
            cluster_token_size = sum([
                num_tokens(unit.text, self.token_encoder) for unit in cluster
            ])
            if cluster_token_size > max_cluster_token_size:
                log.debug(
                    f"Cluster {label} exceeds token size constraint. Splitting into smaller clusters."
                )
                # split the cluster into smaller clusters
                sub_clusters = self.split_cluster(cluster, max_cluster_token_size)
                text_clusters.extend(sub_clusters)
            else:
                # add the cluster as is
                text_clusters.append(TextCluster(id=str(uuid4()), text_units=cluster))
        log.debug(
            f"Corpus token size: {corpus_token_size}. Number of clusters: {len(text_clusters)}"
        )

        return text_clusters

    def split_cluster(
        self, cluster: list[TextUnit], max_cluster_token_size: int
    ) -> list[TextCluster]:
        """
        Split the given cluster into smaller clusters that do not exceed the token size constraint.
        """
        sub_clusters: list[TextCluster] = []
        current_cluster: list[TextUnit] = []

        header = ["id", "text"]
        header_token_size = num_tokens("|".join(header) + "\n", self.token_encoder)
        current_token_size = header_token_size
        for unit in cluster:
            new_context = [unit.short_id, unit.text]
            unit_token_size = num_tokens(
                "|".join(new_context) + "\n", self.token_encoder
            )
            if current_token_size + unit_token_size > max_cluster_token_size:
                # create a new sub-cluster
                sub_clusters.append(
                    TextCluster(id=str(uuid4()), text_units=current_cluster)
                )
                current_cluster = [unit]
                current_token_size = header_token_size + unit_token_size
            else:
                current_cluster.append(unit)
                current_token_size += unit_token_size

        if len(current_cluster) > 0:
            # add the last sub-cluster
            sub_clusters.append(
                TextCluster(id=str(uuid4()), text_units=current_cluster)
            )

        return sub_clusters
