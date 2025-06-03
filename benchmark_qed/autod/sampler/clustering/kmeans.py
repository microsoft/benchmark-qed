# Copyright (c) 2025 Microsoft Corporation.
"""Performs Kmeans clustering on a dataset of text units."""

import logging
import math
from typing import Any, cast

import numpy as np
from sklearn.cluster import KMeans

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.clustering.base import BaseClustering
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster

log: logging.Logger = logging.getLogger(__name__)


class KmeansClustering(BaseClustering):
    """Kmeans clustering algorithm for text units."""

    def cluster(
        self,
        text_units: list[TextUnit],
        num_clusters: int | None = None,
        **_kwargs: Any,
    ) -> list[TextCluster]:
        """Cluster the given text units into k clusters using Kmeans."""
        if num_clusters is None:
            # default to square root of (number of text units/2)
            num_clusters = max(int(math.sqrt(len(text_units) / 2)), 1)
            msg = f"Number of clusters not provided. Defaulting to {num_clusters}."
            log.info(msg)

        # cluster text units into num_clusters clusters using Kmeans
        filtered_text_units = [
            unit for unit in text_units if unit.text_embedding is not None
        ]
        embeddings = np.array([unit.text_embedding for unit in filtered_text_units])
        if len(embeddings) == 0:
            msg = "No valid text embeddings found in the text units."
            raise ValueError(msg)

        model = KMeans(
            n_clusters=num_clusters, random_state=self.random_seed, n_init="auto"
        ).fit(embeddings)
        clusters = {}
        for label, unit in zip(
            cast(np.ndarray, model.labels_), filtered_text_units, strict=False
        ):
            if label not in clusters:
                clusters[label] = [unit]
            else:
                clusters[label].append(unit)

        # log stats for cluster sizes (min, max, mean)
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        msg = f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes)}"
        log.info(msg)

        return [TextCluster(id=str(k), text_units=clusters[k]) for k in clusters]
