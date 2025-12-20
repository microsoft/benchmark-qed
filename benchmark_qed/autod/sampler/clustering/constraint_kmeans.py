# Copyright (c) 2025 Microsoft Corporation.
"""Performs Kmeans clustering with constraints on the text token size of each cluster."""

import logging
import math
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import tiktoken
from sklearn.cluster import KMeans

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.text_utils import num_tokens
from benchmark_qed.autod.sampler.clustering.base import BaseClustering
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster

log: logging.Logger = logging.getLogger(__name__)


class ConstraintKmeansClustering(BaseClustering):
    """Kmeans clustering while constraining for the text token size of each cluster."""

    def __init__(self, token_encoder: tiktoken.Encoding | None = None) -> None:
        super().__init__()
        self.token_encoder = token_encoder

    def cluster(
        self,
        text_units: list[TextUnit],
        max_cluster_token_size: int = defs.MAX_DATA_TOKENS,
        **_kwargs: Any,
    ) -> list[TextCluster]:
        """Cluster the given text units into k clusters using Kmeans with token constraints."""
        # estimate the number of clusters based on the token size constraint
        corpus_token_size = sum(
            num_tokens(unit.text, self.token_encoder) for unit in text_units
        )
        num_clusters = math.ceil(corpus_token_size / max_cluster_token_size)
        if num_clusters < 1:
            num_clusters = 1

        # cluster using kmeans
        embeddings = np.array([
            unit.text_embedding
            for unit in text_units
            if unit.text_embedding is not None
        ])
        if len(embeddings) == 0:
            msg = "No valid text embeddings found in the text units."
            raise ValueError(msg)

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
            cluster_token_size = self._compute_cluster_size(
                cluster, max_cluster_token_size
            )
            if cluster_token_size > max_cluster_token_size:
                msg = f"Cluster {label} exceeds token size constraint with {cluster_token_size} tokens. Splitting into smaller clusters."
                log.debug(msg)

                # recursively split the cluster into smaller clusters
                sub_clusters = self.cluster(cluster, max_cluster_token_size)
                text_clusters.extend(sub_clusters)
            else:
                # add the cluster as is
                text_clusters.append(TextCluster(id=str(uuid4()), text_units=cluster))
        msg = f"Corpus token size: {corpus_token_size}. Number of clusters: {len(text_clusters)}"
        log.debug(msg)

        return text_clusters

    def _compute_cluster_size(
        self, text_units: list[TextUnit], max_cluster_token_size: int
    ) -> int:
        texts = [
            {
                "id": unit.short_id or str(uuid4()),
                "text": unit.text,
            }
            for unit in text_units
        ]
        text_df = pd.DataFrame(texts)
        all_texts = text_df.to_csv(index=False, sep="|", header=True)
        return num_tokens(all_texts, self.token_encoder)
