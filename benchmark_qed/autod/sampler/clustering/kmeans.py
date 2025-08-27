# Copyright (c) 2025 Microsoft Corporation.
"""Performs Kmeans clustering on a dataset of text units."""

import logging
import math
from typing import Any, cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.clustering.base import BaseClustering
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster

log: logging.Logger = logging.getLogger(__name__)


class KmeansClustering(BaseClustering):
    """Kmeans clustering algorithm for text units."""

    def _find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int | None = None,
    ) -> int:
        """
        Find the optimal number of clusters using silhouette score.
        
        Args:
            embeddings: The text embeddings to cluster.
            min_clusters: Minimum number of clusters to try.
            max_clusters: Maximum number of clusters to try. If None, uses sqrt(n_samples/2).
            
        Returns:
            Optimal number of clusters based on highest silhouette score.
        """
        n_samples = len(embeddings)
        if max_clusters is None:
            max_clusters = max(int(math.sqrt(n_samples / 2)), min_clusters)
        
        # Ensure we don't try more clusters than we have samples
        max_clusters = min(max_clusters, n_samples - 1)
        
        if max_clusters < min_clusters:
            log.warning(f"Not enough samples ({n_samples}) for clustering. Using {min_clusters} clusters.")
            return min_clusters
        
        best_score = -1
        best_k = min_clusters
        
        log.info(f"Tuning number of clusters between {min_clusters} and {max_clusters} using silhouette score...")
        
        # Use tqdm to track progress of cluster tuning
        cluster_range = range(min_clusters, max_clusters + 1)
        with tqdm(cluster_range, desc="Finding optimal clusters", unit="k") as pbar:
            for k in pbar:
                try:
                    model = KMeans(
                        n_clusters=k, 
                        random_state=self.random_seed, 
                        n_init="auto"
                    ).fit(embeddings)
                    
                    score = silhouette_score(embeddings, model.labels_)
                    log.debug(f"k={k}, silhouette_score={score:.4f}")
                    
                    # Update progress bar with current best
                    pbar.set_postfix({
                        'current_k': k, 
                        'score': f'{score:.4f}', 
                        'best_k': best_k, 
                        'best_score': f'{best_score:.4f}'
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
                except Exception as e:
                    log.warning(f"Failed to compute silhouette score for k={k}: {e}")
                    pbar.set_postfix({
                        'current_k': k, 
                        'status': 'failed', 
                        'best_k': best_k, 
                        'best_score': f'{best_score:.4f}'
                    })
                    continue
        
        log.info(f"Optimal number of clusters: {best_k} (silhouette_score={best_score:.4f})")
        return best_k

    def cluster(
        self,
        text_units: list[TextUnit],
        num_clusters: int | None = None,
        **_kwargs: Any,
    ) -> list[TextCluster]:
        """Cluster the given text units into k clusters using Kmeans."""
        # cluster text units into num_clusters clusters using Kmeans
        filtered_text_units = [
            unit for unit in text_units if unit.text_embedding is not None
        ]
        embeddings = np.array([unit.text_embedding for unit in filtered_text_units])
        if len(embeddings) == 0:
            msg = "No valid text embeddings found in the text units."
            raise ValueError(msg)

        if num_clusters is None:
            # Use silhouette score to find optimal number of clusters
            num_clusters = self._find_optimal_clusters(embeddings)
        else:
            log.info(f"Using specified number of clusters: {num_clusters}")

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
