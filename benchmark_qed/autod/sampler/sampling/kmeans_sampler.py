import copy
import logging
import math
import random
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.clustering.base import BaseClustering
from benchmark_qed.autod.sampler.clustering.kmeans import KmeansClustering
from benchmark_qed.autod.sampler.enums import (
    ClusterRepresentativeSelectionType,
    DistanceMetricType,
)
from benchmark_qed.autod.sampler.neighboring.semantic_neighbors import (
    get_semantic_neighbors,
)
from benchmark_qed.autod.sampler.sampling.base import BaseTextSampler
from benchmark_qed.config.defaults import RANDOM_SEED

log: logging.Logger = logging.getLogger(__name__)


class KmeansTextSampler(BaseTextSampler):
    def __init__(self, random_seed: int | None = RANDOM_SEED):
        super().__init__(random_seed)
        self.cluster_model: BaseClustering = KmeansClustering(random_seed=random_seed)

    def sample(
        self,
        text_units: list[TextUnit],
        sample_size: int | None,
        num_clusters: int | None = None,
        num_samples_per_cluster: int | None = None,
        representative_selection: ClusterRepresentativeSelectionType = ClusterRepresentativeSelectionType.CENTROID,
        **kwargs: Any,
    ) -> list[TextUnit]:
        """
        Select a subset of text units from the corpus by clustering and selecting representatives from each cluster.
        Algorithm:
        1. Cluster the text units into a specified number of clusters.
        2. For each cluster, select a representative text unit based on the specified selection method (default is the centroid).
        3. For each representative, select a specified number of neighboring text units that are closest to the representative text.
        4. Return the selected text units as a sample.

        Parameters
        ----------
        - text_units: list of TextUnit objects to sample from.
        - sample_size: total number of text units to select (either specify sample size, or specify both num_clusters and num_samples_per_cluster).
        - num_clusters: number of clusters to form (optional). If not specified, it will be set to sample_size.
        - num_samples_per_cluster: number of samples to select from each cluster. If not specified, it will be set to sample_size / num_clusters.
        - representative_selection: method to select the representative text unit from each cluster.
        - kwargs: additional arguments for the clustering and selection methods.

        Returns
        -------
        - list of selected TextUnit objects.
        """
        if sample_size is None:
            # When sample_size is not provided, both cluster parameters must be specified
            if num_clusters is None or num_samples_per_cluster is None:
                raise ValueError(
                    "Either sample_size or both num_clusters and num_samples_per_cluster must be specified."
                )
            sample_size = num_clusters * num_samples_per_cluster
        else:
            # Default behavior when sample_size is provided
            if num_clusters is None:
                num_clusters = sample_size
            if num_samples_per_cluster is None:
                num_samples_per_cluster = math.ceil(sample_size / num_clusters)

        if sample_size is None or sample_size >= len(text_units):
            log.warning(
                "Sample size is None or  exceeds number of text units in the corpus. Returning the entire corpus."
            )
            return text_units

        # cluster text units into clusters
        clusters = self.cluster_model.cluster(
            text_units=text_units, num_clusters=num_clusters
        )

        # sort clusters by size
        clusters.sort(key=lambda x: len(x.text_units), reverse=True)

        # select a single representative of each cluster
        selected_reps: list[TextUnit] = []
        for cluster in clusters:
            match representative_selection:
                case ClusterRepresentativeSelectionType.CENTROID:
                    selected_reps.append(cluster.get_centroid_neighbors(1)[0])
                case ClusterRepresentativeSelectionType.NEIGHBOR_DISTANCE:
                    selected_reps.append(
                        cluster.get_representatives_by_shortest_distances(1)[0]
                    )
                case ClusterRepresentativeSelectionType.ATTRIBUTE_RANKING:
                    # check if there are ranking attributes
                    if "ranking_attributes" in kwargs:
                        ranking_attributes = kwargs["ranking_attributes"]
                    else:
                        raise ValueError(
                            "Attribute ranking requires 'ranking_attributes' to be specified."
                        )
                    ascending = kwargs.get("ascending", False)
                    selected_reps.append(
                        cluster.get_representatives_by_attribute_ranking(
                            n=1,
                            ranking_attributes=ranking_attributes,
                            ascending=ascending,
                        )[0]
                    )
                case _:
                    # This handles unexpected values not defined in the enum
                    raise ValueError(
                        f"Unsupported representative selection type: {representative_selection}"
                    )

        # for each representative, select a specified number of closest neighbors
        if num_samples_per_cluster == 1:
            # if only one sample per cluster is selected, return the representatives
            # add flag to rep's attributes to indicate that it is a representative
            for index, rep in enumerate(selected_reps):
                rep.cluster_id = str(index)
                if rep.attributes is None:
                    rep.attributes = {}
                rep.attributes["is_representative"] = True
            return selected_reps

        selected_sample: list[TextUnit] = []
        selected_ids: set[str] = set([rep.id for rep in selected_reps])
        corpus = [
            unit for unit in copy.deepcopy(text_units) if unit.id not in selected_ids
        ]
        for index, rep in enumerate(selected_reps):
            if rep.attributes is None:
                rep.attributes = {}
            rep.attributes["is_representative"] = True
            rep.cluster_id = str(index)
            selected_sample.append(rep)

            # Get neighbors
            neighbors = get_semantic_neighbors(
                text_unit=rep,
                corpus=[unit for unit in corpus if unit.id not in selected_ids],
                n=num_samples_per_cluster - 1,
                distance_metric=kwargs.get(
                    "distance_metric", DistanceMetricType.COSINE
                ),
            )

            for neighbor in neighbors:
                neighbor.cluster_id = str(index)
                if neighbor.attributes is None:
                    neighbor.attributes = {}
                neighbor.attributes["is_representative"] = False
                selected_ids.add(neighbor.id)
            selected_sample.extend(neighbors)

        if sample_size < len(selected_sample):
            selected_sample = random.sample(selected_sample, sample_size)

        return selected_sample
