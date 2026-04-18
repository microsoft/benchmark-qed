# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests verifying batch distance matrix in KmeansTextSampler matches old per-element results."""

import copy

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.enums import (
    ClusterRepresentativeSelectionType,
    DistanceMetricType,
)
from benchmark_qed.autod.sampler.sampling.kmeans_sampler import KmeansTextSampler

_RNG = np.random.RandomState(42)


def _make_text_unit(
    uid: str, embedding: list[float] | None = None, dim: int = 16
) -> TextUnit:
    """Create a TextUnit with a random or specified embedding."""
    if embedding is None:
        embedding = _RNG.randn(dim).tolist()
    return TextUnit(
        id=uid,
        short_id=uid,
        text=f"text-{uid}",
        text_embedding=embedding,
    )


def _old_neighbor_selection(
    selected_reps: list[TextUnit],
    text_units: list[TextUnit],
    num_samples_per_cluster: int,
    distance_metric: DistanceMetricType = DistanceMetricType.COSINE,
) -> list[TextUnit]:
    """Reimplementation of the old per-element neighbor selection loop.

    Uses scipy cosine for each pair, rebuilds filtered corpus each iteration.
    This is the reference implementation we verify against.
    """
    selected_sample: list[TextUnit] = []
    selected_ids: set[str] = {rep.id for rep in selected_reps}
    corpus = [unit for unit in copy.deepcopy(text_units) if unit.id not in selected_ids]

    for index, rep in enumerate(selected_reps):
        if rep.attributes is None:
            rep.attributes = {}
        rep.attributes["is_representative"] = True
        rep.cluster_id = str(index)
        selected_sample.append(rep)

        available = [u for u in corpus if u.id not in selected_ids]
        query = np.array(rep.text_embedding)
        if distance_metric == DistanceMetricType.COSINE:
            available_sorted = sorted(
                available,
                key=lambda u: float(scipy_cosine(np.array(u.text_embedding), query)),
            )
        else:
            available_sorted = sorted(
                available,
                key=lambda u: float(np.linalg.norm(np.array(u.text_embedding) - query)),
            )

        neighbors = available_sorted[: num_samples_per_cluster - 1]
        for neighbor in neighbors:
            neighbor.cluster_id = str(index)
            if neighbor.attributes is None:
                neighbor.attributes = {}
            neighbor.attributes["is_representative"] = False
            selected_ids.add(neighbor.id)
        selected_sample.extend(neighbors)

    return selected_sample


class TestBatchDistanceMatrix:
    """Verify the batch matmul distance computation matches old per-element scipy code."""

    def test_cosine_batch_matches_scipy_pairwise(self) -> None:
        """Batch cosine distance matrix should match scipy element-by-element."""
        rng = np.random.RandomState(100)
        num_reps = 5
        num_corpus = 30
        dim = 16

        rep_embeddings = rng.randn(num_reps, dim)
        corpus_embeddings = rng.randn(num_corpus, dim)

        # Batch computation (same as in kmeans_sampler)
        rep_norms = np.linalg.norm(rep_embeddings, axis=1, keepdims=True)
        rep_norms = np.where(rep_norms == 0, 1.0, rep_norms)
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        corpus_norms = np.where(corpus_norms == 0, 1.0, corpus_norms)
        similarities = (rep_embeddings / rep_norms) @ (
            corpus_embeddings / corpus_norms
        ).T
        batch_distances = 1.0 - similarities

        # Reference: scipy per-element
        for i in range(num_reps):
            for j in range(num_corpus):
                expected = scipy_cosine(rep_embeddings[i], corpus_embeddings[j])
                np.testing.assert_allclose(
                    batch_distances[i, j],
                    expected,
                    atol=1e-10,
                    err_msg=f"Mismatch at rep={i}, corpus={j}",
                )

    def test_euclidean_batch_matches_numpy_pairwise(self) -> None:
        """Batch euclidean distance matrix should match per-element numpy."""
        rng = np.random.RandomState(200)
        num_reps = 5
        num_corpus = 30
        dim = 16

        rep_embeddings = rng.randn(num_reps, dim)
        corpus_embeddings = rng.randn(num_corpus, dim)

        # Batch computation (same as in kmeans_sampler)
        batch_distances = np.linalg.norm(
            rep_embeddings[:, np.newaxis, :] - corpus_embeddings[np.newaxis, :, :],
            axis=2,
        )

        for i in range(num_reps):
            for j in range(num_corpus):
                expected = np.linalg.norm(rep_embeddings[i] - corpus_embeddings[j])
                np.testing.assert_allclose(
                    batch_distances[i, j],
                    expected,
                    atol=1e-10,
                    err_msg=f"Mismatch at rep={i}, corpus={j}",
                )


class TestKmeansSamplerNeighborSelection:
    """Verify full KmeansTextSampler.sample produces same neighbor assignments as old code."""

    @pytest.fixture
    def small_dataset(self) -> list[TextUnit]:
        """Create a small dataset of 100 text units for testing."""
        rng = np.random.RandomState(500)
        units = []
        for i in range(100):
            unit = TextUnit(
                id=f"unit-{i}",
                short_id=f"{i}",
                text=f"text-{i}",
                text_embedding=rng.randn(16).tolist(),
            )
            units.append(unit)
        return units

    def test_sample_produces_expected_count(
        self, small_dataset: list[TextUnit]
    ) -> None:
        """Sampler should produce num_clusters * num_samples_per_cluster items."""
        sampler = KmeansTextSampler(random_seed=42)
        result = sampler.sample(
            text_units=small_dataset,
            sample_size=None,
            num_clusters=5,
            num_samples_per_cluster=3,
            representative_selection=ClusterRepresentativeSelectionType.CENTROID,
        )
        # 5 clusters * 3 samples each = 15
        assert len(result) == 15

    def test_no_duplicate_ids_in_sample(self, small_dataset: list[TextUnit]) -> None:
        """All sampled text units should have unique IDs."""
        sampler = KmeansTextSampler(random_seed=42)
        result = sampler.sample(
            text_units=small_dataset,
            sample_size=None,
            num_clusters=5,
            num_samples_per_cluster=3,
            representative_selection=ClusterRepresentativeSelectionType.CENTROID,
        )
        ids = [u.id for u in result]
        assert len(ids) == len(set(ids))

    def test_representatives_marked_correctly(
        self, small_dataset: list[TextUnit]
    ) -> None:
        """Each cluster should have exactly one representative."""
        sampler = KmeansTextSampler(random_seed=42)
        result = sampler.sample(
            text_units=small_dataset,
            sample_size=None,
            num_clusters=5,
            num_samples_per_cluster=3,
            representative_selection=ClusterRepresentativeSelectionType.CENTROID,
        )
        reps = [
            u for u in result if u.attributes and u.attributes.get("is_representative")
        ]
        non_reps = [
            u
            for u in result
            if u.attributes and not u.attributes.get("is_representative")
        ]
        assert len(reps) == 5
        assert len(non_reps) == 10

    def test_cluster_ids_assigned(self, small_dataset: list[TextUnit]) -> None:
        """All sampled units should have a cluster_id assigned."""
        sampler = KmeansTextSampler(random_seed=42)
        result = sampler.sample(
            text_units=small_dataset,
            sample_size=None,
            num_clusters=5,
            num_samples_per_cluster=3,
            representative_selection=ClusterRepresentativeSelectionType.CENTROID,
        )
        for unit in result:
            assert unit.cluster_id is not None

    def test_neighbor_selection_matches_old_code(self) -> None:
        """Given fixed reps + corpus, batch neighbor selection should pick same units as old code.

        This directly tests that the batch matmul + greedy argpartition approach
        selects the same nearest neighbors as the old sorted-scipy approach.
        """
        rng = np.random.RandomState(600)
        dim = 16
        # Create reps and corpus with known embeddings
        reps = []
        for i in range(3):
            unit = TextUnit(
                id=f"rep-{i}",
                short_id=f"rep-{i}",
                text=f"rep-{i}",
                text_embedding=rng.randn(dim).tolist(),
            )
            reps.append(unit)

        corpus_units = []
        for i in range(30):
            unit = TextUnit(
                id=f"corpus-{i}",
                short_id=f"corpus-{i}",
                text=f"corpus-{i}",
                text_embedding=rng.randn(dim).tolist(),
            )
            corpus_units.append(unit)

        num_samples_per_cluster = 4  # 1 rep + 3 neighbors

        # Old code: per-element scipy neighbor selection
        old_result = _old_neighbor_selection(
            selected_reps=copy.deepcopy(reps),
            text_units=corpus_units,
            num_samples_per_cluster=num_samples_per_cluster,
        )

        # New code: batch matmul neighbor selection (replicate sampler logic)
        new_reps = copy.deepcopy(reps)
        selected_ids: set[str] = {rep.id for rep in new_reps}
        corpus = [
            unit for unit in copy.deepcopy(corpus_units) if unit.id not in selected_ids
        ]
        corpus_embeddings = np.array([u.text_embedding for u in corpus])
        rep_embeddings = np.array([r.text_embedding for r in new_reps])
        n_neighbors = num_samples_per_cluster - 1

        # Batch distance computation
        rep_norms = np.linalg.norm(rep_embeddings, axis=1, keepdims=True)
        rep_norms = np.where(rep_norms == 0, 1.0, rep_norms)
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        corpus_norms = np.where(corpus_norms == 0, 1.0, corpus_norms)
        similarities = (rep_embeddings / rep_norms) @ (
            corpus_embeddings / corpus_norms
        ).T
        all_distances = 1.0 - similarities

        available_mask = np.ones(len(corpus), dtype=bool)
        new_result: list[TextUnit] = []
        for index, rep in enumerate(new_reps):
            if rep.attributes is None:
                rep.attributes = {}
            rep.attributes["is_representative"] = True
            rep.cluster_id = str(index)
            new_result.append(rep)

            distances_row = all_distances[index].copy()
            distances_row[~available_mask] = np.inf
            top_indices = np.argpartition(distances_row, n_neighbors)[:n_neighbors]
            top_indices = top_indices[np.argsort(distances_row[top_indices])]

            for idx in top_indices[:n_neighbors]:
                neighbor = corpus[idx]
                neighbor.cluster_id = str(index)
                if neighbor.attributes is None:
                    neighbor.attributes = {}
                neighbor.attributes["is_representative"] = False
                available_mask[idx] = False
                new_result.append(neighbor)

        # Compare: same IDs in same order
        old_ids = [u.id for u in old_result]
        new_ids = [u.id for u in new_result]
        assert old_ids == new_ids, (
            f"Neighbor selection mismatch.\nOld: {old_ids}\nNew: {new_ids}"
        )

    def test_single_sample_per_cluster(self, small_dataset: list[TextUnit]) -> None:
        """When num_samples_per_cluster=1, only reps are returned (no neighbor loop)."""
        sampler = KmeansTextSampler(random_seed=42)
        result = sampler.sample(
            text_units=small_dataset,
            sample_size=None,
            num_clusters=5,
            num_samples_per_cluster=1,
            representative_selection=ClusterRepresentativeSelectionType.CENTROID,
        )
        assert len(result) == 5
        for unit in result:
            assert unit.attributes is not None
            assert unit.attributes["is_representative"] is True
