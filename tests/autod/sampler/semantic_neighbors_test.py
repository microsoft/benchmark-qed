# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests verifying vectorized semantic neighbor functions match original scipy-based results."""

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.enums import DistanceMetricType
from benchmark_qed.autod.sampler.neighboring.semantic_neighbors import (
    _cosine_distances_vectorized,
    _euclidean_distances_vectorized,
    compute_similarity_to_references,
    get_semantic_neighbors,
)

_RNG = np.random.RandomState(42)


def _make_text_unit(
    uid: str, embedding: list[float] | None = None, dim: int = 8
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


def _scipy_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Reference cosine distance using scipy (the old implementation)."""
    return float(scipy_cosine(a, b))


def _scipy_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Reference euclidean distance."""
    return float(np.linalg.norm(a - b))


def _old_get_semantic_neighbors(
    text_unit: TextUnit,
    corpus: list[TextUnit],
    n: int = 5,
    distance_metric: DistanceMetricType = DistanceMetricType.COSINE,
) -> list[TextUnit]:
    """Reimplementation of the original scipy-based get_semantic_neighbors."""
    if len(corpus) <= n:
        return corpus
    if text_unit.text_embedding is None:
        return corpus[:n]

    query = np.array(text_unit.text_embedding)
    if distance_metric == DistanceMetricType.COSINE:
        neighbors = sorted(
            corpus,
            key=lambda unit: float(scipy_cosine(np.array(unit.text_embedding), query)),
        )
    else:
        neighbors = sorted(
            corpus,
            key=lambda unit: float(
                np.linalg.norm(np.array(unit.text_embedding) - query)
            ),
        )
    candidate_neighbors = []
    for unit in neighbors:
        if unit.id != text_unit.id:
            candidate_neighbors.append(unit)
        if len(candidate_neighbors) >= n:
            break
    return candidate_neighbors[:n]


class TestCosineDistancesVectorized:
    """Verify vectorized cosine distance matches scipy element-by-element."""

    def test_matches_scipy_random_vectors(self) -> None:
        """Vectorized cosine distances should match scipy for random data."""
        rng = np.random.RandomState(123)
        query = rng.randn(32)
        matrix = rng.randn(50, 32)

        vectorized = _cosine_distances_vectorized(query, matrix)
        expected = np.array([_scipy_cosine_distance(query, row) for row in matrix])
        np.testing.assert_allclose(vectorized, expected, atol=1e-10)

    def test_zero_query_returns_ones(self) -> None:
        """A zero query vector should return distance 1.0 for all rows."""
        query = np.zeros(8)
        matrix = _RNG.randn(10, 8)
        result = _cosine_distances_vectorized(query, matrix)
        np.testing.assert_array_equal(result, np.ones(10))

    def test_identical_vectors_return_zero(self) -> None:
        """Distance between identical vectors should be ~0."""
        vec = _RNG.randn(16)
        matrix = np.array([vec, vec])
        result = _cosine_distances_vectorized(vec, matrix)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have cosine distance ~1.0."""
        query = np.array([1.0, 0.0])
        matrix = np.array([[0.0, 1.0]])
        result = _cosine_distances_vectorized(query, matrix)
        np.testing.assert_allclose(result, [1.0], atol=1e-10)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have cosine distance ~2.0."""
        query = np.array([1.0, 0.0])
        matrix = np.array([[-1.0, 0.0]])
        result = _cosine_distances_vectorized(query, matrix)
        np.testing.assert_allclose(result, [2.0], atol=1e-10)


class TestEuclideanDistancesVectorized:
    """Verify vectorized euclidean distance matches per-element numpy."""

    def test_matches_numpy_random_vectors(self) -> None:
        """Vectorized euclidean distances should match per-element numpy."""
        rng = np.random.RandomState(456)
        query = rng.randn(32)
        matrix = rng.randn(50, 32)

        vectorized = _euclidean_distances_vectorized(query, matrix)
        expected = np.array([_scipy_euclidean_distance(query, row) for row in matrix])
        np.testing.assert_allclose(vectorized, expected, atol=1e-10)

    def test_identical_vectors_return_zero(self) -> None:
        """Distance between identical vectors should be 0."""
        vec = _RNG.randn(16)
        matrix = np.array([vec])
        result = _euclidean_distances_vectorized(vec, matrix)
        np.testing.assert_allclose(result, [0.0], atol=1e-10)


class TestGetSemanticNeighbors:
    """Verify new get_semantic_neighbors returns same results as old scipy-based version."""

    @pytest.fixture
    def corpus_and_query(self) -> tuple[TextUnit, list[TextUnit]]:
        """Create a query and a corpus of 20 text units with 8-dim embeddings."""
        rng = np.random.RandomState(789)
        query = _make_text_unit("query")
        # Override with seeded embedding
        query.text_embedding = rng.randn(8).tolist()
        corpus = []
        for i in range(20):
            unit = _make_text_unit(f"unit-{i}")
            unit.text_embedding = rng.randn(8).tolist()
            corpus.append(unit)
        return query, corpus

    def test_cosine_same_order_as_old(
        self, corpus_and_query: tuple[TextUnit, list[TextUnit]]
    ) -> None:
        """New cosine neighbors should return the same units in the same order."""
        query, corpus = corpus_and_query
        n = 5
        old_result = _old_get_semantic_neighbors(
            query, corpus, n=n, distance_metric=DistanceMetricType.COSINE
        )
        new_result = get_semantic_neighbors(
            query, corpus, n=n, distance_metric=DistanceMetricType.COSINE
        )
        assert [u.id for u in old_result] == [u.id for u in new_result]

    def test_euclidean_same_order_as_old(
        self, corpus_and_query: tuple[TextUnit, list[TextUnit]]
    ) -> None:
        """New euclidean neighbors should return the same units in the same order."""
        query, corpus = corpus_and_query
        n = 5
        old_result = _old_get_semantic_neighbors(
            query, corpus, n=n, distance_metric=DistanceMetricType.EUCLIDEAN
        )
        new_result = get_semantic_neighbors(
            query, corpus, n=n, distance_metric=DistanceMetricType.EUCLIDEAN
        )
        assert [u.id for u in old_result] == [u.id for u in new_result]

    def test_query_in_corpus_excluded(self) -> None:
        """If the query is in the corpus, it should be excluded from results."""
        rng = np.random.RandomState(101)
        query = _make_text_unit("q")
        query.text_embedding = rng.randn(8).tolist()
        corpus = [query] + [_make_text_unit(f"c-{i}") for i in range(10)]
        for unit in corpus[1:]:
            unit.text_embedding = rng.randn(8).tolist()

        result = get_semantic_neighbors(query, corpus, n=3)
        assert all(u.id != "q" for u in result)
        assert len(result) == 3

    def test_corpus_smaller_than_n(self) -> None:
        """If corpus <= n, return the entire corpus."""
        query = _make_text_unit("q")
        corpus = [_make_text_unit(f"c-{i}") for i in range(3)]
        result = get_semantic_neighbors(query, corpus, n=5)
        assert len(result) == 3

    def test_no_embedding_returns_first_n(self) -> None:
        """If the query has no embedding, return first n items."""
        query = TextUnit(id="q", short_id="q", text="q", text_embedding=None)
        corpus = [_make_text_unit(f"c-{i}") for i in range(10)]
        result = get_semantic_neighbors(query, corpus, n=3)
        assert [u.id for u in result] == ["c-0", "c-1", "c-2"]

    def test_precomputed_embeddings_same_result(
        self, corpus_and_query: tuple[TextUnit, list[TextUnit]]
    ) -> None:
        """Passing pre-computed corpus_embeddings should give same results."""
        query, corpus = corpus_and_query
        n = 5
        embeddings = np.array([u.text_embedding for u in corpus])

        result_auto = get_semantic_neighbors(query, corpus, n=n)
        result_precomputed = get_semantic_neighbors(
            query, corpus, n=n, corpus_embeddings=embeddings
        )
        assert [u.id for u in result_auto] == [u.id for u in result_precomputed]

    def test_large_n_cosine(self) -> None:
        """Requesting more neighbors than corpus size works correctly."""
        rng = np.random.RandomState(202)
        query = _make_text_unit("q")
        query.text_embedding = rng.randn(8).tolist()
        corpus = [_make_text_unit(f"c-{i}") for i in range(5)]
        for unit in corpus:
            unit.text_embedding = rng.randn(8).tolist()

        old_result = _old_get_semantic_neighbors(query, corpus, n=10)
        new_result = get_semantic_neighbors(query, corpus, n=10)
        assert [u.id for u in old_result] == [u.id for u in new_result]


class TestComputeSimilarityToReferences:
    """Verify vectorized compute_similarity_to_references matches scipy."""

    def test_matches_scipy_reference(self) -> None:
        """Result should match hand-computed scipy cosine similarities."""
        rng = np.random.RandomState(303)
        query_emb = rng.randn(16).tolist()
        refs = [_make_text_unit(f"r-{i}") for i in range(10)]
        for ref in refs:
            ref.text_embedding = rng.randn(16).tolist()

        result = compute_similarity_to_references(query_emb, refs)

        # Compute expected with scipy
        expected_sims = [
            1.0
            - _scipy_cosine_distance(np.array(query_emb), np.array(ref.text_embedding))
            for ref in refs
        ]
        assert result["min_similarity"] == pytest.approx(min(expected_sims), abs=1e-10)
        assert result["max_similarity"] == pytest.approx(max(expected_sims), abs=1e-10)
        assert result["mean_similarity"] == pytest.approx(
            float(np.mean(expected_sims)), abs=1e-10
        )

    def test_empty_references(self) -> None:
        """Empty references should return all zeros."""
        result = compute_similarity_to_references([1.0, 2.0], [])
        assert result == {
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "mean_similarity": 0.0,
        }

    def test_none_embeddings_filtered(self) -> None:
        """References with None embeddings should be excluded."""
        rng = np.random.RandomState(404)
        query_emb = rng.randn(8).tolist()
        ref_valid = _make_text_unit("valid")
        ref_valid.text_embedding = rng.randn(8).tolist()
        ref_none = TextUnit(
            id="none", short_id="none", text="none", text_embedding=None
        )

        result = compute_similarity_to_references(query_emb, [ref_valid, ref_none])
        expected_sim = 1.0 - _scipy_cosine_distance(
            np.array(query_emb), np.array(ref_valid.text_embedding)
        )
        assert result["min_similarity"] == pytest.approx(expected_sim, abs=1e-10)
        assert result["max_similarity"] == pytest.approx(expected_sim, abs=1e-10)
