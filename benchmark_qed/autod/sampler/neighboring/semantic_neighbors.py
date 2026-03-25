# Copyright (c) 2025 Microsoft Corporation.
"""Functions to retrieve neighboring text units from a given text unit using text embedding similarity."""

import numpy as np
from numpy.typing import NDArray

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.sampler.enums import DistanceMetricType


def _cosine_distances_vectorized(
    query: NDArray[np.floating],
    matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute cosine distances between a query vector and a matrix of vectors.

    Uses vectorized numpy operations for performance on large corpora.
    Returns 1 - cosine_similarity for each row in the matrix.
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.ones(len(matrix), dtype=np.float64)
    row_norms = np.linalg.norm(matrix, axis=1)
    # Avoid division by zero for zero-norm rows
    safe_norms = np.where(row_norms == 0, 1.0, row_norms)
    similarities = matrix @ query / (safe_norms * query_norm)
    return 1.0 - similarities


def _euclidean_distances_vectorized(
    query: NDArray[np.floating],
    matrix: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute euclidean distances between a query vector and a matrix of vectors."""
    return np.linalg.norm(matrix - query, axis=1)  # type: ignore[return-value]


def _compute_distances(
    query: NDArray[np.floating],
    matrix: NDArray[np.floating],
    distance_metric: DistanceMetricType,
) -> NDArray[np.floating]:
    """Compute distances between a query vector and a matrix using the specified metric."""
    if distance_metric == DistanceMetricType.COSINE:
        return _cosine_distances_vectorized(query, matrix)
    return _euclidean_distances_vectorized(query, matrix)


def get_semantic_neighbors(
    text_unit: TextUnit,
    corpus: list[TextUnit],
    n: int = 5,
    distance_metric: DistanceMetricType = DistanceMetricType.COSINE,
    corpus_embeddings: NDArray[np.floating] | None = None,
) -> list[TextUnit]:
    """Get the n most semantically similar text units to the given text unit.

    Args:
        text_unit: The query text unit.
        corpus: List of candidate text units.
        n: Number of neighbors to return.
        distance_metric: Distance metric for similarity.
        corpus_embeddings: Pre-computed embedding matrix for the corpus.
            If provided, avoids rebuilding the matrix on each call.
    """
    if len(corpus) <= n:
        return corpus

    if text_unit.text_embedding is None:
        return corpus[:n]

    query_embedding = np.array(text_unit.text_embedding)

    if corpus_embeddings is None:
        corpus_embeddings = np.array([unit.text_embedding for unit in corpus])

    distances = _compute_distances(query_embedding, corpus_embeddings, distance_metric)
    # Get indices of the n+1 closest (in case query is in corpus)
    top_indices = np.argpartition(distances, min(n + 1, len(distances) - 1))[: n + 1]
    # Sort the top candidates by actual distance
    top_indices = top_indices[np.argsort(distances[top_indices])]

    candidate_neighbors = []
    for idx in top_indices:
        unit = corpus[idx]
        if unit.id != text_unit.id:
            candidate_neighbors.append(unit)
        if len(candidate_neighbors) >= n:
            break
    return candidate_neighbors[:n]


async def get_semantic_neighbors_from_text(
    text: str,
    corpus: list[TextUnit],
    text_embedder: TextEmbedder,
    n: int = 5,
    distance_metric: DistanceMetricType = DistanceMetricType.COSINE,
) -> list[TextUnit]:
    """Get the n most semantically similar text units to the given text string.

    Args:
        text: Text string to find neighbors for.
        corpus: List of text units to search for neighbors.
        text_embedder: Text embedder to generate embedding for the text.
        n: Number of neighbors to return.
        distance_metric: Distance metric to use for similarity calculation.

    Returns
    -------
        List of n most similar text units.
    """
    if len(corpus) <= n:
        return corpus

    # Generate embedding for the text
    text_embedding = await text_embedder.embed_raw_text(text)

    query = np.array(text_embedding)
    corpus_embeddings = np.array([
        unit.text_embedding for unit in corpus if unit.text_embedding is not None
    ])
    # Build filtered corpus matching the embedding matrix
    valid_corpus = [u for u in corpus if u.text_embedding is not None]

    distances = _compute_distances(query, corpus_embeddings, distance_metric)
    top_indices = np.argpartition(distances, min(n, len(distances) - 1))[:n]
    top_indices = top_indices[np.argsort(distances[top_indices])]

    return [valid_corpus[i] for i in top_indices]


def compute_similarity_to_references(
    text_embedding: list[float], references: list[TextUnit]
) -> dict[str, float]:
    """Compute min, max, and mean similarity between the text embedding and the reference text embeddings."""
    valid_refs = [ref for ref in references if ref.text_embedding is not None]
    if not valid_refs:
        return {
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "mean_similarity": 0.0,
        }
    query = np.array(text_embedding)
    ref_matrix = np.array([ref.text_embedding for ref in valid_refs])
    # Cosine similarity = 1 - cosine_distance
    distances = _cosine_distances_vectorized(query, ref_matrix)
    similarities = 1.0 - distances
    return {
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "mean_similarity": float(np.mean(similarities)),
    }


def compute_intra_inter_references_similarity_ratio(
    text_embedding: list[float],
    in_references: list[TextUnit],
    out_references: list[TextUnit],
) -> float:
    """Compute the ratio of the mean intra-references similarity to the mean inter-references similarity."""
    intra_similarity = compute_similarity_to_references(text_embedding, in_references)[
        "mean_similarity"
    ]
    inter_similarity = compute_similarity_to_references(text_embedding, out_references)[
        "mean_similarity"
    ]
    return (
        intra_similarity / inter_similarity
        if inter_similarity > 0
        else intra_similarity
    )
