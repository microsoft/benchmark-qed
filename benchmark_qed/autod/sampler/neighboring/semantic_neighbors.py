# Copyright (c) 2025 Microsoft Corporation.
"""
Functions to retrieve neighboring text units from a given text unit using text embedding similarity.
"""

import numpy as np
from scipy.spatial.distance import cosine

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.enums import DistanceMetricType


def get_semantic_neighbors(
    text_unit: TextUnit,
    corpus: list[TextUnit],
    n: int = 5,
    distance_metric: DistanceMetricType = DistanceMetricType.COSINE,
) -> list[TextUnit]:
    """Get the n most semantically similar text units to the given text unit."""
    if len(corpus) <= n:
        return corpus

    if text_unit.text_embedding is None:
        return corpus[:n]

    text_unit_embedding = np.array(text_unit.text_embedding)
    if distance_metric == DistanceMetricType.COSINE:
        neighbors = sorted(
            corpus,
            key=lambda unit: float(
                cosine(np.array(unit.text_embedding), text_unit_embedding)
            ),
        )
    else:
        neighbors = sorted(
            corpus,
            key=lambda unit: float(
                np.linalg.norm(np.array(unit.text_embedding) - text_unit_embedding)
            ),
        )

    # check if the text unit is in the neighbors list and remove it
    candidate_neighbors = []
    for unit in neighbors:
        if unit.id != text_unit.id:
            candidate_neighbors.append(unit)
        if len(candidate_neighbors) >= n:
            break
    return candidate_neighbors[:n]


def compute_similarity_to_references(
    text_embedding: list[float], references: list[TextUnit]
) -> dict[str, float]:
    """
    Compute min, max, and mean similarity between the text embedding and the reference text embeddings.
    """
    similarities: list[float] = []
    for reference in references:
        if reference.text_embedding is not None:
            similarities.append(
                1 - cosine(np.array(text_embedding), np.array(reference.text_embedding))
            )
    return {
        "min_similarity": min(similarities),
        "max_similarity": max(similarities),
        "mean_similarity": float(np.mean(similarities)) if len(similarities) > 0 else 0,
    }


def compute_intra_inter_references_similarity_ratio(
    text_embedding: list[float],
    in_references: list[TextUnit],
    out_references: list[TextUnit],
) -> float:
    """
    Compute the ratio of the mean intra-references similarity to the mean inter-references similarity.
    """
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
