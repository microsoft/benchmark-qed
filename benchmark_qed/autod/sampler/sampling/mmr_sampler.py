# Copyright (c) 2025 Microsoft Corporation.
"""Sampler that uses Maximal Marginal Relevance (MMR) to select diverse text units."""

import logging
import random
from typing import Any

import numpy as np

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.sampling.base import BaseTextSampler
from benchmark_qed.config.defaults import RANDOM_SEED

log: logging.Logger = logging.getLogger(__name__)


class MMRTextSampler(BaseTextSampler):
    """Sampler that uses Maximal Marginal Relevance to balance quality and diversity.

    MMR iteratively selects items that are both high-quality and diverse from
    already-selected items. The formula is:

        MMR(q) = λ * quality(q) - (1-λ) * max_similarity(q, selected)

    Where:
    - λ (lambda_param) controls the quality-diversity tradeoff (0=max diversity, 1=max quality)
    - quality(q) is a relevance/quality score for the item
    - max_similarity(q, selected) is the maximum cosine similarity to any already-selected item
    """

    def __init__(
        self,
        random_seed: int | None = RANDOM_SEED,
        lambda_param: float = 0.5,
    ) -> None:
        """Initialize the MMR sampler.

        Parameters
        ----------
        random_seed : int | None
            Random seed for reproducibility.
        lambda_param : float
            Trade-off parameter between quality and diversity.
            - 0.0 = maximize diversity (ignore quality)
            - 1.0 = maximize quality (ignore diversity)
            - 0.5 = balanced (default)
        """
        super().__init__(random_seed)
        self.lambda_param = lambda_param

    def sample(
        self,
        text_units: list[TextUnit],
        sample_size: int | None,
        lambda_param: float | None = None,
        quality_attributes: str | list[str] | None = None,
        **_kwargs: Any,
    ) -> list[TextUnit]:
        """Select a diverse subset of text units using MMR.

        Parameters
        ----------
        text_units : list[TextUnit]
            The text units to sample from. Must have text_embedding populated.
        sample_size : int | None
            Number of items to select. If None or >= len(text_units), returns all.
        lambda_param : float | None
            Override the instance lambda_param for this call.
        quality_attributes : str | list[str] | None
            Attribute name(s) in text_unit.attributes to use as quality score.
            If a list, scores are averaged. If None, all items are treated as
            equal quality (quality=1.0).

        Returns
        -------
        list[TextUnit]
            Selected text units in order of selection.
        """
        if sample_size is None or sample_size >= len(text_units):
            return text_units

        if len(text_units) == 0:
            return []

        # Use instance lambda if not overridden
        lam = lambda_param if lambda_param is not None else self.lambda_param

        # Validate embeddings exist
        units_with_embeddings = [u for u in text_units if u.text_embedding is not None]
        if len(units_with_embeddings) < len(text_units):
            log.warning(
                "Only %s/%s text units have embeddings. Using only units with embeddings.",
                len(units_with_embeddings),
                len(text_units),
            )
        if len(units_with_embeddings) == 0:
            log.warning(
                "No text units have embeddings. Falling back to random sampling."
            )
            return random.sample(text_units, min(sample_size, len(text_units)))

        text_units = units_with_embeddings

        # Build embedding matrix
        embeddings = np.array([u.text_embedding for u in text_units])

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        embeddings_normalized = embeddings / norms

        # Get quality scores
        quality_scores = self._get_quality_scores(text_units, quality_attributes)

        # MMR selection
        selected_indices: list[int] = []
        selected_embeddings: list[np.ndarray] = []
        remaining_indices = set(range(len(text_units)))

        # First selection: pick the centroid (most representative item)
        centroid = np.mean(embeddings_normalized, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        similarities_to_centroid = embeddings_normalized @ centroid
        first_idx = int(np.argmax(similarities_to_centroid))
        selected_indices.append(first_idx)
        selected_embeddings.append(embeddings_normalized[first_idx])
        remaining_indices.remove(first_idx)

        # Subsequent selections: use MMR
        for _ in range(min(sample_size, len(text_units)) - 1):
            best_idx = None
            best_score = float("-inf")

            for idx in remaining_indices:
                # Quality term
                quality = quality_scores[idx]

                # Diversity term: max similarity to any selected item
                similarities = [
                    float(np.dot(embeddings_normalized[idx], sel_emb))
                    for sel_emb in selected_embeddings
                ]
                max_sim = max(similarities)

                # MMR score
                mmr_score = lam * quality - (1 - lam) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(embeddings_normalized[best_idx])
                remaining_indices.remove(best_idx)

        selected_units = [text_units[i] for i in selected_indices]
        log.info(
            "MMR selected %s items from %s candidates (lambda=%s)",
            len(selected_units),
            len(text_units),
            lam,
        )
        return selected_units

    def _get_quality_scores(
        self,
        text_units: list[TextUnit],
        quality_attributes: str | list[str] | None,
    ) -> np.ndarray:
        """Extract quality scores from text units.

        Parameters
        ----------
        text_units : list[TextUnit]
            The text units to score.
        quality_attributes : str | list[str] | None
            Attribute name(s) to use as quality. If a list, normalized scores
            are averaged. If None, returns uniform scores.

        Returns
        -------
        np.ndarray
            Normalized quality scores in [0, 1].
        """
        if quality_attributes is None:
            # Uniform quality - all items equally good
            return np.ones(len(text_units))

        # Normalize to list
        if isinstance(quality_attributes, str):
            quality_attributes = [quality_attributes]

        # Get normalized scores for each attribute and average them
        all_scores = []
        for attr in quality_attributes:
            scores = self._get_single_attribute_scores(text_units, attr)
            all_scores.append(scores)

        # Average across all attributes
        return np.mean(all_scores, axis=0)

    def _get_single_attribute_scores(
        self,
        text_units: list[TextUnit],
        attribute: str,
    ) -> np.ndarray:
        """Extract and normalize scores for a single attribute."""
        scores = []
        for unit in text_units:
            if unit.attributes and attribute in unit.attributes:
                score = unit.attributes[attribute]
                scores.append(float(score) if score is not None else 0.0)
            else:
                scores.append(0.0)

        scores = np.array(scores)

        # Normalize to [0, 1]
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
        else:
            scores = np.ones(len(text_units))  # All same score -> treat as uniform

        return scores
