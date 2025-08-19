# Copyright (c) 2025 Microsoft Corporation.
"""Ranking algorithms and utilities for assertion generation."""

from typing import Any, Callable


def calculate_rrf_scores(
    items: list[Any], 
    score_key_func: Callable[[Any], float | int], 
    source_count_key_func: Callable[[Any], int], 
    k: int = 60,
    score_weight: float = 0.3,
    source_count_weight: float = 0.7
) -> dict[int, float]:
    """
    Calculate Reciprocal Rank Fusion (RRF) scores for a list of items.
    
    RRF combines rankings from multiple criteria by computing reciprocal ranks.
    Formula: RRF_score = score_weight * (1/(k + score_rank)) + source_count_weight * (1/(k + source_count_rank))
    
    This is commonly used in information retrieval to combine different ranking signals.
    Higher RRF scores indicate better overall ranking across all criteria.
    
    Tie Handling: Uses dense ranking where tied items receive the same rank and 
    subsequent ranks skip appropriately. For example, values [10, 8, 8, 6] get 
    ranks [1, 2, 2, 4].
    
    Args:
        items: List of items to rank
        score_key_func: Function to extract importance score from item (higher is better)
        source_count_key_func: Function to extract source count from item (higher is better) 
        k: Smoothing constant (default=60, common in literature)
        score_weight: Weight for importance score ranking (default=0.5)
        source_count_weight: Weight for source count ranking (default=0.5)
        
    Returns:
        Dictionary mapping item id() to RRF score (higher scores = better ranking)
        
    Example:
        >>> documents = [
        ...     {"relevance": 0.9, "citations": 5, "title": "Doc A"},
        ...     {"relevance": 0.7, "citations": 10, "title": "Doc B"},
        ...     {"relevance": 0.8, "citations": 2, "title": "Doc C"},
        ... ]
        >>> rrf_scores = calculate_rrf_scores(
        ...     items=documents,
        ...     score_key_func=lambda d: d["relevance"],
        ...     source_count_key_func=lambda d: d["citations"]
        ... )
        >>> # Sort by RRF score (descending)
        >>> ranked_docs = sorted(documents, key=lambda d: -rrf_scores[id(d)])
        
        # Custom weights example (prioritize score over source count):
        >>> rrf_scores_weighted = calculate_rrf_scores(
        ...     items=documents,
        ...     score_key_func=lambda d: d["relevance"],
        ...     source_count_key_func=lambda d: d["citations"],
        ...     score_weight=0.7,
        ...     source_count_weight=0.3
        ... )
        
        # Items with identical scores will have identical RRF scores:
        >>> tied_docs = [{"score": 5, "sources": 2}, {"score": 5, "sources": 2}]
        >>> tied_scores = calculate_rrf_scores(tied_docs, lambda d: d["score"], lambda d: d["sources"])
        >>> # Both documents will have identical RRF scores
    """
    if not items:
        return {}
    
    # Create rankings for each criterion with proper tie handling using dense ranking
    # 1. Rank by importance score (descending: higher scores = better rank)
    score_ranks = calculate_dense_ranks(items, score_key_func, reverse=True)
    
    # 2. Rank by source count (descending: more sources = better rank)  
    source_count_ranks = calculate_dense_ranks(items, source_count_key_func, reverse=True)
    
    # Apply Reciprocal Rank Fusion (RRF) with weights
    rrf_scores = {}
    for item in items:
        item_id = id(item)
        score_rank = score_ranks[item_id]
        source_rank = source_count_ranks[item_id]
        
        # Calculate weighted RRF score (higher is better)
        rrf_score = (score_weight * (1 / (k + score_rank))) + (source_count_weight * (1 / (k + source_rank)))
        rrf_scores[item_id] = rrf_score
    
    return rrf_scores


def calculate_dense_ranks(items: list[Any], key_func: Callable[[Any], float | int], reverse: bool = True) -> dict[int, int]:
    """
    Calculate dense ranks for items, properly handling ties.
    
    Dense ranking assigns the same rank to tied items and skips subsequent ranks
    appropriately. This is the standard ranking method for handling ties in
    ranking algorithms.
    
    Examples:
        - Values [10, 8, 8, 6] get ranks [1, 2, 2, 4]
        - Values [5, 5, 3] get ranks [1, 1, 3]  
        - Values [7, 7, 7] get ranks [1, 1, 1]
    
    Args:
        items: List of items to rank
        key_func: Function to extract the value to rank by
        reverse: If True, higher values get better (lower) ranks (default: True)
        
    Returns:
        Dictionary mapping item id() to rank (1-based)
        
    Example:
        >>> items = [{"score": 10}, {"score": 8}, {"score": 8}, {"score": 6}]
        >>> ranks = calculate_dense_ranks(items, lambda x: x["score"], reverse=True)
        >>> # Results in ranks: [1, 2, 2, 4] for the respective items
    """
    if not items:
        return {}
    
    # Sort items by the ranking criterion
    sorted_items = sorted(items, key=key_func, reverse=reverse)
    
    ranks = {}
    current_rank = 1
    prev_value = None
    
    for i, item in enumerate(sorted_items):
        value = key_func(item)
        
        # If this value is different from previous, update rank to current position + 1
        if prev_value is not None and value != prev_value:
            current_rank = i + 1
            
        ranks[id(item)] = current_rank
        prev_value = value
    
    return ranks
