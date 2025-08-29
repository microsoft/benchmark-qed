# Copyright (c) 2025 Microsoft Corporation.
"""Module for calculating precision metrics from relevance assessment results."""

import logging
from typing import Any

import numpy as np

from benchmark_qed.autoe.retrieval_scores.scoring.retrieval_relevance import BatchRelevanceResult

log = logging.getLogger(__name__)


def calculate_binary_precision(
    batch_result: BatchRelevanceResult,
    relevance_threshold: int = 2
) -> dict[str, float]:
    """
    Calculate binary precision metrics from batch relevance results.
    
    Binary precision treats all scores >= threshold as relevant (1) and < threshold as not relevant (0).
    
    Args:
        batch_result: BatchRelevanceResult containing relevance assessments for multiple queries.
        relevance_threshold: Minimum score to consider a chunk relevant (default: 1).
    
    Returns:
        Dictionary containing precision metrics:
        - macro_averaged_precision: Average precision across all queries
        - macro_std_precision: Standard deviation of precision across queries
        - min_precision: Minimum precision score
        - max_precision: Maximum precision score
        - total_relevant: Total number of relevant chunks across all queries
        - total_chunks: Total number of chunks across all queries
        - micro_averaged_precision: Overall precision across all chunks
    """
    if not batch_result.results:
        return {
            "mean_precision": 0.0,
            "std_precision": 0.0,
            "min_precision": 0.0,
            "max_precision": 0.0,
            "total_relevant": 0,
            "total_chunks": 0,
            "overall_precision": 0.0
        }
    
    precisions = []
    total_relevant = 0
    total_chunks = 0
    
    for result in batch_result.results:
        relevant_count = result.get_relevant_count(relevance_threshold)
        precision = relevant_count / result.total_chunks if result.total_chunks > 0 else 0.0
        precisions.append(precision)
        total_chunks += result.total_chunks
        total_relevant += relevant_count

    micro_averaged_precision = total_relevant / total_chunks if total_chunks > 0 else 0.0
    
    log.info(f"Calculated binary precision for {len(batch_result.results)} queries: "
             f"macro-averaged precision={np.mean(precisions):.3f}, micro-averaged precision={micro_averaged_precision:.3f}")

    return {
        "macro_averaged_precision": float(np.mean(precisions)),
        "macro_std_precision": float(np.std(precisions)),
        "min_precision": float(np.min(precisions)),
        "max_precision": float(np.max(precisions)),
        "total_relevant": total_relevant,
        "total_chunks": total_chunks,
        "micro_averaged_precision": micro_averaged_precision
    }


def calculate_graded_precision(
    batch_result: BatchRelevanceResult,
    min_score: int = 0,
    max_score: int = 3
) -> dict[str, float]:
    """
    Calculate graded precision metrics from batch relevance results.

    Graded precision normalizes relevance scores between min and max scores:
    P_graded = (1/N_retrieved) * Σ(rel(i) - R_min)/(R_max - R_min)
    
    Args:
        batch_result: BatchRelevanceResult containing relevance assessments for multiple queries.
        min_score: Minimum relevance score (default: 0).
        max_score: Maximum relevance score (default: 3).
    
    Returns:
        Dictionary containing graded precision metrics:
        - macro_averaged_graded_precision: Average graded precision across all queries
        - macro_averaged_std_graded_precision: Standard deviation of graded precision
        - min_graded_precision: Minimum graded precision score
        - max_graded_precision: Maximum graded precision score
        - total_graded_score: Sum of all normalized scores
        - total_chunks: Total number of chunks across all queries
        - micro_averaged_graded_precision: Overall graded precision across all chunks
    """
    if not batch_result.results:
        return {
            "macro_averaged_precision": 0.0,
            "macro_averaged_std_precision": 0.0,
            "min_precision": 0.0,
            "max_precision": 0.0,
            "total_graded_score": 0.0,
            "total_chunks": 0,
            "micro_averaged_precision": 0.0
        }
    
    if max_score == min_score:
        log.warning(f"min_score and max_score are equal ({min_score}), returning zero precision")
        return {
            "macro_averaged_precision": 0.0,
            "macro_averaged_std_precision": 0.0,
            "min_precision": 0.0,
            "max_precision": 0.0,
            "total_graded_score": 0.0,
            "total_chunks": sum(result.total_chunks for result in batch_result.results),
            "micro_averaged_precision": 0.0
        }
    
    graded_precisions = []
    total_graded_score = 0.0
    total_chunks = 0
    
    for result in batch_result.results:
        query_graded_score = 0.0
        
        # Calculate graded score for this query using the formula
        for item in result.assessments.assessment:
            score = item.score
            # Normalize score: (rel(i) - R_min) / (R_max - R_min)
            normalized_score = (score - min_score) / (max_score - min_score)
            # Clamp to [0, 1] range in case scores are outside min/max bounds
            normalized_score = max(0.0, min(1.0, normalized_score))
            query_graded_score += normalized_score
        
        # Calculate precision for this query: (1/N_retrieved) * Σ normalized_scores
        query_precision = query_graded_score / result.total_chunks if result.total_chunks > 0 else 0.0
        graded_precisions.append(query_precision)
        
        total_graded_score += query_graded_score
        total_chunks += result.total_chunks

    micro_averaged_precision = total_graded_score / total_chunks if total_chunks > 0 else 0.0

    log.info(f"Calculated graded precision for {len(batch_result.results)} queries: "
             f"mean={np.mean(graded_precisions):.3f}, overall={micro_averaged_precision:.3f}, "
             f"score_range=[{min_score}, {max_score}]")
    
    return {
        "macro_averaged_precision": float(np.mean(graded_precisions)),
        "macro_averaged_std_precision": float(np.std(graded_precisions)),
        "min_precision": float(np.min(graded_precisions)),
        "max_precision": float(np.max(graded_precisions)),
        "total_graded_score": total_graded_score,
        "total_chunks": total_chunks,
        "micro_averaged_precision": micro_averaged_precision
    }


def calculate_precision_by_score_level(
    batch_result: BatchRelevanceResult,
    thresholds: list[int] = [1, 2, 3]
) -> dict[int, dict[str, float]]:
    """
    Calculate precision metrics at different score threshold levels.
    
    Args:
        batch_result: BatchRelevanceResult containing relevance assessments for multiple queries.
    
    Returns:
        Dictionary mapping score thresholds to precision metrics.
        Keys are threshold values (1, 2, 3), values are precision metric dictionaries.
    """
    results = {}

    for threshold in thresholds:
        results[threshold] = calculate_binary_precision(batch_result, threshold)
        log.debug(f"Precision at threshold {threshold}: {results[threshold]['macro_averaged_precision']:.3f}")

    return results


def get_precision_summary(
        batch_result: BatchRelevanceResult, 
        relevance_threshold: int=2,
        candidate_relevance_thresholds: list[int] = [1, 2, 3]
) -> dict[str, Any]:
    """
    Get a comprehensive summary of precision metrics.
    
    Args:
        batch_result: BatchRelevanceResult containing relevance assessments for multiple queries.
    
    Returns:
        Dictionary containing various precision calculations and summaries.
    """
    binary_metrics = calculate_binary_precision(batch_result)
    graded_metrics = calculate_graded_precision(batch_result)
    threshold_metrics = calculate_precision_by_score_level(batch_result, candidate_relevance_thresholds)

    return {
        "binary_precision": binary_metrics,
        "graded_precision": graded_metrics,
        "precision_by_threshold": threshold_metrics,
        "summary": {
            "total_queries": len(batch_result.results),
            "total_chunks": binary_metrics["total_chunks"],
            "binary_relevance_threshold": relevance_threshold,
            "macro_averaged_binary_precision": binary_metrics["macro_averaged_precision"],
            "macro_averaged_graded_precision": graded_metrics["macro_averaged_precision"],
            "total_relevant_chunks": binary_metrics["total_relevant"]
        }
    }
