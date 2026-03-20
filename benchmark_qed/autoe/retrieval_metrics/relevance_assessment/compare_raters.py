# Copyright (c) 2025 Microsoft Corporation.
"""Utility for comparing relevance assessment between different raters."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.bing_rater import (
    BingRelevanceRater,
)
from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.rationale_rater import (
    RationaleRelevanceRater,
)
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.llm.factory import ModelFactory

log = logging.getLogger(__name__)


@dataclass
class RaterComparisonItem:
    """Single item comparison between two raters."""

    query: str
    text_unit_id: str
    text_unit_text: str
    bing_score: int
    rationale_score: int
    rationale_reasoning: str | None = None
    agreement: bool = field(init=False)
    score_diff: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived fields."""
        self.agreement = self.bing_score == self.rationale_score
        self.score_diff = abs(self.bing_score - self.rationale_score)


@dataclass
class RaterStats:
    """Statistics for a single rater's performance."""

    total_time_seconds: float = 0.0
    num_queries: int = 0
    num_text_units: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def avg_time_per_query(self) -> float:
        """Average time per query in seconds."""
        return (
            self.total_time_seconds / self.num_queries if self.num_queries > 0 else 0.0
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary."""
        return {
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_time_per_query_seconds": round(self.avg_time_per_query, 2),
            "num_queries": self.num_queries,
            "num_text_units": self.num_text_units,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class JudgeVerdict:
    """LLM judge verdict for a binary disagreement."""

    query: str
    text_unit_id: str
    text_unit_text: str
    bing_score: int
    rationale_score: int
    threshold: int
    bing_says_relevant: bool
    rationale_says_relevant: bool
    judge_says_relevant: bool
    judge_reasoning: str
    agrees_with_bing: bool = field(init=False)
    agrees_with_rationale: bool = field(init=False)

    def __post_init__(self) -> None:
        """Compute agreement fields."""
        self.agrees_with_bing = self.judge_says_relevant == self.bing_says_relevant
        self.agrees_with_rationale = (
            self.judge_says_relevant == self.rationale_says_relevant
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "text_unit_id": self.text_unit_id,
            "text_unit_text": self.text_unit_text[:200] + "..."
            if len(self.text_unit_text) > 200
            else self.text_unit_text,
            "threshold": self.threshold,
            "bing_score": self.bing_score,
            "rationale_score": self.rationale_score,
            "bing_says_relevant": self.bing_says_relevant,
            "rationale_says_relevant": self.rationale_says_relevant,
            "judge_says_relevant": self.judge_says_relevant,
            "judge_reasoning": self.judge_reasoning,
            "agrees_with_bing": self.agrees_with_bing,
            "agrees_with_rationale": self.agrees_with_rationale,
        }


@dataclass
class JudgeResults:
    """Aggregated results from LLM judge review of binary disagreements."""

    verdicts: list[JudgeVerdict]
    threshold: int
    total_disagreements: int = field(init=False)
    agrees_with_bing: int = field(init=False)
    agrees_with_rationale: int = field(init=False)
    bing_agreement_rate: float = field(init=False)
    rationale_agreement_rate: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute aggregated metrics."""
        self.total_disagreements = len(self.verdicts)
        if self.total_disagreements == 0:
            self.agrees_with_bing = 0
            self.agrees_with_rationale = 0
            self.bing_agreement_rate = 0.0
            self.rationale_agreement_rate = 0.0
            return

        self.agrees_with_bing = sum(1 for v in self.verdicts if v.agrees_with_bing)
        self.agrees_with_rationale = sum(
            1 for v in self.verdicts if v.agrees_with_rationale
        )
        self.bing_agreement_rate = self.agrees_with_bing / self.total_disagreements
        self.rationale_agreement_rate = (
            self.agrees_with_rationale / self.total_disagreements
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "threshold": self.threshold,
            "total_disagreements": self.total_disagreements,
            "agrees_with_bing": self.agrees_with_bing,
            "agrees_with_rationale": self.agrees_with_rationale,
            "bing_agreement_rate": self.bing_agreement_rate,
            "rationale_agreement_rate": self.rationale_agreement_rate,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


@dataclass
class RaterComparisonResult:
    """Aggregated comparison results between two raters."""

    items: list[RaterComparisonItem]
    total_comparisons: int = field(init=False)
    exact_agreements: int = field(init=False)
    agreement_rate: float = field(init=False)
    within_one_agreements: int = field(init=False)
    within_one_rate: float = field(init=False)
    mean_score_diff: float = field(init=False)
    bing_mean_score: float = field(init=False)
    rationale_mean_score: float = field(init=False)
    score_distribution: dict[str, dict[int, int]] = field(init=False)
    # Inter-rater reliability metrics
    cohens_kappa: float = field(init=False)
    weighted_kappa: float = field(init=False)
    spearman_correlation: float = field(init=False)
    pearson_correlation: float = field(init=False)
    confusion_matrix: list[list[int]] = field(init=False)
    # Binary agreement metrics at different thresholds
    binary_agreement_t1: dict[str, float] = field(init=False)  # threshold >= 1
    binary_agreement_t2: dict[str, float] = field(init=False)  # threshold >= 2
    # Rater performance stats (optional, set after init if available)
    bing_stats: RaterStats | None = field(default=None)
    rationale_stats: RaterStats | None = field(default=None)
    # LLM judge results for binary disagreements (optional)
    judge_results_t1: JudgeResults | None = field(default=None)
    judge_results_t2: JudgeResults | None = field(default=None)

    def __post_init__(self) -> None:
        """Compute aggregated metrics."""
        self.total_comparisons = len(self.items)

        if self.total_comparisons == 0:
            self.exact_agreements = 0
            self.agreement_rate = 0.0
            self.within_one_agreements = 0
            self.within_one_rate = 0.0
            self.mean_score_diff = 0.0
            self.bing_mean_score = 0.0
            self.rationale_mean_score = 0.0
            self.score_distribution = {"bing": {}, "rationale": {}}
            self.cohens_kappa = 0.0
            self.weighted_kappa = 0.0
            self.spearman_correlation = 0.0
            self.pearson_correlation = 0.0
            self.confusion_matrix = [[0] * 4 for _ in range(4)]
            self.binary_agreement_t1 = self._empty_binary_metrics()
            self.binary_agreement_t2 = self._empty_binary_metrics()
            return

        self.exact_agreements = sum(1 for item in self.items if item.agreement)
        self.agreement_rate = self.exact_agreements / self.total_comparisons

        self.within_one_agreements = sum(
            1 for item in self.items if item.score_diff <= 1
        )
        self.within_one_rate = self.within_one_agreements / self.total_comparisons

        self.mean_score_diff = (
            sum(item.score_diff for item in self.items) / self.total_comparisons
        )
        self.bing_mean_score = (
            sum(item.bing_score for item in self.items) / self.total_comparisons
        )
        self.rationale_mean_score = (
            sum(item.rationale_score for item in self.items) / self.total_comparisons
        )

        # Score distribution
        bing_dist: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        rationale_dist: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for item in self.items:
            bing_dist[item.bing_score] = bing_dist.get(item.bing_score, 0) + 1
            rationale_dist[item.rationale_score] = (
                rationale_dist.get(item.rationale_score, 0) + 1
            )
        self.score_distribution = {"bing": bing_dist, "rationale": rationale_dist}

        # Compute inter-rater reliability metrics
        self._compute_reliability_metrics()

        # Compute binary agreement metrics at different thresholds
        self.binary_agreement_t1 = self._compute_binary_agreement(threshold=1)
        self.binary_agreement_t2 = self._compute_binary_agreement(threshold=2)

    def _empty_binary_metrics(self) -> dict[str, float]:
        """Return empty binary metrics dictionary."""
        return {
            "agreement_rate": 0.0,
            "cohens_kappa": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "bing_positive_rate": 0.0,
            "rationale_positive_rate": 0.0,
        }

    def _compute_binary_agreement(self, threshold: int) -> dict[str, float]:
        """Compute binary agreement metrics at a given relevance threshold.

        Args:
            threshold: Score >= threshold is considered "relevant" (positive).

        Returns
        -------
            Dictionary with binary agreement metrics.

        """
        if not self.items:
            return self._empty_binary_metrics()

        # Convert to binary: 1 = relevant (score >= threshold), 0 = not relevant
        bing_binary = [1 if item.bing_score >= threshold else 0 for item in self.items]
        rationale_binary = [
            1 if item.rationale_score >= threshold else 0 for item in self.items
        ]

        n = len(self.items)
        agreements = sum(
            1 for b, r in zip(bing_binary, rationale_binary, strict=True) if b == r
        )
        agreement_rate = agreements / n

        # True positives, false positives, false negatives, true negatives
        # Using Rationale as "ground truth" for precision/recall calculation
        # (Bing is the predicted, Rationale is the reference)
        tp = sum(
            1
            for b, r in zip(bing_binary, rationale_binary, strict=True)
            if b == 1 and r == 1
        )
        fp = sum(
            1
            for b, r in zip(bing_binary, rationale_binary, strict=True)
            if b == 1 and r == 0
        )
        fn = sum(
            1
            for b, r in zip(bing_binary, rationale_binary, strict=True)
            if b == 0 and r == 1
        )

        # Precision, recall, F1 (bing vs rationale as reference)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Cohen's Kappa for binary
        kappa = self._compute_cohens_kappa(
            bing_binary, rationale_binary, num_categories=2
        )

        # Positive rates
        bing_positive_rate = sum(bing_binary) / n
        rationale_positive_rate = sum(rationale_binary) / n

        return {
            "agreement_rate": agreement_rate,
            "cohens_kappa": kappa,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "bing_positive_rate": bing_positive_rate,
            "rationale_positive_rate": rationale_positive_rate,
        }

    def _compute_reliability_metrics(self) -> None:
        """Compute inter-rater reliability metrics."""
        from scipy import stats

        bing_scores = [item.bing_score for item in self.items]
        rationale_scores = [item.rationale_score for item in self.items]

        # Confusion matrix (4x4 for scores 0-3)
        # Row = Bing score, Column = Rationale score
        self.confusion_matrix = [[0] * 4 for _ in range(4)]
        for item in self.items:
            self.confusion_matrix[item.bing_score][item.rationale_score] += 1

        # Cohen's Kappa (unweighted)
        self.cohens_kappa = self._compute_cohens_kappa(bing_scores, rationale_scores)

        # Weighted Kappa (quadratic weights - better for ordinal data)
        self.weighted_kappa = self._compute_weighted_kappa(
            bing_scores, rationale_scores, weights="quadratic"
        )

        # Spearman correlation (rank-based, good for ordinal)
        if len(set(bing_scores)) > 1 and len(set(rationale_scores)) > 1:
            spearman_result = stats.spearmanr(bing_scores, rationale_scores)
            self.spearman_correlation = float(spearman_result[0])  # type: ignore[arg-type]
        else:
            self.spearman_correlation = 0.0

        # Pearson correlation
        if len(set(bing_scores)) > 1 and len(set(rationale_scores)) > 1:
            pearson_result = stats.pearsonr(bing_scores, rationale_scores)
            self.pearson_correlation = float(pearson_result[0])  # type: ignore[arg-type]
        else:
            self.pearson_correlation = 0.0

    def _compute_cohens_kappa(
        self, rater1: list[int], rater2: list[int], num_categories: int = 4
    ) -> float:
        """Compute Cohen's Kappa coefficient.

        Args:
            rater1: Scores from first rater.
            rater2: Scores from second rater.
            num_categories: Number of score categories (default 4 for 0-3).

        Returns
        -------
            Cohen's Kappa coefficient (-1 to 1, 1 = perfect agreement).

        """
        n = len(rater1)
        if n == 0:
            return 0.0

        # Build confusion matrix
        matrix = [[0] * num_categories for _ in range(num_categories)]
        for r1, r2 in zip(rater1, rater2, strict=True):
            matrix[r1][r2] += 1

        # Observed agreement (diagonal sum / total)
        observed = sum(matrix[i][i] for i in range(num_categories)) / n

        # Expected agreement by chance
        row_sums = [sum(matrix[i]) for i in range(num_categories)]
        col_sums = [
            sum(matrix[i][j] for i in range(num_categories))
            for j in range(num_categories)
        ]
        expected = sum(row_sums[i] * col_sums[i] for i in range(num_categories)) / (
            n * n
        )

        # Kappa
        if expected == 1.0:
            return 1.0 if observed == 1.0 else 0.0
        return (observed - expected) / (1 - expected)

    def _compute_weighted_kappa(
        self,
        rater1: list[int],
        rater2: list[int],
        weights: str = "quadratic",
        num_categories: int = 4,
    ) -> float:
        """Compute weighted Cohen's Kappa (better for ordinal data).

        Args:
            rater1: Scores from first rater.
            rater2: Scores from second rater.
            weights: Weight scheme - "linear" or "quadratic".
            num_categories: Number of score categories (default 4 for 0-3).

        Returns
        -------
            Weighted Kappa coefficient (-1 to 1, 1 = perfect agreement).

        """
        n = len(rater1)
        if n == 0:
            return 0.0

        # Build confusion matrix
        matrix = [[0] * num_categories for _ in range(num_categories)]
        for r1, r2 in zip(rater1, rater2, strict=True):
            matrix[r1][r2] += 1

        # Build weight matrix
        weight_matrix = [[0.0] * num_categories for _ in range(num_categories)]
        for i in range(num_categories):
            for j in range(num_categories):
                if weights == "linear":
                    weight_matrix[i][j] = abs(i - j) / (num_categories - 1)
                else:  # quadratic
                    weight_matrix[i][j] = ((i - j) ** 2) / ((num_categories - 1) ** 2)

        # Row and column marginals
        row_sums = [sum(matrix[i]) / n for i in range(num_categories)]
        col_sums = [
            sum(matrix[i][j] for i in range(num_categories)) / n
            for j in range(num_categories)
        ]

        # Observed weighted disagreement
        observed = sum(
            weight_matrix[i][j] * matrix[i][j] / n
            for i in range(num_categories)
            for j in range(num_categories)
        )

        # Expected weighted disagreement
        expected = sum(
            weight_matrix[i][j] * row_sums[i] * col_sums[j]
            for i in range(num_categories)
            for j in range(num_categories)
        )

        # Weighted Kappa (1 - observed/expected)
        if expected == 0:
            return 1.0 if observed == 0 else 0.0
        return 1 - (observed / expected)

    def print_summary(self) -> None:
        """Print a summary of the comparison results."""
        print(f"\n{'=' * 60}")
        print("RATER COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total comparisons: {self.total_comparisons}")
        print("\nAgreement Metrics:")
        print(
            f"  Exact agreement: {self.exact_agreements}/{self.total_comparisons} ({self.agreement_rate:.1%})"
        )
        print(
            f"  Within 1 point:  {self.within_one_agreements}/{self.total_comparisons} ({self.within_one_rate:.1%})"
        )
        print(f"  Mean score diff: {self.mean_score_diff:.2f}")
        print("\nInter-Rater Reliability:")
        print(f"  Cohen's Kappa:       {self.cohens_kappa:.3f}")
        print(f"  Weighted Kappa (QW): {self.weighted_kappa:.3f}")
        print(f"  Spearman corr:       {self.spearman_correlation:.3f}")
        print(f"  Pearson corr:        {self.pearson_correlation:.3f}")
        print("\n  Kappa Interpretation:")
        print("    < 0.00: Poor | 0.00-0.20: Slight | 0.21-0.40: Fair")
        print(
            "    0.41-0.60: Moderate | 0.61-0.80: Substantial | 0.81-1.00: Almost Perfect"
        )
        print("\nMean Scores:")
        print(f"  Bing rater:      {self.bing_mean_score:.2f}")
        print(f"  Rationale rater: {self.rationale_mean_score:.2f}")
        print("\nScore Distribution:")
        print(f"  {'Score':<8} {'Bing':<10} {'Rationale':<10}")
        for score in range(4):
            print(
                f"  {score:<8} {self.score_distribution['bing'].get(score, 0):<10} {self.score_distribution['rationale'].get(score, 0):<10}"
            )
        print("\nConfusion Matrix (Bing rows x Rationale cols):")
        print(f"  {'':>6} {0:>6} {1:>6} {2:>6} {3:>6}")
        for i in range(4):
            row = self.confusion_matrix[i]
            print(f"  {i:>6} {row[0]:>6} {row[1]:>6} {row[2]:>6} {row[3]:>6}")
        print("\nBinary Agreement (threshold >= 1, i.e. score 1+ is relevant):")
        print(
            f"  Agreement rate:      {self.binary_agreement_t1['agreement_rate']:.1%}"
        )
        print(f"  Cohen's Kappa:       {self.binary_agreement_t1['cohens_kappa']:.3f}")
        print(f"  Precision:           {self.binary_agreement_t1['precision']:.3f}")
        print(f"  Recall:              {self.binary_agreement_t1['recall']:.3f}")
        print(f"  F1 Score:            {self.binary_agreement_t1['f1_score']:.3f}")
        print(
            f"  Bing positive rate:  {self.binary_agreement_t1['bing_positive_rate']:.1%}"
        )
        print(
            f"  Rationale pos rate:  {self.binary_agreement_t1['rationale_positive_rate']:.1%}"
        )
        print("\nBinary Agreement (threshold >= 2, i.e. score 2+ is relevant):")
        print(
            f"  Agreement rate:      {self.binary_agreement_t2['agreement_rate']:.1%}"
        )
        print(f"  Cohen's Kappa:       {self.binary_agreement_t2['cohens_kappa']:.3f}")
        print(f"  Precision:           {self.binary_agreement_t2['precision']:.3f}")
        print(f"  Recall:              {self.binary_agreement_t2['recall']:.3f}")
        print(f"  F1 Score:            {self.binary_agreement_t2['f1_score']:.3f}")
        print(
            f"  Bing positive rate:  {self.binary_agreement_t2['bing_positive_rate']:.1%}"
        )
        print(
            f"  Rationale pos rate:  {self.binary_agreement_t2['rationale_positive_rate']:.1%}"
        )
        if self.bing_stats or self.rationale_stats:
            print("\nRater Performance:")
            if self.bing_stats:
                print("  Bing Rater:")
                print(
                    f"    Total time:       {self.bing_stats.total_time_seconds:.1f}s"
                )
                print(
                    f"    Avg time/query:   {self.bing_stats.avg_time_per_query:.2f}s"
                )
                print(f"    Prompt tokens:    {self.bing_stats.prompt_tokens:,}")
                print(f"    Completion tokens:{self.bing_stats.completion_tokens:,}")
                print(f"    Total tokens:     {self.bing_stats.total_tokens:,}")
            if self.rationale_stats:
                print("  Rationale Rater:")
                print(
                    f"    Total time:       {self.rationale_stats.total_time_seconds:.1f}s"
                )
                print(
                    f"    Avg time/query:   {self.rationale_stats.avg_time_per_query:.2f}s"
                )
                print(f"    Prompt tokens:    {self.rationale_stats.prompt_tokens:,}")
                print(
                    f"    Completion tokens:{self.rationale_stats.completion_tokens:,}"
                )
                print(f"    Total tokens:     {self.rationale_stats.total_tokens:,}")
        if self.judge_results_t1 or self.judge_results_t2:
            print("\nLLM Judge Review of Binary Disagreements:")
            if self.judge_results_t1 and self.judge_results_t1.total_disagreements > 0:
                print("  Threshold >= 1:")
                print(
                    f"    Disagreements reviewed: {self.judge_results_t1.total_disagreements}"
                )
                print(
                    f"    Judge agrees with Bing:      {self.judge_results_t1.agrees_with_bing} ({self.judge_results_t1.bing_agreement_rate:.1%})"
                )
                print(
                    f"    Judge agrees with Rationale: {self.judge_results_t1.agrees_with_rationale} ({self.judge_results_t1.rationale_agreement_rate:.1%})"
                )
            if self.judge_results_t2 and self.judge_results_t2.total_disagreements > 0:
                print("  Threshold >= 2:")
                print(
                    f"    Disagreements reviewed: {self.judge_results_t2.total_disagreements}"
                )
                print(
                    f"    Judge agrees with Bing:      {self.judge_results_t2.agrees_with_bing} ({self.judge_results_t2.bing_agreement_rate:.1%})"
                )
                print(
                    f"    Judge agrees with Rationale: {self.judge_results_t2.agrees_with_rationale} ({self.judge_results_t2.rationale_agreement_rate:.1%})"
                )
        print(f"{'=' * 60}\n")

    def get_disagreements(self, min_diff: int = 2) -> list[RaterComparisonItem]:
        """Get items where raters disagree by at least min_diff points.

        Args:
            min_diff: Minimum score difference to consider a disagreement.

        Returns
        -------
            List of items with significant disagreements.

        """
        return [item for item in self.items if item.score_diff >= min_diff]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_comparisons": self.total_comparisons,
            "exact_agreements": self.exact_agreements,
            "agreement_rate": self.agreement_rate,
            "within_one_agreements": self.within_one_agreements,
            "within_one_rate": self.within_one_rate,
            "mean_score_diff": self.mean_score_diff,
            "bing_mean_score": self.bing_mean_score,
            "rationale_mean_score": self.rationale_mean_score,
            "score_distribution": self.score_distribution,
            "inter_rater_reliability": {
                "cohens_kappa": self.cohens_kappa,
                "weighted_kappa": self.weighted_kappa,
                "spearman_correlation": self.spearman_correlation,
                "pearson_correlation": self.pearson_correlation,
            },
            "confusion_matrix": self.confusion_matrix,
            "binary_agreement_threshold_1": self.binary_agreement_t1,
            "binary_agreement_threshold_2": self.binary_agreement_t2,
            "bing_stats": self.bing_stats.to_dict() if self.bing_stats else None,
            "rationale_stats": self.rationale_stats.to_dict()
            if self.rationale_stats
            else None,
            "judge_results_threshold_1": self.judge_results_t1.to_dict()
            if self.judge_results_t1
            else None,
            "judge_results_threshold_2": self.judge_results_t2.to_dict()
            if self.judge_results_t2
            else None,
            "items": [
                {
                    "query": item.query[:100] + "..."
                    if len(item.query) > 100
                    else item.query,
                    "text_unit_id": item.text_unit_id,
                    "bing_score": item.bing_score,
                    "rationale_score": item.rationale_score,
                    "agreement": item.agreement,
                    "score_diff": item.score_diff,
                    "rationale_reasoning": item.rationale_reasoning,
                }
                for item in self.items
            ],
        }

    def get_binary_disagreements(self, threshold: int) -> list[RaterComparisonItem]:
        """Get items where raters disagree on binary relevance at given threshold.

        Args:
            threshold: Score >= threshold is considered "relevant".

        Returns
        -------
            List of items where one rater says relevant and other says not relevant.

        """
        disagreements = []
        for item in self.items:
            bing_relevant = item.bing_score >= threshold
            rationale_relevant = item.rationale_score >= threshold
            if bing_relevant != rationale_relevant:
                disagreements.append(item)
        return disagreements


JUDGE_PROMPT_TEMPLATE = """You are an impartial judge evaluating whether a text passage is relevant to a query.

Query: {query}

Text passage:
{text}

Task: Determine if this text passage is relevant to answering the query.

A passage is RELEVANT if it contains information that would help answer the query, even partially.
A passage is NOT RELEVANT if it does not contain useful information for answering the query.

Respond with a JSON object:
{{"relevant": true/false, "reasoning": "brief explanation"}}"""


async def judge_binary_disagreements(
    comparison_result: RaterComparisonResult,
    threshold: int,
    llm_config: LLMConfig,
    max_judgments: int | None = None,
) -> JudgeResults:
    """Have an LLM judge review binary disagreements between raters.

    Args:
        comparison_result: The comparison result containing disagreements.
        threshold: Score >= threshold is considered "relevant".
        llm_config: LLM configuration for the judge.
        max_judgments: Maximum number of disagreements to judge (None = all).

    Returns
    -------
        JudgeResults with verdicts and agreement statistics.

    """
    from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object

    disagreements = comparison_result.get_binary_disagreements(threshold)

    if max_judgments and len(disagreements) > max_judgments:
        import random

        disagreements = random.sample(disagreements, max_judgments)

    if not disagreements:
        return JudgeResults(verdicts=[], threshold=threshold)

    log.info(
        "Judging %d binary disagreements at threshold %d", len(disagreements), threshold
    )

    llm_client = ModelFactory.create_chat_model(llm_config)

    verdicts: list[JudgeVerdict] = []

    from tqdm import tqdm

    for item in tqdm(
        disagreements, desc=f"Judge reviewing (t={threshold})", unit="item"
    ):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=item.query,
            text=item.text_unit_text,
        )

        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                **llm_config.call_args,
            )
            response_text = response.output.content if response.output else ""

            # Parse JSON response
            _, parsed = try_parse_json_object(response_text)
            if parsed:
                judge_relevant = bool(parsed.get("relevant", False))
                judge_reasoning = str(parsed.get("reasoning", ""))
            else:
                # Fallback: look for "relevant": true/false
                judge_relevant = (
                    '"relevant": true' in response_text.lower()
                    or '"relevant":true' in response_text.lower()
                )
                judge_reasoning = response_text

            bing_relevant = item.bing_score >= threshold
            rationale_relevant = item.rationale_score >= threshold

            verdicts.append(
                JudgeVerdict(
                    query=item.query,
                    text_unit_id=item.text_unit_id,
                    text_unit_text=item.text_unit_text,
                    bing_score=item.bing_score,
                    rationale_score=item.rationale_score,
                    threshold=threshold,
                    bing_says_relevant=bing_relevant,
                    rationale_says_relevant=rationale_relevant,
                    judge_says_relevant=judge_relevant,
                    judge_reasoning=judge_reasoning,
                )
            )
        except Exception as e:
            log.warning("Failed to judge item %s: %s", item.text_unit_id, e)

    return JudgeResults(verdicts=verdicts, threshold=threshold)


async def compare_raters(
    queries: list[str],
    text_units: list[TextUnit],
    llm_config: LLMConfig,
    cache_dir: Path | None = None,
) -> RaterComparisonResult:
    """Compare Bing and Rationale raters on the same queries and text units.

    Args:
        queries: List of queries to assess.
        text_units: List of text units to assess for each query.
        llm_config: LLM configuration for both raters.
        cache_dir: Optional cache directory for assessments.

    Returns
    -------
        RaterComparisonResult with aggregated comparison metrics.

    """
    # Create LLM client
    llm_client = ModelFactory.create_chat_model(llm_config)

    # Create both raters
    bing_cache = cache_dir / "bing" if cache_dir else None
    rationale_cache = cache_dir / "rationale" if cache_dir else None

    bing_rater = BingRelevanceRater(
        llm_client=llm_client,
        llm_config=llm_config,
        cache_dir=bing_cache,
        cache_enabled=cache_dir is not None,
    )

    rationale_rater = RationaleRelevanceRater(
        llm_client=llm_client,
        llm_config=llm_config,
        cache_dir=rationale_cache,
        cache_enabled=cache_dir is not None,
    )

    comparison_items: list[RaterComparisonItem] = []

    for query in queries:
        log.info("Comparing raters for query: %s...", query[:50])

        # Run both raters
        bing_response = await bing_rater.rate_relevance(query, text_units)
        rationale_response = await rationale_rater.rate_relevance(query, text_units)

        # Pair up results by text unit ID
        bing_by_id = {
            item.text_unit.id: item
            for item in bing_response.assessment
            if item.text_unit is not None
        }
        rationale_by_id = {
            item.text_unit.id: item
            for item in rationale_response.assessment
            if item.text_unit is not None
        }

        for text_unit in text_units:
            bing_item = bing_by_id.get(text_unit.id)
            rationale_item = rationale_by_id.get(text_unit.id)

            if bing_item and rationale_item:
                comparison_items.append(
                    RaterComparisonItem(
                        query=query,
                        text_unit_id=text_unit.id,
                        text_unit_text=text_unit.text[:200],
                        bing_score=bing_item.score,
                        rationale_score=rationale_item.score,
                        rationale_reasoning=rationale_item.reasoning,
                    )
                )

    log.info("Comparison complete. %d items compared.", len(comparison_items))
    return RaterComparisonResult(items=comparison_items)


def _select_topk_by_similarity(
    query_embedding: list[float],
    text_units: list[TextUnit],
    k: int,
) -> list[TextUnit]:
    """Select top-K text units by cosine similarity to query.

    Args:
        query_embedding: Query embedding vector.
        text_units: List of text units with embeddings.
        k: Number of top text units to select.

    Returns
    -------
        List of top-K text units sorted by similarity (highest first).

    """
    import numpy as np

    query_vec = np.array(query_embedding)
    query_norm = np.linalg.norm(query_vec)

    similarities = []
    for tu in text_units:
        if tu.text_embedding is None:
            similarities.append(-1.0)  # No embedding, lowest priority
        else:
            tu_vec = np.array(tu.text_embedding)
            tu_norm = np.linalg.norm(tu_vec)
            if query_norm > 0 and tu_norm > 0:
                sim = float(np.dot(query_vec, tu_vec) / (query_norm * tu_norm))
            else:
                sim = 0.0
            similarities.append(sim)

    # Get indices of top-K by similarity
    top_indices = np.argsort(similarities)[::-1][:k]
    return [text_units[i] for i in top_indices]


async def _compare_raters_with_topk_selection(
    queries: list[str],
    query_embeddings: list[list[float] | None],
    all_text_units: list[TextUnit],
    llm_config: LLMConfig,
    max_text_units: int,
    cache_dir: Path | None = None,
    embedding_config: LLMConfig | None = None,
) -> RaterComparisonResult:
    """Compare raters using top-K text units by similarity for each query.

    Args:
        queries: List of query texts.
        query_embeddings: List of query embeddings (may contain None).
        all_text_units: All text units to select from.
        llm_config: LLM configuration for raters.
        max_text_units: Number of top text units to select per query.
        cache_dir: Optional cache directory.
        embedding_config: LLM config for embedding queries without embeddings.

    Returns
    -------
        RaterComparisonResult with aggregated comparison metrics.

    """
    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
    from benchmark_qed.llm.factory import ModelFactory

    # Create LLM client and raters
    llm_client = ModelFactory.create_chat_model(llm_config)

    bing_cache = cache_dir / "bing" if cache_dir else None
    rationale_cache = cache_dir / "rationale" if cache_dir else None

    bing_rater = BingRelevanceRater(
        llm_client=llm_client,
        llm_config=llm_config,
        cache_dir=bing_cache,
        cache_enabled=cache_dir is not None,
    )

    rationale_rater = RationaleRelevanceRater(
        llm_client=llm_client,
        llm_config=llm_config,
        cache_dir=rationale_cache,
        cache_enabled=cache_dir is not None,
    )

    # Embed queries that don't have embeddings
    queries_needing_embedding = [
        (i, q)
        for i, (q, emb) in enumerate(zip(queries, query_embeddings, strict=True))
        if emb is None
    ]

    if queries_needing_embedding:
        if embedding_config is None:
            log.warning(
                "No embedding config provided and %d queries lack embeddings. "
                "Using llm_config for embeddings.",
                len(queries_needing_embedding),
            )
            embedding_config = llm_config

        embedding_model = ModelFactory.create_embedding_model(embedding_config)
        text_embedder = TextEmbedder(embedding_model)

        log.info(
            "Embedding %d queries without embeddings...", len(queries_needing_embedding)
        )
        texts_to_embed = [q for _, q in queries_needing_embedding]
        embeddings = await embedding_model.embed(texts_to_embed)

        # Update query_embeddings list
        query_embeddings = list(query_embeddings)  # Make mutable copy
        for (idx, _), emb in zip(queries_needing_embedding, embeddings, strict=True):
            query_embeddings[idx] = emb

    comparison_items: list[RaterComparisonItem] = []

    # Use tqdm for progress tracking
    import time

    from tqdm import tqdm

    query_pairs = [
        (q, emb)
        for q, emb in zip(queries, query_embeddings, strict=True)
        if emb is not None
    ]

    # Track timing and token usage
    bing_stats = RaterStats(
        num_queries=len(query_pairs), num_text_units=len(query_pairs) * max_text_units
    )
    rationale_stats = RaterStats(
        num_queries=len(query_pairs), num_text_units=len(query_pairs) * max_text_units
    )

    # Get initial token counts
    bing_initial_usage = bing_rater.llm_client.get_usage()
    rationale_initial_usage = rationale_rater.llm_client.get_usage()

    for query, query_emb in tqdm(
        query_pairs,
        desc="Comparing raters",
        unit="query",
    ):
        # Select top-K text units by similarity
        selected_text_units = _select_topk_by_similarity(
            query_emb, all_text_units, max_text_units
        )

        # Run Bing rater with timing
        bing_start = time.perf_counter()
        bing_response = await bing_rater.rate_relevance(query, selected_text_units)
        bing_stats.total_time_seconds += time.perf_counter() - bing_start

        # Run Rationale rater with timing
        rationale_start = time.perf_counter()
        rationale_response = await rationale_rater.rate_relevance(
            query, selected_text_units
        )
        rationale_stats.total_time_seconds += time.perf_counter() - rationale_start

        # Pair up results by text unit ID
        bing_by_id = {
            item.text_unit.id: item
            for item in bing_response.assessment
            if item.text_unit is not None
        }
        rationale_by_id = {
            item.text_unit.id: item
            for item in rationale_response.assessment
            if item.text_unit is not None
        }

        for text_unit in selected_text_units:
            bing_item = bing_by_id.get(text_unit.id)
            rationale_item = rationale_by_id.get(text_unit.id)

            if bing_item and rationale_item:
                comparison_items.append(
                    RaterComparisonItem(
                        query=query,
                        text_unit_id=text_unit.id,
                        text_unit_text=text_unit.text[:200],
                        bing_score=bing_item.score,
                        rationale_score=rationale_item.score,
                        rationale_reasoning=rationale_item.reasoning,
                    )
                )

    # Calculate token usage (final - initial)
    bing_final_usage = bing_rater.llm_client.get_usage()
    rationale_final_usage = rationale_rater.llm_client.get_usage()

    bing_stats.prompt_tokens = bing_final_usage.get(
        "prompt_tokens", 0
    ) - bing_initial_usage.get("prompt_tokens", 0)
    bing_stats.completion_tokens = bing_final_usage.get(
        "completion_tokens", 0
    ) - bing_initial_usage.get("completion_tokens", 0)
    rationale_stats.prompt_tokens = rationale_final_usage.get(
        "prompt_tokens", 0
    ) - rationale_initial_usage.get("prompt_tokens", 0)
    rationale_stats.completion_tokens = rationale_final_usage.get(
        "completion_tokens", 0
    ) - rationale_initial_usage.get("completion_tokens", 0)

    log.info("Comparison complete. %d items compared.", len(comparison_items))

    result = RaterComparisonResult(items=comparison_items)
    result.bing_stats = bing_stats
    result.rationale_stats = rationale_stats

    return result


async def compare_raters_from_files(
    questions_path: Path,
    text_units_path: Path,
    llm_config: LLMConfig,
    max_questions: int | None = None,
    max_text_units: int | None = None,
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    random_seed: int | None = 42,
    use_similarity_selection: bool = True,
    embedding_config: LLMConfig | None = None,
    run_judge: bool = False,
    judge_config: LLMConfig | None = None,
    max_judgments: int | None = None,
) -> RaterComparisonResult:
    """Compare raters using questions and text units from files.

    Args:
        questions_path: Path to questions JSON file.
        text_units_path: Path to text units parquet file.
        llm_config: LLM configuration for both raters.
        max_questions: Maximum number of questions to process.
        max_text_units: Maximum number of text units per question.
        cache_dir: Optional cache directory for assessments.
        output_path: Optional path to save comparison results as JSON.
        random_seed: Random seed for reproducible sampling. Set to None for no seed.
        use_similarity_selection: If True, select top-K text units by similarity
            to each query. If False, use random sampling.
        embedding_config: LLM config for embeddings (required if use_similarity_selection
            is True and questions don't have embeddings).
        run_judge: If True, run an LLM judge to review binary disagreements.
        judge_config: LLM config for the judge. If None, uses llm_config.
        max_judgments: Maximum number of disagreements to judge per threshold.

    Returns
    -------
        RaterComparisonResult with aggregated comparison metrics.

    """
    import random

    import numpy as np
    import pandas as pd

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Load questions
    with open(questions_path) as f:
        questions_data = json.load(f)

    # Sample questions if needed
    if max_questions and max_questions < len(questions_data):
        questions_data = random.sample(questions_data, max_questions)

    queries = [q["text"] for q in questions_data]
    query_embeddings = [q.get("embedding") for q in questions_data]

    # Load text units
    df = pd.read_parquet(text_units_path)
    all_text_units = [
        TextUnit(
            id=str(row.get("id", idx)),
            short_id=str(row.get("short_id", idx)),
            text=row["text"],
            text_embedding=row.get("text_embedding"),
        )
        for idx, row in df.iterrows()
    ]

    if use_similarity_selection and max_text_units:
        # Select top-K text units by similarity for each query
        log.info(
            "Selecting top %d text units by similarity for each of %d questions",
            max_text_units,
            len(queries),
        )
        result = await _compare_raters_with_topk_selection(
            queries=queries,
            query_embeddings=query_embeddings,
            all_text_units=all_text_units,
            llm_config=llm_config,
            max_text_units=max_text_units,
            cache_dir=cache_dir,
            embedding_config=embedding_config,
        )
    else:
        # Random sampling (original behavior)
        if max_text_units and max_text_units < len(all_text_units):
            all_text_units = random.sample(all_text_units, max_text_units)

        log.info(
            "Comparing raters on %d questions x %d text units (random)",
            len(queries),
            len(all_text_units),
        )

        result = await compare_raters(
            queries=queries,
            text_units=all_text_units,
            llm_config=llm_config,
            cache_dir=cache_dir,
        )

    # Run judge if requested
    if run_judge:
        judge_llm_config = judge_config or llm_config
        log.info("Running LLM judge to review binary disagreements...")

        result.judge_results_t1 = await judge_binary_disagreements(
            comparison_result=result,
            threshold=1,
            llm_config=judge_llm_config,
            max_judgments=max_judgments,
        )

        result.judge_results_t2 = await judge_binary_disagreements(
            comparison_result=result,
            threshold=2,
            llm_config=judge_llm_config,
            max_judgments=max_judgments,
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        log.info("Saved comparison results to %s", output_path)

    return result
