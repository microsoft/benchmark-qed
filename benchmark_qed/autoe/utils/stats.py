# Copyright (c) 2025 Microsoft Corporation.
"""Statistical significance testing utilities for comparing multiple independent samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Display names for p-value correction methods
CORRECTION_DISPLAY_NAMES: dict[str, str] = {
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "fdr_bh": "Benjamini-Hochberg FDR",
}


@dataclass
class NormalityResult:
    """Result of normality test for a single group."""

    group_name: str
    statistic: float
    p_value: float
    is_normal: bool


@dataclass
class OmnibusTestResult:
    """Result of an omnibus test for comparing multiple groups."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    is_normal: bool
    normality_results: list[NormalityResult]

    def summary(self) -> str:
        """Return a summary string of the omnibus test result."""
        sig_str = "significant" if self.is_significant else "not significant"
        normal_str = "normal" if self.is_normal else "non-normal"
        return (
            f"{self.test_name} ({normal_str} data): "
            f"statistic={self.statistic:.4f}, p={self.p_value:.4f} ({sig_str})"
        )


@dataclass
class PairwiseComparison:
    """Result of a single pairwise comparison."""

    group1: str
    group2: str
    statistic: float
    p_value_raw: float
    p_value_corrected: float
    is_significant: bool


@dataclass
class PostHocResult:
    """Result of post-hoc pairwise comparisons."""

    test_name: str
    correction_method: str
    alpha: float
    comparisons: list[PairwiseComparison] = field(default_factory=list)

    def summary(self) -> str:
        """Return a summary string of significant differences."""
        sig = [c for c in self.comparisons if c.is_significant]
        if not sig:
            return "No significant pairwise differences found."
        lines = [
            f"Significant differences ({self.test_name}, {self.correction_method} correction):"
        ]
        lines.extend(
            f"  {c.group1} vs {c.group2}: p={c.p_value_corrected:.4f}" for c in sig
        )
        return "\n".join(lines)

    def get_significant_pairs(self) -> list[tuple[str, str]]:
        """Return list of significant group pairs."""
        return [(c.group1, c.group2) for c in self.comparisons if c.is_significant]


@dataclass
class GroupComparisonResult:
    """Complete result of comparing multiple groups."""

    omnibus: OmnibusTestResult
    posthoc: PostHocResult | None

    def summary(self) -> str:
        """Return a complete summary of the comparison."""
        lines = [self.omnibus.summary()]
        if self.posthoc:
            lines.append(self.posthoc.summary())
        elif self.omnibus.is_significant:
            lines.append("Post-hoc tests not performed.")
        else:
            lines.append("No post-hoc tests needed (omnibus not significant).")
        return "\n".join(lines)


def check_normality(
    data: Sequence[float],
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """
    Test if data is normally distributed.

    Uses Shapiro-Wilk test for n <= 5000, Kolmogorov-Smirnov test for larger samples.

    Args:
        data: Sample data to test
        alpha: Significance level for normality test

    Returns
    -------
        Tuple of (statistic, p_value, is_normal)
    """
    data_array = np.asarray(data)
    if len(data_array) < 3:
        # Not enough data for normality test, assume non-normal
        return 0.0, 0.0, False
    if len(data_array) > 5000:
        # Shapiro-Wilk is limited to 5000 samples, use Kolmogorov-Smirnov
        # Standardize data to test against standard normal
        standardized = (data_array - np.mean(data_array)) / np.std(data_array, ddof=1)
        stat, p_value = stats.kstest(standardized, "norm")
    else:
        stat, p_value = stats.shapiro(data_array)
    return float(stat), float(p_value), p_value > alpha


def run_omnibus_test(
    groups: Mapping[str, Sequence[float]],
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    test: str = "auto",
) -> OmnibusTestResult:
    """
    Run an omnibus test to check if there are significant differences between groups.

    Args:
        groups: Dictionary mapping group names to their data samples
        alpha: Significance level for the omnibus test
        normality_alpha: Significance level for normality testing
        test: Which test to use:
            - "auto": ANOVA if all groups normal, Kruskal-Wallis otherwise
            - "anova": One-way ANOVA (assumes normality)
            - "kruskal": Kruskal-Wallis H-test (non-parametric)

    Returns
    -------
        OmnibusTestResult with test statistics and significance
    """
    if len(groups) < 2:
        msg = "At least 2 groups are required for comparison"
        raise ValueError(msg)

    group_names = list(groups.keys())
    group_data = [np.asarray(groups[name]) for name in group_names]

    # Test normality for each group
    normality_results: list[NormalityResult] = []
    all_normal = True
    for name, data in groups.items():
        stat, p_val, is_normal = check_normality(data, normality_alpha)
        normality_results.append(
            NormalityResult(
                group_name=name,
                statistic=stat,
                p_value=p_val,
                is_normal=is_normal,
            )
        )
        if not is_normal:
            all_normal = False

    # Choose and run appropriate test
    if test == "anova" or (test == "auto" and all_normal):
        stat, p_value = stats.f_oneway(*group_data)
        test_name = "One-way ANOVA"
        is_normal = True
    else:  # kruskal or (auto and not all_normal)
        stat, p_value = stats.kruskal(*group_data)
        test_name = "Kruskal-Wallis H-test"
        is_normal = False

    return OmnibusTestResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        alpha=alpha,
        is_normal=is_normal,
        normality_results=normality_results,
    )


def run_posthoc_pairwise(
    groups: Mapping[str, Sequence[float]],
    is_normal: bool | None = None,
    alpha: float = 0.05,
    correction: str = "holm",
) -> PostHocResult:
    """
    Run pairwise post-hoc tests between all groups with multiple comparison correction.

    For normal data: Tukey HSD (built-in FWER control) or t-tests with correction
    For non-normal data: Dunn's test with correction

    Args:
        groups: Dictionary mapping group names to their data samples
        is_normal: Whether data is normally distributed. If None, will test automatically.
        alpha: Significance level
        correction: Correction method - "holm" (recommended), "bonferroni", or "fdr_bh"

    Returns
    -------
        PostHocResult with all pairwise comparisons
    """
    if len(groups) < 2:
        msg = "At least 2 groups are required for comparison"
        raise ValueError(msg)

    group_names = list(groups.keys())

    # Determine normality if not specified
    if is_normal is None:
        all_normal = True
        for data in groups.values():
            _, _, normal = check_normality(data)
            if not normal:
                all_normal = False
                break
        is_normal = all_normal

    if is_normal:
        return _run_tukey_posthoc(groups, group_names, alpha, correction)
    return _run_dunn_posthoc(groups, group_names, alpha, correction)


def _run_tukey_posthoc(
    groups: Mapping[str, Sequence[float]],
    group_names: list[str],
    alpha: float,
    correction: str,  # noqa: ARG001 - kept for API consistency
) -> PostHocResult:
    """Run Tukey HSD test for normal data.

    Tukey HSD computes p-values that account for multiple pairwise comparisons,
    so no additional correction is needed.
    """
    group_data = [np.asarray(groups[name]) for name in group_names]

    # scipy.stats.tukey_hsd available in scipy >= 1.8
    result = stats.tukey_hsd(*group_data)

    comparisons: list[PairwiseComparison] = []
    for i, name1 in enumerate(group_names):
        for j, name2 in enumerate(group_names[i + 1 :], start=i + 1):
            # tukey_hsd returns a result object with pvalue attribute as a matrix
            p_val: float = result.pvalue[i, j]  # type: ignore[union-attr]
            stat: float = result.statistic[i, j]  # type: ignore[union-attr]
            comparisons.append(
                PairwiseComparison(
                    group1=name1,
                    group2=name2,
                    statistic=stat,
                    p_value_raw=p_val,
                    p_value_corrected=p_val,  # Tukey accounts for multiple comparisons
                    is_significant=p_val < alpha,
                )
            )

    return PostHocResult(
        test_name="Tukey HSD",
        correction_method="Studentized Range",
        alpha=alpha,
        comparisons=comparisons,
    )


def _run_dunn_posthoc(
    groups: Mapping[str, Sequence[float]],
    group_names: list[str],
    alpha: float,
    correction: str,
) -> PostHocResult:
    """Run Dunn's test with correction for non-normal data."""
    try:
        import scikit_posthocs as sp

        # Prepare data for scikit-posthocs
        group_data = [np.asarray(groups[name]) for name in group_names]

        # Run Dunn's test - returns a DataFrame of corrected p-values
        p_matrix = sp.posthoc_dunn(group_data, p_adjust=correction)

        # Convert to list of comparisons
        comparisons: list[PairwiseComparison] = []
        for i, name1 in enumerate(group_names):
            for j, name2 in enumerate(group_names[i + 1 :], start=i + 1):
                p_corrected = float(p_matrix.iloc[i, j])  # type: ignore[arg-type]
                comparisons.append(
                    PairwiseComparison(
                        group1=name1,
                        group2=name2,
                        statistic=np.nan,  # Dunn's doesn't provide per-pair statistic
                        p_value_raw=np.nan,  # scikit-posthocs only returns corrected
                        p_value_corrected=p_corrected,
                        is_significant=p_corrected < alpha,
                    )
                )

        return PostHocResult(
            test_name="Dunn's test",
            correction_method=CORRECTION_DISPLAY_NAMES.get(correction, correction),
            alpha=alpha,
            comparisons=comparisons,
        )

    except ImportError:
        # Fallback to Mann-Whitney U if scikit-posthocs not available
        return _run_mannwhitney_posthoc(groups, group_names, alpha, correction)


def _run_mannwhitney_posthoc(
    groups: Mapping[str, Sequence[float]],
    group_names: list[str],
    alpha: float,
    correction: str,
) -> PostHocResult:
    """Fallback: Run Mann-Whitney U tests with correction for non-normal data."""
    comparisons_raw: list[tuple[str, str, float, float]] = []

    for i, name1 in enumerate(group_names):
        for name2 in group_names[i + 1 :]:
            data1 = np.asarray(groups[name1])
            data2 = np.asarray(groups[name2])
            result = stats.mannwhitneyu(data1, data2, alternative="two-sided")
            stat: float = result.statistic  # type: ignore[union-attr]
            p_val: float = result.pvalue  # type: ignore[union-attr]
            comparisons_raw.append((name1, name2, stat, p_val))

    test_name = "Mann-Whitney U"
    return _apply_correction_and_build_result(
        comparisons_raw, test_name, alpha, correction
    )


def _apply_correction_and_build_result(
    comparisons_raw: list[tuple[str, str, float, float]],
    test_name: str,
    alpha: float,
    correction: str,
) -> PostHocResult:
    """Apply multiple comparison correction and build PostHocResult."""
    p_values = [c[3] for c in comparisons_raw]

    _, corrected_pvals, _, _ = multipletests(p_values, alpha=alpha, method=correction)

    comparisons = [
        PairwiseComparison(
            group1=raw[0],
            group2=raw[1],
            statistic=raw[2],
            p_value_raw=raw[3],
            p_value_corrected=float(corrected_pvals[i]),
            is_significant=corrected_pvals[i] < alpha,
        )
        for i, raw in enumerate(comparisons_raw)
    ]

    return PostHocResult(
        test_name=test_name,
        correction_method=CORRECTION_DISPLAY_NAMES.get(correction, correction),
        alpha=alpha,
        comparisons=comparisons,
    )


def compare_groups(
    groups: Mapping[str, Sequence[float]],
    alpha: float = 0.05,
    correction: str = "holm",
    run_posthoc_if_not_significant: bool = False,
) -> GroupComparisonResult:
    """
    Complete statistical comparison of multiple independent groups.

    First runs an omnibus test (ANOVA or Kruskal-Wallis based on normality),
    then if significant, runs pairwise post-hoc tests with the specified correction.

    Args:
        groups: Dictionary mapping group names to their data samples
        alpha: Significance level
        correction: Correction method for post-hoc tests ("holm", "bonferroni", "fdr_bh")
        run_posthoc_if_not_significant: Whether to run post-hoc even if omnibus is not significant

    Returns
    -------
        GroupComparisonResult with omnibus and optional post-hoc results

    Example:
        >>> groups = {
        ...     "method_a": [0.82, 0.85, 0.79, 0.88, 0.84],
        ...     "method_b": [0.75, 0.72, 0.78, 0.71, 0.74],
        ...     "method_c": [0.80, 0.83, 0.81, 0.79, 0.82],
        ... }
        >>> result = compare_groups(groups)
        >>> print(result.summary())
    """
    omnibus = run_omnibus_test(groups, alpha=alpha)

    posthoc = None
    if omnibus.is_significant or run_posthoc_if_not_significant:
        posthoc = run_posthoc_pairwise(
            groups,
            is_normal=omnibus.is_normal,
            alpha=alpha,
            correction=correction,
        )

    return GroupComparisonResult(omnibus=omnibus, posthoc=posthoc)
