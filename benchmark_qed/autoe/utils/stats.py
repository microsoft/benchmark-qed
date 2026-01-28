# Copyright (c) 2025 Microsoft Corporation.
"""Statistical significance testing utilities for comparing multiple samples.

Supports both independent samples and repeated measures (paired) designs:
- Independent samples: Different subjects in each group (ANOVA, Kruskal-Wallis)
- Repeated measures: Same subjects measured under multiple conditions (Friedman, RM-ANOVA)

For RAG evaluation where the same questions are answered by different methods,
use repeated measures design (paired=True) for more accurate statistical testing.
"""

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


def _run_repeated_measures_anova(group_data: list[np.ndarray]) -> tuple[float, float]:
    """
    Run a simple repeated measures ANOVA (one-way within-subjects ANOVA).

    This is a simplified implementation that computes the F-statistic for
    repeated measures design. For more complex designs, consider using
    statsmodels or pingouin.

    Args:
        group_data: List of arrays, one per condition/group. All must have same length.

    Returns
    -------
        Tuple of (F-statistic, p-value)
    """
    k = len(group_data)  # number of conditions
    n = len(group_data[0])  # number of subjects

    # Stack data: rows = subjects, columns = conditions
    data_matrix = np.column_stack(group_data)

    # Grand mean
    grand_mean = np.mean(data_matrix)

    # Condition means (across subjects)
    condition_means = np.mean(data_matrix, axis=0)

    # Subject means (across conditions)
    subject_means = np.mean(data_matrix, axis=1)

    # Sum of squares
    # SS_total = sum of (each value - grand mean)^2
    ss_total = np.sum((data_matrix - grand_mean) ** 2)

    # SS_between (conditions) = n * sum of (condition mean - grand mean)^2
    ss_between = n * np.sum((condition_means - grand_mean) ** 2)

    # SS_subjects = k * sum of (subject mean - grand mean)^2
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)

    # Error sum of squares: total minus between-conditions minus between-subjects
    ss_error = ss_total - ss_between - ss_subjects

    # Degrees of freedom
    df_between = k - 1
    df_error = (n - 1) * (k - 1)

    # Mean squares
    ms_between = ss_between / df_between
    ms_error = ss_error / df_error if df_error > 0 else 1e-10

    # F-statistic
    f_stat = ms_between / ms_error

    # p-value from F-distribution
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_error)

    return float(f_stat), float(p_value)


def run_omnibus_test(
    groups: Mapping[str, Sequence[float]],
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    test: str = "auto",
    paired: bool = False,
) -> OmnibusTestResult:
    """
    Run an omnibus test to check if there are significant differences between groups.

    Args:
        groups: Dictionary mapping group names to their data samples.
                For paired=True, all groups must have the same length and be aligned
                (i.e., index i in each group corresponds to the same subject/question).
        alpha: Significance level for the omnibus test
        normality_alpha: Significance level for normality testing
        test: Which test to use:
            - "auto": Choose based on normality and paired setting
            - "anova": One-way ANOVA (independent) or Repeated Measures ANOVA (paired)
            - "kruskal": Kruskal-Wallis (independent) or Friedman (paired)
        paired: Whether this is a repeated measures design (same subjects across groups).
                Set to True when comparing RAG methods on the same set of questions.

    Returns
    -------
        OmnibusTestResult with test statistics and significance
    """
    if len(groups) < 2:
        msg = "At least 2 groups are required for comparison"
        raise ValueError(msg)

    group_names = list(groups.keys())
    group_data = [np.asarray(groups[name]) for name in group_names]

    # For paired design, validate that all groups have the same length
    if paired:
        lengths = [len(data) for data in group_data]
        if len(set(lengths)) > 1:
            msg = (
                f"For paired (repeated measures) design, all groups must have the same "
                f"number of observations. Got lengths: {dict(zip(group_names, lengths, strict=False))}"
            )
            raise ValueError(msg)

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

    # Choose and run appropriate test based on paired setting and normality
    use_parametric = test == "anova" or (test == "auto" and all_normal)

    if paired:
        # Repeated measures design
        if use_parametric:
            # For repeated measures ANOVA, we need to use a different approach
            # scipy doesn't have built-in RM-ANOVA, so we use Friedman as fallback
            # or compute F-statistic manually
            stat, p_value = _run_repeated_measures_anova(group_data)
            test_name = "Repeated Measures ANOVA"
            is_normal = True
        else:
            # Friedman test - non-parametric repeated measures
            stat, p_value = stats.friedmanchisquare(*group_data)
            test_name = "Friedman test"
            is_normal = False
    else:
        # Independent samples design
        if use_parametric:
            stat, p_value = stats.f_oneway(*group_data)
            test_name = "One-way ANOVA"
            is_normal = True
        else:
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
    paired: bool = False,
) -> PostHocResult:
    """
    Run pairwise post-hoc tests between all groups with multiple comparison correction.

    For independent samples (paired=False):
        - Normal data: Tukey HSD (built-in FWER control) or t-tests with correction
        - Non-normal data: Dunn's test or Mann-Whitney U with correction

    For repeated measures (paired=True):
        - Normal data: Paired t-tests with correction
        - Non-normal data: Wilcoxon signed-rank tests with correction

    Args:
        groups: Dictionary mapping group names to their data samples.
                For paired=True, all groups must have the same length and be aligned.
        is_normal: Whether data is normally distributed. If None, will test automatically.
        alpha: Significance level
        correction: Correction method - "holm" (recommended), "bonferroni", or "fdr_bh"
        paired: Whether this is a repeated measures design (same subjects across groups).
                Set to True when comparing RAG methods on the same set of questions.

    Returns
    -------
        PostHocResult with all pairwise comparisons
    """
    if len(groups) < 2:
        msg = "At least 2 groups are required for comparison"
        raise ValueError(msg)

    group_names = list(groups.keys())

    # For paired design, validate that all groups have the same length
    if paired:
        lengths = [len(groups[name]) for name in group_names]
        if len(set(lengths)) > 1:
            msg = (
                f"For paired (repeated measures) design, all groups must have the same "
                f"number of observations. Got lengths: {dict(zip(group_names, lengths, strict=False))}"
            )
            raise ValueError(msg)

    # Determine normality if not specified
    if is_normal is None:
        all_normal = True
        for data in groups.values():
            _, _, normal = check_normality(data)
            if not normal:
                all_normal = False
                break
        is_normal = all_normal

    if paired:
        # Repeated measures design - use paired tests
        if is_normal:
            return _run_paired_ttest_posthoc(groups, group_names, alpha, correction)
        return _run_wilcoxon_posthoc(groups, group_names, alpha, correction)
    # Independent samples design
    if is_normal:
        return _run_tukey_posthoc(groups, group_names, alpha, correction)
    return _run_dunn_posthoc(groups, group_names, alpha, correction)


def _run_tukey_posthoc(
    groups: Mapping[str, Sequence[float]],
    group_names: list[str],
    alpha: float,
    correction: str,  # noqa: ARG001 - kept for API consistency
) -> PostHocResult:
    """Run Tukey HSD test for normal data (independent samples).

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


def _run_paired_ttest_posthoc(
    groups: Mapping[str, Sequence[float]],
    group_names: list[str],
    alpha: float,
    correction: str,
) -> PostHocResult:
    """Run paired t-tests with correction for normal repeated measures data."""
    comparisons_raw: list[tuple[str, str, float, float]] = []

    for i, name1 in enumerate(group_names):
        for name2 in group_names[i + 1 :]:
            data1 = np.asarray(groups[name1])
            data2 = np.asarray(groups[name2])
            result = stats.ttest_rel(data1, data2)
            stat: float = result.statistic  # type: ignore[union-attr]
            p_val: float = result.pvalue  # type: ignore[union-attr]
            comparisons_raw.append((name1, name2, stat, p_val))

    test_name = "Paired t-test"
    return _apply_correction_and_build_result(
        comparisons_raw, test_name, alpha, correction
    )


def _run_wilcoxon_posthoc(
    groups: Mapping[str, Sequence[float]],
    group_names: list[str],
    alpha: float,
    correction: str,
) -> PostHocResult:
    """Run Wilcoxon signed-rank tests with correction for non-normal repeated measures data."""
    comparisons_raw: list[tuple[str, str, float, float]] = []

    for i, name1 in enumerate(group_names):
        for name2 in group_names[i + 1 :]:
            data1 = np.asarray(groups[name1])
            data2 = np.asarray(groups[name2])
            # Use 'zsplit' method to handle ties and zeros
            try:
                result = stats.wilcoxon(data1, data2, alternative="two-sided")
                stat: float = result.statistic  # type: ignore[union-attr]
                p_val: float = result.pvalue  # type: ignore[union-attr]
            except ValueError:
                # If all differences are zero, set p-value to 1 (no difference)
                stat = 0.0
                p_val = 1.0
            comparisons_raw.append((name1, name2, stat, p_val))

    test_name = "Wilcoxon signed-rank"
    return _apply_correction_and_build_result(
        comparisons_raw, test_name, alpha, correction
    )


def _run_dunn_posthoc(
    groups: Mapping[str, Sequence[float]],
    group_names: list[str],
    alpha: float,
    correction: str,
) -> PostHocResult:
    """Run Dunn's test with correction for non-normal data (independent samples)."""
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
    paired: bool = False,
) -> GroupComparisonResult:
    """
    Complete statistical comparison of multiple groups.

    Supports both independent samples and repeated measures (paired) designs.

    For independent samples (paired=False):
        - Omnibus: ANOVA or Kruskal-Wallis based on normality
        - Post-hoc: Tukey HSD or Dunn's test

    For repeated measures (paired=True):
        - Omnibus: Repeated Measures ANOVA or Friedman test based on normality
        - Post-hoc: Paired t-tests or Wilcoxon signed-rank tests with correction

    Use paired=True when comparing RAG methods on the same set of questions,
    as this is a repeated measures design (same questions measured under
    multiple conditions).

    Args:
        groups: Dictionary mapping group names to their data samples.
                For paired=True, all groups must have the same length and be aligned
                (i.e., index i in each group corresponds to the same subject/question).
        alpha: Significance level
        correction: Correction method for post-hoc tests ("holm", "bonferroni", "fdr_bh")
        run_posthoc_if_not_significant: Whether to run post-hoc even if omnibus is not significant
        paired: Whether this is a repeated measures design. Set to True when comparing
                RAG methods on the same set of questions.

    Returns
    -------
        GroupComparisonResult with omnibus and optional post-hoc results

    Example:
        >>> # Independent samples (different questions per method)
        >>> groups = {
        ...     "method_a": [0.82, 0.85, 0.79, 0.88, 0.84],
        ...     "method_b": [0.75, 0.72, 0.78, 0.71, 0.74],
        ...     "method_c": [0.80, 0.83, 0.81, 0.79, 0.82],
        ... }
        >>> result = compare_groups(groups)

        >>> # Repeated measures (same questions, different methods)
        >>> groups = {
        ...     "method_a": [0.82, 0.85, 0.79, 0.88, 0.84],  # scores for questions 1-5
        ...     "method_b": [0.75, 0.72, 0.78, 0.71, 0.74],  # scores for questions 1-5
        ...     "method_c": [0.80, 0.83, 0.81, 0.79, 0.82],  # scores for questions 1-5
        ... }
        >>> result = compare_groups(groups, paired=True)
        >>> print(result.summary())
    """
    omnibus = run_omnibus_test(groups, alpha=alpha, paired=paired)

    posthoc = None
    if omnibus.is_significant or run_posthoc_if_not_significant:
        posthoc = run_posthoc_pairwise(
            groups,
            is_normal=omnibus.is_normal,
            alpha=alpha,
            correction=correction,
            paired=paired,
        )

    return GroupComparisonResult(omnibus=omnibus, posthoc=posthoc)
