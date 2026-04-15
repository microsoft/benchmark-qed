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


def _compare_two_groups_paired(
    groups: Mapping[str, Sequence[float]],
    alpha: float = 0.05,
) -> GroupComparisonResult:
    """
    Compare exactly 2 groups with a paired design.

    For 2 groups, omnibus tests like Friedman are not applicable (they require
    3+ conditions). Instead, we directly perform a paired comparison test.

    Args:
        groups: Dictionary mapping exactly 2 group names to their data samples.
                Groups must have the same length and be aligned by subject/question.
        alpha: Significance level.

    Returns
    -------
        GroupComparisonResult with the pairwise comparison as both the "omnibus"
        summary and a single post-hoc comparison.
    """
    if len(groups) != 2:
        msg = "_compare_two_groups_paired requires exactly 2 groups"
        raise ValueError(msg)

    group_names = list(groups.keys())
    group_a, group_b = group_names[0], group_names[1]
    data_a = np.array(groups[group_a])
    data_b = np.array(groups[group_b])

    if len(data_a) != len(data_b):
        msg = "Paired comparison requires groups of equal length"
        raise ValueError(msg)

    # Compute differences and check normality of the differences
    differences = data_a - data_b
    stat_normality, p_normality, is_normal = check_normality(
        differences.tolist(), alpha=alpha
    )

    # Create a normality result for the differences
    normality_results = [
        NormalityResult(
            group_name="differences",
            statistic=float(stat_normality),
            p_value=float(p_normality),
            is_normal=is_normal,
        )
    ]

    # Choose test based on normality of differences
    if is_normal:
        result = stats.ttest_rel(data_a, data_b)
        stat, p_value = float(result.statistic), float(result.pvalue)
        test_name = "Paired t-test"
    else:
        result = stats.wilcoxon(data_a, data_b)
        stat, p_value = float(result.statistic), float(result.pvalue)  # type: ignore[union-attr]
        test_name = "Wilcoxon signed-rank test"

    is_significant = p_value < alpha

    # Create an "omnibus-like" result for consistency
    omnibus = OmnibusTestResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_value),
        is_significant=is_significant,
        is_normal=is_normal,
        alpha=alpha,
        normality_results=normality_results,
    )

    # Also create a single pairwise comparison for the post-hoc result
    comparison = PairwiseComparison(
        group1=group_a,
        group2=group_b,
        statistic=float(stat),
        p_value_raw=float(p_value),
        p_value_corrected=float(p_value),  # No correction needed for single comparison
        is_significant=is_significant,
    )

    posthoc = PostHocResult(
        test_name=test_name,
        correction_method="none (single comparison)",
        alpha=alpha,
        comparisons=[comparison],
    )

    return GroupComparisonResult(omnibus=omnibus, posthoc=posthoc)


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
        - 2 groups: Direct paired test (no omnibus needed)
        - 3+ groups: Omnibus (RM-ANOVA/Friedman) then paired post-hoc tests

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
        ...     "method_a": [
        ...         0.82,
        ...         0.85,
        ...         0.79,
        ...         0.88,
        ...         0.84,
        ...     ],
        ...     "method_b": [
        ...         0.75,
        ...         0.72,
        ...         0.78,
        ...         0.71,
        ...         0.74,
        ...     ],
        ...     "method_c": [
        ...         0.80,
        ...         0.83,
        ...         0.81,
        ...         0.79,
        ...         0.82,
        ...     ],
        ... }
        >>> result = (
        ...     compare_groups(
        ...         groups
        ...     )
        ... )

        >>> # Repeated measures (same questions, different methods)
        >>> groups = {
        ...     "method_a": [
        ...         0.82,
        ...         0.85,
        ...         0.79,
        ...         0.88,
        ...         0.84,
        ...     ],  # scores for questions 1-5
        ...     "method_b": [
        ...         0.75,
        ...         0.72,
        ...         0.78,
        ...         0.71,
        ...         0.74,
        ...     ],  # scores for questions 1-5
        ...     "method_c": [
        ...         0.80,
        ...         0.83,
        ...         0.81,
        ...         0.79,
        ...         0.82,
        ...     ],  # scores for questions 1-5
        ... }
        >>> result = (
        ...     compare_groups(
        ...         groups,
        ...         paired=True,
        ...     )
        ... )
        >>> print(
        ...     result.summary()
        ... )
    """
    # For paired design with exactly 2 groups, skip omnibus and go directly to pairwise
    # (Friedman test requires 3+ groups)
    if paired and len(groups) == 2:
        return _compare_two_groups_paired(groups, alpha=alpha)

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


def _compute_f_statistic(
    values: np.ndarray,
    labels: np.ndarray,
    group_names: list[str],
) -> float:
    """Compute one-way F-statistic for K groups.

    Args:
        values: Array of observation values.
        labels: Array of group labels (same length as values).
        group_names: Unique group names.

    Returns
    -------
        F-statistic (between-group variance / within-group variance).
        Returns 0.0 if within-group variance is zero.
    """
    grand_mean = np.mean(values)
    ss_between = 0.0
    ss_within = 0.0
    for name in group_names:
        mask = labels == name
        group_vals = values[mask]
        n_g = len(group_vals)
        group_mean = np.mean(group_vals)
        ss_between += n_g * (group_mean - grand_mean) ** 2
        ss_within += np.sum((group_vals - group_mean) ** 2)

    k = len(group_names)
    n = len(values)
    df_between = k - 1
    df_within = n - k

    if df_within <= 0 or ss_within == 0:
        return 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return float(ms_between / ms_within)


def run_clustered_permutation_test(
    groups: Mapping[str, Sequence[float]],
    cluster_ids: Mapping[str, Sequence[str]],
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    correction: str = "holm",
    seed: int | None = None,
) -> GroupComparisonResult:
    """Run a clustered permutation test for comparing groups.

    Compares assertion-level scores across RAG methods while accounting
    for within-cluster (within-question) correlation by permuting group
    labels at the cluster level. All observations within a cluster are
    reassigned together, preserving the within-cluster correlation
    structure.

    This is the recommended approach for assertion-level significance
    testing when assertions within the same question are correlated
    (Gail et al., 1996; Ernst, 2004).

    For K=2 groups, the test statistic is the absolute difference in
    group means. For K>2 groups, the one-way F-statistic is used as
    the omnibus test, with pairwise 2-group clustered permutation tests
    as post-hoc comparisons.

    Args:
        groups: Dictionary mapping group names (RAG methods) to their
            observation values. Each value is a sequence of floats
            (e.g., assertion scores).
        cluster_ids: Dictionary mapping group names to cluster
            identifiers for each observation, parallel to groups.
            Cluster IDs identify which question each assertion
            belongs to.
        n_permutations: Number of random permutations for Monte
            Carlo approximation (default 10,000).
        alpha: Significance level (default 0.05).
        correction: P-value correction method for post-hoc pairwise
            tests ("holm", "bonferroni", "fdr_bh"). Only used when
            K > 2 groups.
        seed: Random seed for reproducibility (default None).

    Returns
    -------
        GroupComparisonResult with omnibus test result and optional
        post-hoc pairwise comparisons.

    Raises
    ------
        ValueError: If fewer than 2 groups provided, or if groups
            and cluster_ids have mismatched lengths.
    """
    if len(groups) < 2:
        msg = "Need at least 2 groups to compare"
        raise ValueError(msg)

    group_names = list(groups.keys())

    # Validate parallel arrays
    for name in group_names:
        if len(groups[name]) != len(cluster_ids[name]):
            msg = (
                f"Group '{name}' has {len(groups[name])} values but "
                f"{len(cluster_ids[name])} cluster IDs"
            )
            raise ValueError(msg)

    # Build unified arrays: values, labels, clusters
    all_values: list[float] = []
    all_labels: list[str] = []
    all_clusters: list[str] = []
    for name in group_names:
        all_values.extend(groups[name])
        all_labels.extend([name] * len(groups[name]))
        all_clusters.extend(cluster_ids[name])

    values = np.array(all_values, dtype=float)
    labels = np.array(all_labels)
    clusters = np.array(all_clusters)
    unique_clusters = np.unique(clusters)

    rng = np.random.default_rng(seed)

    if len(group_names) == 2:
        return _clustered_permutation_two_groups(
            values=values,
            labels=labels,
            clusters=clusters,
            unique_clusters=unique_clusters,
            group_names=group_names,
            n_permutations=n_permutations,
            alpha=alpha,
            rng=rng,
        )

    return _clustered_permutation_k_groups(
        values=values,
        labels=labels,
        clusters=clusters,
        unique_clusters=unique_clusters,
        group_names=group_names,
        groups=groups,
        cluster_ids=cluster_ids,
        n_permutations=n_permutations,
        alpha=alpha,
        correction=correction,
        rng=rng,
    )


def _clustered_permutation_two_groups(
    values: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    unique_clusters: np.ndarray,
    group_names: list[str],
    n_permutations: int,
    alpha: float,
    rng: np.random.Generator,
) -> GroupComparisonResult:
    """Run clustered permutation test for exactly 2 groups.

    Uses absolute difference in group means as the test statistic.

    Args:
        values: All observation values.
        labels: Group label for each observation.
        clusters: Cluster ID for each observation.
        unique_clusters: Array of unique cluster IDs.
        group_names: List of exactly 2 group names.
        n_permutations: Number of permutations.
        alpha: Significance level.
        rng: Random number generator.

    Returns
    -------
        GroupComparisonResult with the pairwise comparison as both
        omnibus and post-hoc result.
    """
    g1, g2 = group_names

    # Observed statistic: absolute difference in means
    obs_stat = abs(np.mean(values[labels == g1]) - np.mean(values[labels == g2]))

    # Build cluster-to-indices mapping
    cluster_indices: dict[str, np.ndarray] = {}
    for c in unique_clusters:
        cluster_indices[c] = np.where(clusters == c)[0]

    # Permutation loop
    count_extreme = 0
    for _ in range(n_permutations):
        perm_labels = labels.copy()
        for c in unique_clusters:
            idx = cluster_indices[c]
            if rng.random() < 0.5:
                # Swap labels for all observations in this cluster
                perm_labels[idx] = np.where(
                    perm_labels[idx] == g1,
                    g2,
                    np.where(perm_labels[idx] == g2, g1, perm_labels[idx]),
                )
        perm_stat = abs(
            np.mean(values[perm_labels == g1]) - np.mean(values[perm_labels == g2])
        )
        if perm_stat >= obs_stat:
            count_extreme += 1

    p_value = count_extreme / n_permutations

    omnibus = OmnibusTestResult(
        test_name="Clustered permutation test",
        statistic=float(obs_stat),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        alpha=alpha,
        is_normal=False,  # Non-parametric test
        normality_results=[],
    )

    comparison = PairwiseComparison(
        group1=g1,
        group2=g2,
        statistic=float(obs_stat),
        p_value_raw=float(p_value),
        p_value_corrected=float(p_value),
        is_significant=p_value < alpha,
    )

    posthoc = PostHocResult(
        test_name="Clustered permutation test",
        correction_method="none (single comparison)",
        alpha=alpha,
        comparisons=[comparison],
    )

    return GroupComparisonResult(omnibus=omnibus, posthoc=posthoc)


def _clustered_permutation_k_groups(
    values: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    unique_clusters: np.ndarray,
    group_names: list[str],
    groups: Mapping[str, Sequence[float]],
    cluster_ids: Mapping[str, Sequence[str]],
    n_permutations: int,
    alpha: float,
    correction: str,
    rng: np.random.Generator,
) -> GroupComparisonResult:
    """Run clustered permutation test for K > 2 groups.

    Uses the one-way F-statistic as the omnibus test statistic,
    followed by pairwise 2-group clustered permutation tests with
    multiple comparison correction.

    Args:
        values: All observation values.
        labels: Group label for each observation.
        clusters: Cluster ID for each observation.
        unique_clusters: Array of unique cluster IDs.
        group_names: List of K group names (K > 2).
        groups: Original groups mapping (for pairwise tests).
        cluster_ids: Original cluster_ids mapping (for pairwise tests).
        n_permutations: Number of permutations.
        alpha: Significance level.
        correction: P-value correction method for pairwise tests.
        rng: Random number generator.

    Returns
    -------
        GroupComparisonResult with F-statistic omnibus and corrected
        pairwise post-hoc comparisons.
    """
    # Observed F-statistic
    obs_stat = _compute_f_statistic(values, labels, group_names)

    # Build cluster-to-indices mapping
    cluster_indices: dict[str, np.ndarray] = {}
    for c in unique_clusters:
        cluster_indices[c] = np.where(clusters == c)[0]

    # Permutation loop: for each cluster, randomly permute which
    # group label maps to which group's observations
    count_extreme = 0
    for _ in range(n_permutations):
        perm_labels = labels.copy()
        for c in unique_clusters:
            idx = cluster_indices[c]
            # Create a random permutation of group names for this
            # cluster
            perm_map = dict(
                zip(
                    group_names,
                    rng.permutation(group_names),
                    strict=True,
                )
            )
            perm_labels[idx] = np.array([perm_map[lbl] for lbl in labels[idx]])
        perm_stat = _compute_f_statistic(values, perm_labels, group_names)
        if perm_stat >= obs_stat:
            count_extreme += 1

    p_value = count_extreme / n_permutations

    omnibus = OmnibusTestResult(
        test_name="Clustered permutation test (F-statistic)",
        statistic=float(obs_stat),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        alpha=alpha,
        is_normal=False,
        normality_results=[],
    )

    # Pairwise post-hoc tests (only if omnibus is significant)
    posthoc = None
    if p_value < alpha:
        comparisons_raw: list[tuple[str, str, float, float]] = []
        for i, name1 in enumerate(group_names):
            for name2 in group_names[i + 1 :]:
                pair_groups = {
                    name1: groups[name1],
                    name2: groups[name2],
                }
                pair_clusters = {
                    name1: cluster_ids[name1],
                    name2: cluster_ids[name2],
                }
                pair_result = run_clustered_permutation_test(
                    groups=pair_groups,
                    cluster_ids=pair_clusters,
                    n_permutations=n_permutations,
                    alpha=alpha,
                    seed=rng.integers(0, 2**31),
                )
                comparisons_raw.append((
                    name1,
                    name2,
                    pair_result.omnibus.statistic,
                    pair_result.omnibus.p_value,
                ))

        posthoc = _apply_correction_and_build_result(
            comparisons_raw,
            "Clustered permutation test",
            alpha,
            correction,
        )

    return GroupComparisonResult(omnibus=omnibus, posthoc=posthoc)


# ---------------------------------------------------------------------------
#  P-value combination across independent datasets
# ---------------------------------------------------------------------------


@dataclass
class CombinedPValueResult:
    """Result of combining p-values from independent tests.

    Attributes
    ----------
        method: Combination method used (e.g. "fisher", "stouffer").
        statistic: Test statistic from the combination method.
        combined_p_value: Combined p-value.
        is_significant: Whether the combined p-value is below alpha.
        alpha: Significance threshold used.
        input_p_values: Original p-values that were combined.
        dataset_names: Optional names identifying each p-value's source.
        weights: Weights applied (Stouffer's method only).
    """

    method: str
    statistic: float
    combined_p_value: float
    is_significant: bool
    alpha: float
    input_p_values: list[float]
    dataset_names: list[str] = field(default_factory=list)
    weights: list[float] | None = None

    def summary(self) -> str:
        """Return a human-readable summary of the combination result."""
        sig_str = "significant" if self.is_significant else "not significant"
        parts = [
            f"{self.method} combination: "
            f"statistic={self.statistic:.4f}, "
            f"combined_p={self.combined_p_value:.4f} ({sig_str})",
        ]
        for i, p in enumerate(self.input_p_values):
            name = (
                self.dataset_names[i] if i < len(self.dataset_names) else f"dataset_{i}"
            )
            parts.append(f"  {name}: p={p:.4f}")
        return "\n".join(parts)


def combine_pvalues(
    p_values: Sequence[float],
    *,
    method: str = "stouffer",
    weights: Sequence[float] | None = None,
    dataset_names: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> CombinedPValueResult:
    """Combine p-values from independent statistical tests.

    Useful when the same comparison (e.g. RAG-A vs RAG-B) is run on
    multiple independent datasets and you want a single combined
    p-value that aggregates all evidence.

    Supported methods
    -----------------
    - **stouffer** (default): Converts each p-value to a Z-score,
      takes the (optionally weighted) sum, and normalises.  Supports
      weights (e.g. ``sqrt(n_questions)`` per dataset) for unequal
      sample sizes.  More powerful than Fisher when effects are
      consistent in direction.
    - **fisher**: Uses ``-2 * sum(ln(p_i))`` which follows a
      chi-squared distribution with ``2k`` degrees of freedom.
      Sensitive to any single small p-value.

    Args:
        p_values: Sequence of p-values from independent tests.
        method: Combination method — ``"stouffer"`` or ``"fisher"``.
        weights: Optional weights for Stouffer's method.  Ignored for
            Fisher's.  Typically ``sqrt(n_i)`` where ``n_i`` is the
            number of observations in dataset *i*.
        dataset_names: Optional labels identifying each dataset
            (used in the result summary).
        alpha: Significance threshold for the combined test.

    Returns
    -------
        CombinedPValueResult with the combined test statistic and
        p-value.

    Raises
    ------
        ValueError: If fewer than 2 p-values are provided, if any
            p-value is outside (0, 1], if weights length does not
            match p_values, or if an unsupported method is given.
    """
    _validate_combine_inputs(p_values, method, weights)

    p_arr = np.asarray(p_values, dtype=np.float64)

    if method == "fisher":
        statistic, combined_p = stats.combine_pvalues(p_arr, method="fisher")
    elif method == "stouffer":
        w = np.asarray(weights, dtype=np.float64) if weights is not None else None
        statistic, combined_p = stats.combine_pvalues(
            p_arr, method="stouffer", weights=w
        )
    else:
        msg = f"Unsupported method '{method}'. Use 'fisher' or 'stouffer'."
        raise ValueError(msg)

    return CombinedPValueResult(
        method=method,
        statistic=float(statistic),  # type: ignore[arg-type]
        combined_p_value=float(combined_p),  # type: ignore[arg-type]
        is_significant=bool(combined_p < alpha),  # type: ignore[operator]
        alpha=alpha,
        input_p_values=list(p_values),
        dataset_names=list(dataset_names) if dataset_names else [],
        weights=list(weights) if weights is not None else None,
    )


def _validate_combine_inputs(
    p_values: Sequence[float],
    method: str,
    weights: Sequence[float] | None,
) -> None:
    """Validate inputs for combine_pvalues.

    Args:
        p_values: The p-values to validate.
        method: Combination method name.
        weights: Optional weights to validate.

    Raises
    ------
        ValueError: On invalid inputs.
    """
    if len(p_values) < 2:
        msg = f"Need at least 2 p-values to combine, got {len(p_values)}."
        raise ValueError(msg)

    for i, p in enumerate(p_values):
        if not (0 < p <= 1):
            msg = f"p_values[{i}]={p} is outside the valid range (0, 1]."
            raise ValueError(msg)

    if method not in {"fisher", "stouffer"}:
        msg = f"Unsupported method '{method}'. Use 'fisher' or 'stouffer'."
        raise ValueError(msg)

    if weights is not None and len(weights) != len(p_values):
        msg = (
            f"weights length ({len(weights)}) must match "
            f"p_values length ({len(p_values)})."
        )
        raise ValueError(msg)
