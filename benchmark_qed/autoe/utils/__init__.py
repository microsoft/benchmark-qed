# Copyright (c) 2025 Microsoft Corporation.
"""Utility modules for autoe."""

from benchmark_qed.autoe.utils.stats import (
    CORRECTION_DISPLAY_NAMES,
    GroupComparisonResult,
    NormalityResult,
    OmnibusTestResult,
    PairwiseComparison,
    PostHocResult,
    check_normality,
    compare_groups,
    run_omnibus_test,
    run_posthoc_pairwise,
)

__all__ = [
    "CORRECTION_DISPLAY_NAMES",
    "GroupComparisonResult",
    "NormalityResult",
    "OmnibusTestResult",
    "PairwiseComparison",
    "PostHocResult",
    "check_normality",
    "compare_groups",
    "run_omnibus_test",
    "run_posthoc_pairwise",
]
