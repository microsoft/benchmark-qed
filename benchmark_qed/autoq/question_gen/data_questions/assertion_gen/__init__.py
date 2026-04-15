# Copyright (c) 2025 Microsoft Corporation.
"""Assertion generation for evaluating answer accuracy in question-answering systems."""

from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.base import (
    Assertion,
    AssertionGenerationResult,
    BaseAssertionGenerator,
    ClaimDict,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.global_claim_assertion_gen import (
    GlobalClaimAssertionGenerator,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.local_claim_assertion_gen import (
    LocalClaimAssertionGenerator,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.stats import (
    AssertionStats,
    DistributionStats,
    compute_assertion_stats,
    compute_stats_from_file,
    generate_stats_for_assertion_file,
    generate_stats_for_directory,
    save_stats_to_file,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.validator import (
    AssertionValidator,
    ValidationResult,
    ValidationScores,
    ValidationSummary,
)

__all__ = [
    "Assertion",
    "AssertionGenerationResult",
    "AssertionStats",
    "AssertionValidator",
    "BaseAssertionGenerator",
    "ClaimDict",
    "DistributionStats",
    "GlobalClaimAssertionGenerator",
    "LocalClaimAssertionGenerator",
    "ValidationResult",
    "ValidationScores",
    "ValidationSummary",
    "compute_assertion_stats",
    "compute_stats_from_file",
    "generate_stats_for_assertion_file",
    "generate_stats_for_directory",
    "save_stats_to_file",
]
