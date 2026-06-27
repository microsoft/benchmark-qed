# Copyright (c) 2025 Microsoft Corporation.
"""Data models for chunk-level assertion evaluation."""

from pathlib import Path
from typing import Any, ClassVar

from graphrag_storage.storage_config import StorageConfig
from pydantic import BaseModel, Field

from benchmark_qed.config.llm_config import LLMConfig


class ChunkAssertionGrade:
    """Chunk-level assertion grade constants."""

    FULL_SUPPORT = "full_support"
    PARTIAL_SUPPORT = "partial_support"
    NO_SUPPORT = "no_support"

    ALL: ClassVar[list[str]] = [FULL_SUPPORT, PARTIAL_SUPPORT, NO_SUPPORT]


def grade_to_score(grade: str) -> float:
    """Convert grade string to numeric score.

    Args:
        grade: One of 'full_support', 'partial_support', 'no_support'

    Returns
    -------
        1.0 for full_support, 0.5 for partial_support, 0.0 for no_support
    """
    if grade == ChunkAssertionGrade.FULL_SUPPORT:
        return 1.0
    if grade == ChunkAssertionGrade.PARTIAL_SUPPORT:
        return 0.5
    return 0.0


class EvalSummary:
    """Per-k chunk-eval summary with stable metric surface.

    Exposes Coverage, Strict Coverage, Coverage Strength, plus bookkeeping
    counts needed for paired significance testing. Per-query rows are
    persisted alongside for downstream analysis.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.k: int | None = kwargs.get("k")
        self.n_questions: int = kwargs.get("n_questions", 0)
        self.n_assertions: int = kwargs.get("n_assertions", 0)
        self.n_questions_total: int = kwargs.get("n_questions_total", self.n_questions)
        self.n_assertions_total: int = kwargs.get(
            "n_assertions_total", self.n_assertions
        )
        self.coverage: float = kwargs.get("coverage", 0.0)
        self.strict_coverage: float = kwargs.get("strict_coverage", 0.0)
        self.mean_score: float = kwargs.get("mean_score", 0.0)
        self.mean_retrieved_chunks: float = kwargs.get("mean_retrieved_chunks", 0.0)
        self.pass_rate: float = kwargs.get("pass_rate", 0.0)
        self.total_calls: int = kwargs.get("total_calls", 0)
        self.successful_calls: int = kwargs.get("successful_calls", 0)
        self.failed_calls: int = kwargs.get("failed_calls", 0)
        self.eval_mode: str = kwargs.get("eval_mode", "chunk")
        # Per-question rows: {q_idx_str: {coverage, strict_coverage, mean_score}}
        # Same shape across every k so single comparator can process them uniformly
        self.per_query_metrics: dict[str, dict[str, float]] = kwargs.get(
            "per_query_metrics", {}
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "k": self.k,
            "n_questions": self.n_questions,
            "n_assertions": self.n_assertions,
            "n_questions_total": self.n_questions_total,
            "n_assertions_total": self.n_assertions_total,
            "coverage": self.coverage,
            "strict_coverage": self.strict_coverage,
            "mean_score": self.mean_score,
            "mean_retrieved_chunks": self.mean_retrieved_chunks,
            "pass_rate": self.pass_rate,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "eval_mode": self.eval_mode,
            "per_query_metrics": self.per_query_metrics,
        }


class SystemPromptConfig(BaseModel):
    """System prompt configuration."""

    prompt: str | None = Field(default=None)
    template: str | None = Field(default=None)


class UserPromptConfig(BaseModel):
    """User prompt configuration."""

    prompt: str | None = Field(default=None)
    template: str | None = Field(default=None)


class PromptConfig(BaseModel):
    """Prompt configuration for chunk evaluation."""

    system_prompt: SystemPromptConfig | None = Field(default=None)
    user_prompt: UserPromptConfig | None = Field(default=None)


class GeneratedConfig(BaseModel):
    """Generated/answers configuration."""

    name: str
    answer_base_path: Path | None = Field(default=None)
    chunks_path: Path | None = Field(default=None)


class AssertionsConfig(BaseModel):
    """Assertions configuration."""

    assertions_path: Path


class ChunkAssertionConfig(BaseModel):
    """Configuration for chunk-level assertion evaluation."""

    generated: GeneratedConfig
    assertions: AssertionsConfig
    k_list: list[int] = Field(default=[5, 10, 20, 50])
    pass_threshold: float = Field(default=0.5)
    max_chunks_per_question: int | None = Field(
        default=None,
        description="Cap the number of chunks evaluated per question (keeps the "
        "highest-ranked chunks). Useful for reducing LLM call volume during "
        "testing. None means evaluate all retrieved chunks.",
    )
    cache_dir: str | None = Field(default=None)
    output_storage: StorageConfig | None = Field(default=None)
    input_storage: StorageConfig | None = Field(default=None)
    llm_config: LLMConfig
    prompt_config: PromptConfig | None = Field(default=None)
