# Copyright (c) 2025 Microsoft Corporation.
"""Data models for chunk-level assertion evaluation."""

from pathlib import Path
from typing import ClassVar

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


class EvalSummary(BaseModel):
    """Per-k chunk-eval summary with stable metric surface.

    Exposes Coverage, Strict Coverage, Coverage Strength, plus bookkeeping
    counts needed for paired significance testing. Per-query rows are
    persisted alongside for downstream analysis.
    """

    k: int | None = Field(default=None)
    n_questions: int = Field(default=0)
    n_assertions: int = Field(default=0)
    n_questions_total: int = Field(default=0)
    n_assertions_total: int = Field(default=0)
    coverage: float = Field(default=0.0)
    strict_coverage: float = Field(default=0.0)
    mean_score: float = Field(default=0.0)
    mean_retrieved_chunks: float = Field(default=0.0)
    pass_rate: float = Field(default=0.0)
    total_calls: int = Field(default=0)
    successful_calls: int = Field(default=0)
    failed_calls: int = Field(default=0)
    eval_mode: str = Field(default="chunk")
    # Per-question rows: {q_idx_str: {coverage, strict_coverage, mean_score}}
    # Same shape across every k so single comparator can process them uniformly
    per_query_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)


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
    """Generated/retrieval configuration."""

    name: str
    retrieval_path: Path | None = Field(
        default=None,
        description="Path to a JSON array of RetrievalResult records "
        "(question_id, question_text, context[]).",
    )


class AssertionsConfig(BaseModel):
    """Assertions configuration."""

    assertions_path: Path


class ChunkAssertionConfig(BaseModel):
    """Configuration for chunk-level assertion evaluation."""

    generated: GeneratedConfig
    assertions: AssertionsConfig
    k_list: list[int] = Field(default_factory=lambda: [5, 10, 20, 50])
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
