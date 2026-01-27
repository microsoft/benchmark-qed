# Copyright (c) 2025 Microsoft Corporation.
"""Scoring configuration models."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from benchmark_qed.autoe.prompts import assertion as assertion_prompts
from benchmark_qed.autoe.prompts import pairwise as pairwise_prompts
from benchmark_qed.autoe.prompts import reference as reference_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.model.score import (
    Assertions,
    Condition,
    Criteria,
    pairwise_scores_criteria,
    reference_scores_criteria,
)
from benchmark_qed.config.prompt_config import PromptConfig

# Available relevance assessor types
RelevanceAssessorType = Literal["rationale", "bing"]


class TextUnitFieldsConfig(BaseModel):
    """Configuration for mapping text unit column names from input data.

    Use this to map column names from your parquet/JSON file to the expected fields.
    Set a field to None if the column doesn't exist and should be auto-generated.
    """

    id_col: str = Field(
        default="id",
        description="Column name for unique text unit identifier.",
    )

    text_col: str = Field(
        default="text",
        description="Column name for text content.",
    )

    embedding_col: str | None = Field(
        default="text_embedding",
        description="Column name for embeddings. Set to None to auto-generate embeddings.",
    )

    short_id_col: str | None = Field(
        default="short_id",
        description="Column name for short ID. Set to None to auto-generate from index.",
    )


class AutoEPromptConfig(BaseModel):
    """Configuration for prompts used in AutoE scoring."""

    user_prompt: PromptConfig = Field(
        ...,
        description="User prompt configuration for scoring.",
    )

    system_prompt: PromptConfig = Field(
        ...,
        description="System prompt configuration for scoring.",
    )


class BaseAutoEConfig(BaseModel):
    """Base configuration for AutoE scoring."""

    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for scoring.",
    )

    trials: int = Field(
        default=4,
        description="Number of trials to run for each condition.",
    )

    prompt_config: AutoEPromptConfig = Field(
        ...,
        description="Configuration for prompts used in scoring.",
    )

    @model_validator(mode="after")
    def check_trials_even(self) -> Self:
        """Check if the number of trials is even."""
        if self.trials % 2 != 0:
            msg = "The number of trials must be even to allow for counterbalancing of conditions."
            raise ValueError(msg)
        return self


class PairwiseConfig(BaseAutoEConfig):
    """Configuration for scoring a set of conditions."""

    base: Condition | None = Field(default=None, description="Base Conditions.")

    others: list[Condition] = Field(
        default_factory=list,
        description="Other Conditions to compare against the base.",
    )

    question_sets: list[str] = Field(
        default_factory=list,
        description="List of question sets to use for scoring.",
    )

    criteria: list[Criteria] = Field(
        default_factory=pairwise_scores_criteria,
        description="List of criteria to use for scoring.",
    )

    prompt_config: AutoEPromptConfig = Field(
        default=AutoEPromptConfig(
            user_prompt=PromptConfig(
                prompt=Path(pairwise_prompts.__file__).parent
                / "pairwise_user_prompt.txt",
            ),
            system_prompt=PromptConfig(
                prompt=Path(pairwise_prompts.__file__).parent
                / "pairwise_system_prompt.txt",
            ),
        ),
        description="Configuration for prompts used in pairwise scoring.",
    )


class ReferenceConfig(BaseAutoEConfig):
    """Configuration for scoring based on reference answers."""

    reference: Condition = Field(
        ..., description="Condition with the reference answers."
    )
    generated: list[Condition] = Field(
        default_factory=list,
        description="Conditions with the generated answers to score.",
    )
    criteria: list[Criteria] = Field(
        default_factory=reference_scores_criteria,
        description="List of criteria to use for scoring.",
    )
    score_min: int = Field(1, description="Minimum score for the criteria.")
    score_max: int = Field(10, description="Maximum score for the criteria.")

    prompt_config: AutoEPromptConfig = Field(
        default=AutoEPromptConfig(
            user_prompt=PromptConfig(
                prompt=Path(reference_prompts.__file__).parent
                / "reference_user_prompt.txt",
            ),
            system_prompt=PromptConfig(
                prompt=Path(reference_prompts.__file__).parent
                / "reference_system_prompt.txt",
            ),
        ),
        description="Configuration for prompts used in reference scoring.",
    )


class AssertionConfig(BaseAutoEConfig):
    """Configuration for scoring based on assertions."""

    generated: Condition = Field(
        ...,
        description="Conditions with the generated answers to test.",
    )
    assertions: Assertions = Field(
        ...,
        description="List of assertions to use for scoring.",
    )

    pass_threshold: float = Field(
        0.5,
        description="Threshold for passing the assertion score.",
    )

    prompt_config: AutoEPromptConfig = Field(
        default=AutoEPromptConfig(
            user_prompt=PromptConfig(
                prompt=Path(assertion_prompts.__file__).parent
                / "assertion_user_prompt.txt",
            ),
            system_prompt=PromptConfig(
                prompt=Path(assertion_prompts.__file__).parent
                / "assertion_system_prompt.txt",
            ),
        ),
        description="Configuration for prompts used in assertion scoring.",
    )

    @model_validator(mode="after")
    def check_trials_even(self) -> Self:
        """Even number of trials check does not apply for assertion scoring."""
        return self


class RAGMethod(BaseModel):
    """Configuration for a RAG method to evaluate."""

    name: str = Field(..., description="Name of the RAG method.")
    retrieval_results_path: Path = Field(
        ..., description="Path to the retrieval results JSON file."
    )


class QuestionSetConfig(BaseModel):
    """Configuration for a single question set."""

    name: str = Field(..., description="Name identifier for this question set (e.g., 'global', 'local').")
    questions_path: Path = Field(..., description="Path to JSON file with questions.")


class RetrievalReferenceConfig(BaseModel):
    """Configuration for generating retrieval reference data.

    Supports multiple question sets and cluster counts for batch processing.
    Results are saved in structured subdirectories:
      output_dir/
        clusters/
          clusters_{num_clusters}.json
        {question_set_name}/
          clusters_{num_clusters}/
            reference.json
            model_usage.json
    """

    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for relevance assessment.",
    )

    embedding_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the embedding model (used if text units need embeddings).",
    )

    # Support both single path (backward compatible) and multiple question sets
    questions_path: Path | None = Field(
        default=None,
        description="Path to JSON file with questions (for single question set mode).",
    )

    question_sets: list[QuestionSetConfig] | None = Field(
        default=None,
        description="List of question sets to process. If provided, questions_path is ignored.",
    )

    clusters_path: Path | None = Field(
        default=None,
        description="Path to JSON file with pre-computed cluster data. If not provided, clustering will be performed on text_units_path.",
    )

    text_units_path: Path = Field(
        ..., description="Path to parquet or JSON file with text units and embeddings."
    )

    output_dir: Path = Field(
        ..., description="Directory to save reference results."
    )

    # Support both single value (backward compatible) and multiple cluster counts
    num_clusters: int | list[int] | None = Field(
        default=None,
        description="Number of clusters to create. Can be a single int or list of ints for multiple runs. If None, will be auto-determined.",
    )

    save_clusters: bool = Field(
        default=True,
        description="Whether to save clustering results to separate files for debugging.",
    )

    semantic_neighbors: int = Field(
        default=10,
        description="Number of semantically similar chunks to test per cluster.",
    )

    centroid_neighbors: int = Field(
        default=5,
        description="Number of centroid neighbors to test per cluster.",
    )

    relevance_threshold: int = Field(
        default=2,
        description="Minimum relevance score to consider relevant (0-3 scale).",
    )

    assessor_type: RelevanceAssessorType = Field(
        default="rationale",
        description="Type of relevance assessor: 'rationale' (structured JSON with reasoning) or 'bing' (UMBRELA DNA prompt).",
    )

    concurrent_requests: int = Field(
        default=16,
        description="Maximum number of concurrent LLM requests for relevance assessment.",
    )

    max_questions: int | None = Field(
        default=None,
        description="Maximum number of questions to process. If None, process all questions.",
    )

    text_unit_fields: TextUnitFieldsConfig = Field(
        default_factory=TextUnitFieldsConfig,
        description="Column name mappings for text unit data.",
    )

    cache_dir: Path | None = Field(
        default=None,
        description="Directory for caching relevance assessments.",
    )

    @model_validator(mode="after")
    def validate_question_config(self) -> Self:
        """Ensure either questions_path or question_sets is provided."""
        if self.questions_path is None and self.question_sets is None:
            msg = "Either 'questions_path' or 'question_sets' must be provided."
            raise ValueError(msg)
        return self

    def get_question_sets(self) -> list[QuestionSetConfig]:
        """Get list of question sets to process."""
        if self.question_sets is not None:
            return self.question_sets
        # Backward compatible: single questions_path
        return [QuestionSetConfig(name="default", questions_path=self.questions_path)]  # type: ignore[arg-type]

    def get_cluster_counts(self) -> list[int | None]:
        """Get list of cluster counts to process."""
        if self.num_clusters is None:
            return [None]
        if isinstance(self.num_clusters, int):
            return [self.num_clusters]
        return self.num_clusters


class RetrievalScoresConfig(BaseModel):
    """Configuration for retrieval metrics evaluation."""

    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for relevance assessment.",
    )

    assessor_type: RelevanceAssessorType = Field(
        default="rationale",
        description="Type of relevance assessor: 'rationale' (structured JSON with reasoning) or 'bing' (UMBRELA DNA prompt). Must match the assessor used in generate-retrieval-reference to use the cache.",
    )

    rag_methods: list[RAGMethod] = Field(
        default_factory=list,
        description="List of RAG methods to evaluate.",
    )

    question_sets: list[str] = Field(
        default_factory=list,
        description="List of question set names to evaluate.",
    )

    reference_dir: Path = Field(
        ..., description="Directory containing reference data from generate-retrieval-reference."
    )

    reference_filename: str = Field(
        default="reference.json",
        description="Filename for reference data within reference_dir.",
    )

    clusters_path: Path = Field(
        ..., description="Path to JSON file with cluster data."
    )

    text_units_path: Path = Field(
        ..., description="Path to JSON file with text units."
    )

    output_dir: Path = Field(
        ..., description="Directory to save evaluation results."
    )

    relevance_threshold: int = Field(
        default=2,
        description="Minimum relevance score to consider relevant (0-3 scale).",
    )

    cache_dir: Path | None = Field(
        default=None,
        description="Directory for caching relevance assessments (shared across RAG methods).",
    )

    context_id_key: str = Field(
        default="chunk_id",
        description="Key name for chunk ID in retrieval results.",
    )

    context_text_key: str = Field(
        default="text",
        description="Key name for chunk text in retrieval results.",
    )

    run_significance_test: bool = Field(
        default=True,
        description="Whether to run statistical significance tests.",
    )

    significance_alpha: float = Field(
        default=0.05,
        description="Alpha level for significance tests.",
    )

    significance_correction: str = Field(
        default="holm",
        description="P-value correction method for post-hoc tests.",
    )

    fidelity_metric: str = Field(
        default="js",
        description="Fidelity metric to use: 'js' (Jensen-Shannon) or 'tvd' (Total Variation Distance).",
    )
