# Copyright (c) 2025 Microsoft Corporation.
"""Configuration for the autoq question generation process."""

from pathlib import Path

from pydantic import BaseModel, Field

from benchmark_qed.autod.io.enums import InputDataType
from benchmark_qed.config.llm_config import LLMConfig


class InputConfig(BaseModel):
    """Configuration for the input data used in question generation."""

    dataset_path: Path = Field(
        ...,
        description="Path to the input dataset file.",
    )

    input_type: InputDataType = Field(
        default=InputDataType.CSV, description="The type of the input data."
    )
    text_column: str = Field(
        default="text", description="The column containing the text data."
    )
    metadata_columns: list[str] | None = Field(
        default=None, description="The columns containing metadata information."
    )
    file_encoding: str = Field(
        default="utf-8", description="The encoding of the input files."
    )


class QuestionConfig(BaseModel):
    """Configuration for the question generation process."""

    num_questions: int = Field(
        default=20,
        description="Number of questions to generate for each question class.",
    )
    oversample_factor: float = Field(
        default=2.0,
        description="Factor by which to overgenerate candidate questions before filtering.",
    )


class EncodingModelConfig(BaseModel):
    """Configuration for the encoding model used in question generation."""

    model_name: str = Field(
        default="o200k_base",
        description="Name of the encoding model to use for chunking documents.",
    )
    chunk_size: int = Field(
        default=600,
        description="Size of each text chunk to be processed by the encoding model.",
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap size between consecutive text chunks.",
    )


class SamplingConfig(BaseModel):
    """Configuration for data sampling in question generation."""

    num_clusters: int = Field(
        default=50,
        description="Number of clusters to sample from the dataset.",
    )
    num_samples_per_cluster: int = Field(
        default=10,
        description="Number of samples to take from each cluster.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility of sampling.",
    )


class ActivityQuestionConfig(QuestionConfig):
    """Configuration for generating activity questions."""

    num_personas: int = Field(
        default=5,
        description="Number of personas to generate questions for.",
    )
    num_tasks_per_persona: int = Field(
        default=5,
        description="Number of tasks to generate for each persona.",
    )
    num_entities_per_task: int = Field(
        default=10,
        description="Number of entities to include in each task.",
    )


class QuestionGenerationConfig(BaseModel):
    """Configuration for question generation."""

    input: InputConfig = Field(
        ...,
        description="Configuration for the input data used in question generation.",
    )

    data_local: QuestionConfig = Field(
        default_factory=QuestionConfig,
        description="Configuration for generating questions from local data.",
    )

    data_global: QuestionConfig = Field(
        default_factory=QuestionConfig,
        description="Configuration for generating questions from global data.",
    )

    activity_local: ActivityQuestionConfig = Field(
        default_factory=ActivityQuestionConfig,
        description="Configuration for generating local activity questions.",
    )

    activity_global: ActivityQuestionConfig = Field(
        default_factory=ActivityQuestionConfig,
        description="Configuration for generating global activity questions.",
    )

    concurrent_requests: int = Field(
        default=8,
        description="Control for request concurrency. Adjust this based on your model capacity.",
    )

    encoding: EncodingModelConfig = Field(
        default_factory=EncodingModelConfig,
        description="Configuration for the encoding model to use for question generation.",
    )

    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Configuration for data sampling in question generation.",
    )

    chat_model: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for chat.",
    )

    embedding_model: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for embedding.",
    )
