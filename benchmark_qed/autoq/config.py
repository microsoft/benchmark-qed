# Copyright (c) 2025 Microsoft Corporation.
"""Configuration for the autoq question generation process."""

from pathlib import Path

from pydantic import BaseModel, Field

from benchmark_qed.autod import prompts as autod_prompts
from benchmark_qed.autod.io.enums import InputDataType
from benchmark_qed.autoq.prompts import activity_questions as activity_prompts
from benchmark_qed.autoq.prompts import data_questions as data_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.prompt_config import PromptConfig

AUTOD_PROMPTS_PATH = Path(autod_prompts.__file__).parent
AUTOQ_ACTIVITY_PROMPTS_PATH = Path(activity_prompts.__file__).parent
AUTOQ_DATA_PROMPTS_PATH = Path(data_prompts.__file__).parent


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


class MapReducePromptConfig(BaseModel):
    """Configuration for the map/reduce summary prompts."""

    map_summary_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOD_PROMPTS_PATH / "map_summary_system_prompt.txt"
        ),
        description="System prompt for the map summary step in question generation.",
    )
    map_summary_user_prompt: PromptConfig = Field(
        default=PromptConfig(prompt=AUTOD_PROMPTS_PATH / "map_summary_user_prompt.txt"),
        description="User prompt for the map summary step in question generation.",
    )
    reduce_summary_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOD_PROMPTS_PATH / "reduce_summary_system_prompt.txt"
        ),
        description="System prompt for the reduce summary step in question generation.",
    )
    reduce_summary_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOD_PROMPTS_PATH / "reduce_summary_user_prompt.txt"
        ),
        description="User prompt for the reduce summary step in question generation.",
    )


class AutoQActivityConfig(BaseModel):
    """Configuration for activity-related prompts."""

    activity_identification_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH / "activity_identification_prompt.txt"
        ),
        description="Prompt for identifying activities in the question generation process.",
    )
    global_generation_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH / "global_generation_system_prompt.txt"
        ),
        description="System prompt for generating global questions in question generation.",
    )
    global_generation_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH / "global_generation_user_prompt.txt"
        ),
        description="User prompt for generating global questions in question generation.",
    )
    local_generation_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH / "local_generation_system_prompt.txt"
        ),
        description="System prompt for generating local activity questions.",
    )
    local_generation_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH / "local_generation_user_prompt.txt"
        ),
        description="User prompt for generating local activity questions.",
    )
    map_entity_extraction_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH
            / "map_entity_extraction_system_prompt.txt"
        ),
        description="System prompt for extracting entities in the map step.",
    )
    map_entity_extraction_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH / "map_entity_extraction_user_prompt.txt"
        ),
        description="User prompt for extracting entities in the map step.",
    )
    reduce_entity_extraction_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH
            / "reduce_entity_extraction_system_prompt.txt"
        ),
        description="System prompt for extracting entities in the reduce step.",
    )
    reduce_entity_extraction_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_PROMPTS_PATH
            / "reduce_entity_extraction_user_prompt.txt"
        ),
        description="User prompt for extracting entities in the reduce step.",
    )


class AutoQDataConfig(BaseModel):
    """Configuration for data-related prompts."""

    claim_extraction_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_PROMPTS_PATH / "claim_extraction_system_prompt.txt"
        ),
        description="System prompt for extracting claims from data tables.",
    )
    global_extraction_input_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_PROMPTS_PATH / "global_extraction_input_prompt.txt"
        ),
        description="Prompt for input to global extraction.",
    )
    global_extraction_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_PROMPTS_PATH / "global_extraction_prompt.txt"
        ),
        description="Prompt for extracting global questions from local questions.",
    )
    local_extraction_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_PROMPTS_PATH / "local_extraction_prompt.txt"
        ),
        description="Prompt for extracting local questions from input texts.",
    )
    local_generation_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_PROMPTS_PATH / "local_generation_prompt.txt"
        ),
        description="Prompt for generating local questions from input questions.",
    )
    local_text_input_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_PROMPTS_PATH / "local_text_input_prompt.txt"
        ),
        description="Prompt for input texts for local question generation.",
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

    map_reduce_prompt_config: MapReducePromptConfig = Field(
        default_factory=MapReducePromptConfig,
        description="Configuration for the map/reduce summary prompts.",
    )

    activity_questions_prompt_config: AutoQActivityConfig = Field(
        default_factory=AutoQActivityConfig,
        description="Configuration for activity-related prompts.",
    )

    data_questions_prompt_config: AutoQDataConfig = Field(
        default_factory=AutoQDataConfig,
        description="Configuration for data-related prompts.",
    )
