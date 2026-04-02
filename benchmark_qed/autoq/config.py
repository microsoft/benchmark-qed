# Copyright (c) 2025 Microsoft Corporation.
"""Configuration for the autoq question generation process."""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from benchmark_qed.autod import prompts as autod_prompts
from benchmark_qed.autod.io.enums import InputDataType
from benchmark_qed.autoq.prompts import data_questions as autoq_data_prompts
from benchmark_qed.autoq.prompts.activity_questions import (
    activity_context as activity_context_prompts,
)
from benchmark_qed.autoq.prompts.activity_questions import (
    global_questions as activity_global_prompts,
)
from benchmark_qed.autoq.prompts.activity_questions import (
    local_questions as activity_local_prompts,
)
from benchmark_qed.autoq.prompts.data_questions import (
    global_questions as data_global_prompts,
)
from benchmark_qed.autoq.prompts.data_questions import (
    linked_questions as data_linked_prompts,
)
from benchmark_qed.autoq.prompts.data_questions import (
    local_questions as data_local_prompts,
)
from benchmark_qed.config import defaults as defs
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.prompt_config import PromptConfig

AUTOD_PROMPTS_PATH = Path(autod_prompts.__file__).parent

AUTOQ_ACTIVITY_CONTEXT_PROMPTS_PATH = Path(activity_context_prompts.__file__).parent
AUTOQ_ACTIVITY_GLOBAL_PROMPTS_PATH = Path(activity_global_prompts.__file__).parent
AUTOQ_ACTIVITY_LOCAL_PROMPTS_PATH = Path(activity_local_prompts.__file__).parent

AUTOQ_DATA_PROMPTS_PATH = Path(autoq_data_prompts.__file__).parent
AUTOQ_DATA_GLOBAL_PROMPTS_PATH = Path(data_global_prompts.__file__).parent
AUTOQ_DATA_LINKED_PROMPTS_PATH = Path(data_linked_prompts.__file__).parent
AUTOQ_DATA_LOCAL_PROMPTS_PATH = Path(data_local_prompts.__file__).parent
AUTOQ_ASSERTIONS_PROMPTS_PATH = AUTOQ_DATA_PROMPTS_PATH / "assertions"


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
        default=defs.TEXT_COLUMN, description="The column containing the text data."
    )
    metadata_columns: list[str] | None = Field(
        default=None, description="The columns containing metadata information."
    )
    file_encoding: str = Field(
        default=defs.FILE_ENCODING, description="The encoding of the input files."
    )


class QuestionConfig(BaseModel):
    """Configuration for the question generation process."""

    num_questions: int = Field(
        default=defs.NUM_QUESTIONS,
        description="Number of questions to generate for each question class.",
    )
    oversample_factor: float = Field(
        default=defs.OVERSAMPLE_FACTOR,
        description="Factor by which to overgenerate candidate questions before filtering.",
    )


class DataGlobalQuestionConfig(QuestionConfig):
    """Configuration for data-global question generation."""

    min_questions_in_context: int = Field(
        default=defs.MIN_QUESTIONS_IN_CONTEXT,
        description="Minimum number of local questions required in context for data_global generation. Categories with fewer questions are excluded.",
    )
    min_claim_count: int = Field(
        default=defs.MIN_CLAIM_COUNT,
        description="Minimum number of claims required for a question to be considered valid. Questions with fewer claims are filtered before assertion generation.",
    )
    min_relevant_reference_count: int = Field(
        default=defs.MIN_RELEVANT_REFERENCE_COUNT,
        description="Soft minimum relevant reference count (local questions contributing claims). Questions below this threshold are deprioritized but included if needed.",
    )
    enable_question_validation: bool = Field(
        default=True,
        description="Whether to validate generated questions for naturalness, answerability, and clarity using LLM.",
    )


class DataLinkedQuestionConfig(QuestionConfig):
    """Configuration for data-linked question generation.

    Linked questions combine multiple local questions that share named entities
    to create harder, multi-hop style questions (similar to HotpotQA).
    """

    min_questions_per_entity: int = Field(
        default=2,
        description="Minimum number of local questions required to form an entity group.",
    )
    max_questions_per_entity: int = Field(
        default=3,
        description="Maximum number of local questions to include per entity group. If more exist, selects most related by embedding similarity.",
    )
    type_balance_weight: float = Field(
        default=0.5,
        description="Weight for type-balance penalty in MMR selection. 0=ignore type balance, 1=strong type balancing. Higher values produce more even distribution across question types (bridge, comparison, temporal, intersection).",
    )
    max_questions_to_generate: int = Field(
        default=2,
        description="Maximum number of entity questions to generate per entity group. LLM chooses appropriate question types.",
    )
    entity_frequency_threshold: int = Field(
        default=2,
        description="Entity must appear in at least this many local questions to be considered for grouping.",
    )


class LocalAssertionConfig(BaseModel):
    """Configuration for local assertion generation."""

    max_assertions: int | None = Field(
        default=defs.MAX_ASSERTIONS,
        description="Maximum number of assertions per question. Set to 0 to disable, or None for unlimited.",
    )
    enable_validation: bool = Field(
        default=defs.ENABLE_ASSERTION_VALIDATION,
        description="Whether to validate assertions against sources for quality filtering.",
    )
    min_validation_score: int = Field(
        default=defs.MIN_ASSERTION_VALIDATION_SCORE,
        description="Minimum score (1-5) for grounding, relevance, and verifiability criteria.",
    )
    max_source_count: int = Field(
        default=defs.ASSERTION_MAX_SOURCE_COUNT,
        description="Maximum number of deduplicated sources per assertion. Assertions exceeding this cause the entire question to be dropped.",
    )
    concurrent_llm_calls: int = Field(
        default=defs.ASSERTION_CONCURRENT_LLM_CALLS,
        description="Number of concurrent LLM calls for validation.",
    )
    max_concurrent_questions: int | None = Field(
        default=defs.ASSERTION_MAX_CONCURRENT_LOCAL_QUESTIONS,
        description="Maximum questions to process in parallel. Set to 1 for sequential.",
    )


class GlobalAssertionConfig(BaseModel):
    """Configuration for global assertion generation."""

    max_assertions: int | None = Field(
        default=defs.MAX_ASSERTIONS,
        description="Maximum number of assertions per question. Set to 0 to disable, or None for unlimited.",
    )
    enable_validation: bool = Field(
        default=defs.ENABLE_ASSERTION_VALIDATION,
        description="Whether to validate assertions against sources for quality filtering.",
    )
    min_validation_score: int = Field(
        default=defs.MIN_ASSERTION_VALIDATION_SCORE,
        description="Minimum score (1-5) for grounding, relevance, and verifiability criteria.",
    )
    max_source_count: int = Field(
        default=defs.ASSERTION_MAX_SOURCE_COUNT,
        description="Maximum number of deduplicated sources per assertion. Assertions exceeding this cause the entire question to be dropped.",
    )
    batch_size: int = Field(
        default=defs.ASSERTION_BATCH_SIZE,
        description="Batch size for processing claims in map-reduce assertion generation (used when semantic grouping is disabled).",
    )
    map_data_tokens: int = Field(
        default=defs.ASSERTION_MAP_DATA_TOKENS,
        description="Maximum tokens per cluster in the map step when semantic grouping is enabled.",
    )
    reduce_data_tokens: int = Field(
        default=defs.ASSERTION_REDUCE_DATA_TOKENS,
        description="Maximum input data tokens for the reduce step.",
    )
    enable_semantic_grouping: bool = Field(
        default=True,
        description="Whether to group similar claims together using embedding-based clustering before the map step. Requires text_embedder to be provided.",
    )
    validate_map_assertions: bool = Field(
        default=True,
        description="Whether to validate map assertions before the reduce step. Filters low-quality assertions early but increases LLM calls.",
    )
    validate_reduce_assertions: bool = Field(
        default=True,
        description="Whether to validate final assertions after the reduce step. Filters low-quality consolidated assertions.",
    )
    concurrent_llm_calls: int = Field(
        default=defs.ASSERTION_CONCURRENT_LLM_CALLS,
        description="Number of concurrent LLM calls for batch processing and validation.",
    )
    max_concurrent_questions: int | None = Field(
        default=defs.ASSERTION_MAX_CONCURRENT_GLOBAL_QUESTIONS,
        description="Maximum questions to process in parallel. Set to 1 for sequential.",
    )


class LinkedAssertionConfig(BaseModel):
    """Configuration for linked assertion generation.

    Linked assertions use a simplified approach: single LLM call with all claims,
    then optional validation. No MAP or DEDUPE phases needed since claims are
    already filtered to those relevant to the question.
    """

    max_assertions: int | None = Field(
        default=defs.MAX_ASSERTIONS,
        description="Maximum number of assertions per question. Set to 0 to disable, or None for unlimited.",
    )
    enable_validation: bool = Field(
        default=defs.ENABLE_ASSERTION_VALIDATION,
        description="Whether to validate assertions against sources for quality filtering.",
    )
    min_validation_score: int = Field(
        default=defs.MIN_ASSERTION_VALIDATION_SCORE,
        description="Minimum score (1-5) for grounding, relevance, and verifiability criteria.",
    )
    max_source_count: int = Field(
        default=defs.ASSERTION_MAX_SOURCE_COUNT,
        description="Maximum number of deduplicated sources per assertion. Assertions exceeding this cause the entire question to be dropped.",
    )
    concurrent_llm_calls: int = Field(
        default=defs.ASSERTION_CONCURRENT_LLM_CALLS,
        description="Number of concurrent LLM calls for batch processing and validation.",
    )
    max_concurrent_questions: int | None = Field(
        default=defs.ASSERTION_MAX_CONCURRENT_GLOBAL_QUESTIONS,
        description="Maximum questions to process in parallel. Set to 1 for sequential.",
    )


class AssertionConfig(BaseModel):
    """Configuration for assertion generation (local, global, and linked)."""

    local: LocalAssertionConfig = Field(
        default_factory=LocalAssertionConfig,
        description="Configuration for local assertion generation.",
    )
    global_: GlobalAssertionConfig = Field(
        default_factory=GlobalAssertionConfig,
        alias="global",
        description="Configuration for global assertion generation.",
    )
    linked: LinkedAssertionConfig = Field(
        default_factory=LinkedAssertionConfig,
        description="Configuration for linked assertion generation.",
    )

    model_config: ClassVar[ConfigDict] = {"populate_by_name": True}


class AssertionPromptConfig(BaseModel):
    """Configuration for assertion generation prompts."""

    local_assertion_gen_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ASSERTIONS_PROMPTS_PATH
            / "local_claim_assertion_gen_prompt.txt"
        ),
        description="Prompt for generating local assertions from claims.",
    )
    global_assertion_map_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ASSERTIONS_PROMPTS_PATH
            / "global_claim_assertion_map_prompt.txt"
        ),
        description="Prompt for the map step in global assertion generation.",
    )
    global_assertion_reduce_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ASSERTIONS_PROMPTS_PATH
            / "global_claim_assertion_reduce_prompt.txt"
        ),
        description="Prompt for the reduce step in global assertion generation.",
    )
    local_validation_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ASSERTIONS_PROMPTS_PATH / "local_validation_prompt.txt"
        ),
        description="Prompt for validating local assertions (fact-focused) against sources.",
    )
    global_validation_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ASSERTIONS_PROMPTS_PATH / "global_validation_prompt.txt"
        ),
        description="Prompt for validating global assertions (theme-focused) against sources.",
    )
    linked_assertion_dedupe_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ASSERTIONS_PROMPTS_PATH / "linked_assertion_dedupe_prompt.txt"
        ),
        description="Prompt for deduplicating linked assertions (no synthesis, just filtering).",
    )


class EncodingModelConfig(BaseModel):
    """Configuration for the encoding model used in question generation."""

    model_name: str = Field(
        default=defs.ENCODING_MODEL,
        description="Name of the encoding model to use for chunking documents.",
    )
    chunk_size: int = Field(
        default=defs.CHUNK_SIZE,
        description="Size of each text chunk to be processed by the encoding model.",
    )
    chunk_overlap: int = Field(
        default=defs.CHUNK_OVERLAP,
        description="Overlap size between consecutive text chunks.",
    )


class SamplingConfig(BaseModel):
    """Configuration for data sampling in question generation."""

    num_clusters: int = Field(
        default=defs.NUM_CLUSTERS,
        description="Number of clusters to sample from the dataset.",
    )
    num_samples_per_cluster: int = Field(
        default=defs.NUM_SAMPLES_PER_CLUSTER,
        description="Number of samples to take from each cluster.",
    )
    random_seed: int = Field(
        default=defs.RANDOM_SEED,
        description="Random seed for reproducibility of sampling.",
    )


class ActivityQuestionConfig(QuestionConfig):
    """Configuration for generating activity questions."""

    num_personas: int = Field(
        default=defs.NUM_PERSONAS,
        description="Number of personas to generate questions for.",
    )
    num_tasks_per_persona: int = Field(
        default=defs.NUM_TASKS_PER_PERSONA,
        description="Number of tasks to generate for each persona.",
    )
    num_entities_per_task: int = Field(
        default=defs.NUM_ENTITIES_PER_TASK,
        description="Number of entities to include in each task.",
    )


class DataSummaryPromptConfig(BaseModel):
    """Configuration for the map/reduce summary prompts."""

    summary_map_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOD_PROMPTS_PATH / "summarization/summary_map_system_prompt.txt"
        ),
        description="System prompt for the map summary step in question generation.",
    )
    summary_map_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOD_PROMPTS_PATH / "summarization/summary_map_user_prompt.txt"
        ),
        description="User prompt for the map summary step in question generation.",
    )
    summary_reduce_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOD_PROMPTS_PATH / "summarization/summary_reduce_system_prompt.txt"
        ),
        description="System prompt for the reduce summary step in question generation.",
    )
    summary_reduce_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOD_PROMPTS_PATH / "summarization/summary_reduce_user_prompt.txt"
        ),
        description="User prompt for the reduce summary step in question generation.",
    )


class ActivityContextPromptConfig(BaseModel):
    """Configuration for the activity context prompts."""

    data_summary_prompt_config: DataSummaryPromptConfig = Field(
        default_factory=DataSummaryPromptConfig,
        description="Configuration for the map/reduce summary prompts.",
    )

    entity_extraction_map_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_CONTEXT_PROMPTS_PATH
            / "entity_extraction_map_system_prompt.txt"
        ),
        description="System prompt for extracting entities in the map step.",
    )
    entity_extraction_map_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_CONTEXT_PROMPTS_PATH
            / "entity_extraction_map_user_prompt.txt"
        ),
        description="User prompt for extracting entities in the map step.",
    )
    entity_extraction_reduce_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_CONTEXT_PROMPTS_PATH
            / "entity_extraction_reduce_system_prompt.txt"
        ),
        description="System prompt for extracting entities in the reduce step.",
    )
    entity_extraction_reduce_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_CONTEXT_PROMPTS_PATH
            / "entity_extraction_reduce_user_prompt.txt"
        ),
        description="User prompt for extracting entities in the reduce step.",
    )

    activity_identification_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_CONTEXT_PROMPTS_PATH
            / "activity_identification_prompt.txt"
        ),
        description="Prompt for identifying activities in the question generation process.",
    )


class ActivityGlobalPromptConfig(BaseModel):
    """Configuration for global activity question generation prompts."""

    activity_global_gen_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_GLOBAL_PROMPTS_PATH
            / "activity_global_gen_system_prompt.txt"
        ),
        description="System prompt for generating global questions in question generation.",
    )
    activity_global_gen_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_GLOBAL_PROMPTS_PATH
            / "activity_global_gen_user_prompt.txt"
        ),
        description="User prompt for generating global questions in question generation.",
    )


class ActivityLocalPromptConfig(BaseModel):
    """Configuration for local activity question generation prompts."""

    activity_local_gen_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_LOCAL_PROMPTS_PATH
            / "activity_local_gen_system_prompt.txt"
        ),
        description="System prompt for generating local activity questions.",
    )
    activity_local_gen_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_ACTIVITY_LOCAL_PROMPTS_PATH
            / "activity_local_gen_user_prompt.txt"
        ),
        description="User prompt for generating local activity questions.",
    )


class ActivityQuestionsPromptConfig(BaseModel):
    """Configuration for activity-related prompts."""

    activity_context_prompt_config: ActivityContextPromptConfig = Field(
        default_factory=ActivityContextPromptConfig,
        description="Configuration for the map/reduce summary prompts.",
    )

    activity_global_prompt_config: ActivityGlobalPromptConfig = Field(
        default_factory=ActivityGlobalPromptConfig,
        description="Configuration for global activity question generation prompts.",
    )

    activity_local_prompt_config: ActivityLocalPromptConfig = Field(
        default_factory=ActivityLocalPromptConfig,
        description="Configuration for local activity question generation prompts.",
    )


class DataGlobalPromptConfig(BaseModel):
    """Configuration for global data question generation prompts."""

    data_global_gen_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_GLOBAL_PROMPTS_PATH / "data_global_gen_user_prompt.txt"
        ),
        description="Input prompt for extracting global questions from local questions.",
    )
    data_global_gen_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_GLOBAL_PROMPTS_PATH / "data_global_gen_system_prompt.txt"
        ),
        description="Prompt for extracting global questions from local questions.",
    )


class DataLocalPromptConfig(BaseModel):
    """Configuration for local data question generation prompts."""

    data_local_gen_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LOCAL_PROMPTS_PATH / "data_local_gen_system_prompt.txt"
        ),
        description="Prompt for extracting local questions from input texts.",
    )
    data_local_expansion_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LOCAL_PROMPTS_PATH
            / "data_local_expansion_system_prompt.txt"
        ),
        description="Prompt for generating local questions from input questions.",
    )
    data_local_gen_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LOCAL_PROMPTS_PATH / "data_local_gen_user_prompt.txt"
        ),
        description="Prompt for input texts for local question generation.",
    )


class DataLinkedPromptConfig(BaseModel):
    """Configuration for data-linked question generation prompts."""

    bridge_question_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LINKED_PROMPTS_PATH / "bridge_question_system_prompt.txt"
        ),
        description="System prompt for generating bridge-style linked questions.",
    )
    comparison_question_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LINKED_PROMPTS_PATH
            / "comparison_question_system_prompt.txt"
        ),
        description="System prompt for generating comparison-style linked questions.",
    )
    intersection_question_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LINKED_PROMPTS_PATH
            / "intersection_question_system_prompt.txt"
        ),
        description="System prompt for generating intersection-style linked questions.",
    )
    temporal_question_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LINKED_PROMPTS_PATH
            / "temporal_question_system_prompt.txt"
        ),
        description="System prompt for generating temporal-style linked questions.",
    )
    linked_question_user_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LINKED_PROMPTS_PATH / "linked_question_user_prompt.txt"
        ),
        description="User prompt for linked question generation.",
    )
    batch_validation_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_LINKED_PROMPTS_PATH / "batch_validation_prompt.txt"
        ),
        description="Prompt for batch validation of linked questions.",
    )


class DataQuestionsPromptConfig(BaseModel):
    """Configuration for data-related prompts."""

    claim_extraction_system_prompt: PromptConfig = Field(
        default=PromptConfig(
            prompt=AUTOQ_DATA_PROMPTS_PATH / "claim_extraction_system_prompt.txt"
        ),
        description="System prompt for extracting claims from data tables.",
    )

    data_global_prompt_config: DataGlobalPromptConfig = Field(
        default_factory=DataGlobalPromptConfig,
        description="Configuration for global data question generation prompts.",
    )

    data_local_prompt_config: DataLocalPromptConfig = Field(
        default_factory=DataLocalPromptConfig,
        description="Configuration for local data question generation prompts.",
    )

    data_linked_prompt_config: DataLinkedPromptConfig = Field(
        default_factory=DataLinkedPromptConfig,
        description="Configuration for linked data question generation prompts.",
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

    data_global: DataGlobalQuestionConfig = Field(
        default_factory=DataGlobalQuestionConfig,
        description="Configuration for generating questions from global data.",
    )

    data_linked: DataLinkedQuestionConfig = Field(
        default_factory=DataLinkedQuestionConfig,
        description="Configuration for generating linked questions from local questions.",
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
        default=defs.CONCURRENT_REQUESTS,
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

    activity_questions_prompt_config: ActivityQuestionsPromptConfig = Field(
        default_factory=ActivityQuestionsPromptConfig,
        description="Configuration for activity-related prompts.",
    )

    data_questions_prompt_config: DataQuestionsPromptConfig = Field(
        default_factory=DataQuestionsPromptConfig,
        description="Configuration for data-related prompts.",
    )

    assertions: AssertionConfig = Field(
        default_factory=AssertionConfig,
        description="Configuration for assertion generation.",
    )

    assertion_prompts: AssertionPromptConfig = Field(
        default_factory=AssertionPromptConfig,
        description="Configuration for assertion generation prompts.",
    )


class QuestionType(str):  # noqa: FURB189
    """Enumeration for question types that support assertion regeneration."""

    __slots__ = ()

    DATA_LOCAL = "data_local"
    DATA_GLOBAL = "data_global"
    DATA_LINKED = "data_linked"


class AssertionRegenConfig(BaseModel):
    """Configuration for regenerating assertions for existing questions.

    This config is used to regenerate assertions for questions that were
    previously generated without assertions or with different assertion settings.
    """

    question_type: str = Field(
        ...,
        description="Type of questions to regenerate assertions for: 'data_local', 'data_global', or 'data_linked'.",
    )

    questions_path: Path = Field(
        ...,
        description="Path to the questions JSON file (e.g., selected_questions.json).",
    )

    output_path: Path = Field(
        ...,
        description="Output directory to save regenerated assertions.",
    )

    chat_model: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the LLM to use for assertion generation.",
    )

    assertions: AssertionConfig = Field(
        default_factory=AssertionConfig,
        description="Configuration for assertion generation settings.",
    )

    assertion_prompts: AssertionPromptConfig = Field(
        default_factory=AssertionPromptConfig,
        description="Configuration for assertion generation prompts.",
    )
