# Copyright (c) 2025 Microsoft Corporation.
"""Autoq CLI for generating questions."""

import asyncio
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import tiktoken
import typer
from graphrag_common.config import load_config
from graphrag_storage import Storage
from graphrag_storage.file_storage import FileStorage
from graphrag_storage.storage_factory import create_storage
from graphrag_storage.tables.parquet_table_provider import ParquetTableProvider
from rich import print as rich_print

from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.io.text_unit import load_text_units
from benchmark_qed.autod.sampler.sample_gen import acreate_clustered_sample
from benchmark_qed.autoq.config import (
    ActivityContextPromptConfig,
    ActivityGlobalPromptConfig,
    ActivityLocalPromptConfig,
    AssertionConfig,
    AssertionPromptConfig,
    DataGlobalPromptConfig,
    DataLocalPromptConfig,
    QuestionGenerationConfig,
)
from benchmark_qed.autoq.io.activity import load_activity_context, save_activity_context
from benchmark_qed.autoq.io.question import load_questions, save_questions
from benchmark_qed.autoq.question_gen.activity_questions.context_gen.activity_context_gen import (
    ActivityContextGen,
)
from benchmark_qed.autoq.question_gen.activity_questions.global_question_gen import (
    ActivityGlobalQuestionGen,
)
from benchmark_qed.autoq.question_gen.activity_questions.local_question_gen import (
    ActivityLocalQuestionGen,
)
from benchmark_qed.autoq.question_gen.data_questions.global_question_gen import (
    DataGlobalQuestionGen,
)
from benchmark_qed.autoq.question_gen.data_questions.local_question_gen import (
    DataLocalQuestionGen,
)
from benchmark_qed.llm.factory import ModelFactory
from benchmark_qed.llm.type.base import ChatModel

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)


class GenerationType(StrEnum):
    """Enumeration for the scope of question generation."""

    data_local = "data_local"
    data_global = "data_global"
    activity_local = "activity_local"
    activity_global = "activity_global"


async def __generate_data_local(
    output_storage: Storage,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    num_questions: int,
    oversample_factor: float,
    random_seed: int,
    concurrent_requests: int,
    config: DataLocalPromptConfig,
    assertion_config: AssertionConfig,
    assertion_prompt_config: AssertionPromptConfig,
    llm_params: dict[str, Any],
) -> None:
    table_provider = ParquetTableProvider(output_storage)
    sample_texts_df = await table_provider.read_dataframe("sample_texts")
    sample_texts = load_text_units(df=sample_texts_df)

    data_local_generator = DataLocalQuestionGen(
        llm=llm,
        text_embedder=text_embedder,
        text_units=sample_texts,
        concurrent_coroutines=concurrent_requests,
        random_seed=random_seed,
        llm_params=llm_params,
        generation_system_prompt=config.data_local_gen_system_prompt.template,
        generation_user_prompt=config.data_local_gen_user_prompt.template,
        expansion_system_prompt=config.data_local_expansion_system_prompt.template,
        assertion_config=assertion_config,
        assertion_prompt_config=assertion_prompt_config,
    )

    data_local_question_results = await data_local_generator.agenerate(
        num_questions=num_questions,
        oversample_factor=oversample_factor,
    )

    # save both candidate questions and the final selected questions
    data_local_storage = output_storage.child("data_local_questions")
    await save_questions(
        data_local_question_results.selected_questions,
        data_local_storage,
        "selected_questions",
    )
    await save_questions(
        data_local_question_results.selected_questions,
        data_local_storage,
        "selected_questions_text",
        question_text_only=True,
    )
    await save_questions(
        data_local_question_results.candidate_questions,
        data_local_storage,
        "candidate_questions",
    )


async def __generate_data_global(
    output_storage: Storage,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    num_questions: int,
    oversample_factor: float,
    min_questions_in_context: int,
    random_seed: int,
    concurrent_requests: int,
    config: DataGlobalPromptConfig,
    assertion_config: AssertionConfig,
    assertion_prompt_config: AssertionPromptConfig,
    llm_params: dict[str, Any],
) -> None:
    data_local_storage = output_storage.child("data_local_questions")
    if not await data_local_storage.has("candidate_questions.json"):
        rich_print(
            "Local candidate questions not found. Please run data_local generation first."
        )
        return

    local_questions = await load_questions(
        data_local_storage, "candidate_questions.json"
    )

    data_global_generator = DataGlobalQuestionGen(
        llm=llm,
        text_embedder=text_embedder,
        local_questions=local_questions,
        concurrent_coroutines=concurrent_requests,
        random_seed=random_seed,
        llm_params=llm_params,
        generation_system_prompt=config.data_global_gen_system_prompt.template,
        generation_user_prompt=config.data_global_gen_user_prompt.template,
        assertion_config=assertion_config,
        assertion_prompt_config=assertion_prompt_config,
        min_questions_in_context=min_questions_in_context,
    )

    data_global_question_results = await data_global_generator.agenerate(
        num_questions=num_questions,
        oversample_factor=oversample_factor,
    )

    # save both candidate questions and the final selected questions
    data_global_storage = output_storage.child("data_global_questions")
    await save_questions(
        data_global_question_results.selected_questions,
        data_global_storage,
        "selected_questions",
    )
    await save_questions(
        data_global_question_results.selected_questions,
        data_global_storage,
        "selected_questions_text",
        question_text_only=True,
    )
    await save_questions(
        data_global_question_results.candidate_questions,
        data_global_storage,
        "candidate_questions",
    )


async def __generate_activity_context(
    output_storage: Storage,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    token_encoder: tiktoken.Encoding,
    num_personas: int,
    num_tasks_per_persona: int,
    num_entities_per_task: int,
    oversample_factor: float,
    concurrent_requests: int,
    config: ActivityContextPromptConfig,
    llm_params: dict[str, Any],
    use_representative_samples_only: bool = True,
    skip_warning: bool = False,
) -> None:
    context_storage = output_storage.child("context")
    if (await context_storage.has("activity_context_full.json")) and not skip_warning:
        rich_print(
            "Activity context already exists. Skipping generation.\n"
            "[bold yellow]If you want to generate a new context, delete context folder from output.[/bold yellow]"
        )
        return
    table_provider = ParquetTableProvider(output_storage)
    sample_texts_df = await table_provider.read_dataframe("sample_texts")
    sample_texts = load_text_units(
        df=sample_texts_df, attributes_cols=["is_representative"]
    )

    activity_generator = ActivityContextGen(
        llm=llm,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
        text_units=sample_texts,
        concurrent_coroutines=concurrent_requests,
        llm_params=llm_params,
        activity_identification_prompt=config.activity_identification_prompt.template,
        map_system_prompt=config.data_summary_prompt_config.summary_map_system_prompt.template,
        map_user_prompt=config.data_summary_prompt_config.summary_map_user_prompt.template,
        reduce_system_prompt=config.data_summary_prompt_config.summary_reduce_system_prompt.template,
        reduce_user_prompt=config.data_summary_prompt_config.summary_reduce_user_prompt.template,
    )

    activity_context = await activity_generator.agenerate(
        num_personas=num_personas,
        num_tasks=num_tasks_per_persona,
        num_entities_per_task=num_entities_per_task,
        oversample_factor=oversample_factor,
        use_representative_samples_only=use_representative_samples_only,
    )

    await save_activity_context(activity_context, context_storage)


async def __generate_activity_local(
    output_storage: Storage,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    num_questions: int,
    oversample_factor: float,
    random_seed: int,
    concurrent_requests: int,
    config: ActivityLocalPromptConfig,
    llm_params: dict[str, Any],
) -> None:
    context_storage = output_storage.child("context")
    activity_context = await load_activity_context(
        context_storage, "activity_context_full.json"
    )

    # Use PromptConfig.template property for all prompt templates
    activity_local_generator = ActivityLocalQuestionGen(
        llm=llm,
        text_embedder=text_embedder,
        activity_context=activity_context,
        concurrent_coroutines=concurrent_requests,
        random_seed=random_seed,
        llm_params=llm_params,
        generation_system_prompt=config.activity_local_gen_system_prompt.template,
        generation_user_prompt=config.activity_local_gen_user_prompt.template,
    )

    activity_local_question_results = await activity_local_generator.agenerate(
        num_questions=num_questions,
        oversample_factor=oversample_factor,
    )

    # save both candidate questions and the final selected questions
    activity_local_storage = output_storage.child("activity_local_questions")
    await save_questions(
        activity_local_question_results.selected_questions,
        activity_local_storage,
        "selected_questions",
    )
    await save_questions(
        activity_local_question_results.selected_questions,
        activity_local_storage,
        "selected_questions_text",
        question_text_only=True,
    )
    await save_questions(
        activity_local_question_results.candidate_questions,
        activity_local_storage,
        "candidate_questions",
    )


async def __generate_activity_global(
    output_storage: Storage,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    num_questions: int,
    oversample_factor: float,
    random_seed: int,
    concurrent_requests: int,
    config: ActivityGlobalPromptConfig,
    llm_params: dict[str, Any],
) -> None:
    context_storage = output_storage.child("context")
    activity_context = await load_activity_context(
        context_storage, "activity_context_full.json"
    )

    # Use PromptConfig.template property for all prompt templates
    activity_global_generator = ActivityGlobalQuestionGen(
        llm=llm,
        text_embedder=text_embedder,
        activity_context=activity_context,
        concurrent_coroutines=concurrent_requests,
        random_seed=random_seed,
        llm_params=llm_params,
        generation_system_prompt=config.activity_global_gen_system_prompt.template,
        generation_user_prompt=config.activity_global_gen_user_prompt.template,
    )

    activity_global_question_results = await activity_global_generator.agenerate(
        num_questions=num_questions,
        oversample_factor=oversample_factor,
    )

    # save both candidate questions and the final selected questions
    activity_global_storage = output_storage.child("activity_global_questions")
    await save_questions(
        activity_global_question_results.selected_questions,
        activity_global_storage,
        "selected_questions",
    )
    await save_questions(
        activity_global_question_results.selected_questions,
        activity_global_storage,
        "selected_questions_text",
        question_text_only=True,
    )
    await save_questions(
        activity_global_question_results.candidate_questions,
        activity_global_storage,
        "candidate_questions",
    )


SCOPE_SOURCE_MAPPING: dict[Any, Any] = {
    GenerationType.activity_local: __generate_activity_local,
    GenerationType.activity_global: __generate_activity_global,
    GenerationType.data_local: __generate_data_local,
    GenerationType.data_global: __generate_data_global,
}


async def __create_clustered_sample(
    input_data_path: Path,
    output_storage: Storage,
    text_embedder: TextEmbedder,
    num_clusters: int,
    num_samples_per_cluster: int,
    input_type: str,
    text_column: str,
    metadata_columns: list[str] | None,
    file_encoding: str,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    random_seed: int = 42,
    input_storage: Storage | None = None,
) -> None:
    if await output_storage.has("sample_texts.parquet"):
        rich_print(
            "Sample files already exist. Skipping sampling step.\n"
            "[bold yellow]If you want to generate a new sample, delete sample_texts.parquet from the output.[/bold yellow]"
        )
        return

    await acreate_clustered_sample(
        input_path=input_data_path.as_posix(),
        output_path="",
        text_embedder=text_embedder,
        num_clusters=num_clusters,
        num_samples_per_cluster=num_samples_per_cluster,
        input_type=input_type,
        text_tag=text_column,
        metadata_tags=metadata_columns,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        file_encoding=file_encoding,
        token_encoding=model_name,
        random_seed=random_seed,
        input_storage=input_storage,
        output_storage=output_storage,
    )


@app.command()
def autoq(
    configuration_path: Annotated[
        Path,
        typer.Argument(help="The path to the file containing the configuration."),
    ],
    output_data_path: Annotated[
        Path, typer.Argument(help="The path to the output folder for the results.")
    ],
    generation_types: Annotated[
        list[GenerationType] | None,
        typer.Option(help="The source of the question generation."),
    ] = None,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics after scoring."),
    ] = False,
) -> None:
    """Generate questions from the input data."""
    config = load_config(QuestionGenerationConfig, configuration_path)

    if generation_types is None:
        generation_types = [
            GenerationType.data_local,
            GenerationType.data_global,
            GenerationType.activity_local,
            GenerationType.activity_global,
        ]

    embedding_model = ModelFactory.create_embedding_model(config.embedding_model)

    text_embedder = TextEmbedder(embedding_model)
    chat_model = ModelFactory.create_chat_model(config.chat_model)
    token_encoder = tiktoken.get_encoding(config.encoding.model_name)
    loop = asyncio.get_event_loop()

    # Create storage for output
    if config.output_storage:
        output_storage = create_storage(config.output_storage)
        output_posix = output_data_path.as_posix().strip("./")
        if output_posix:
            output_storage = output_storage.child(output_posix)
    else:
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_storage = FileStorage(base_dir=str(output_data_path))

    # Create storage for input (if configured)
    if config.input.storage:
        base_storage = create_storage(config.input.storage)
        # Use dataset_path as a sub-path within the storage container
        dataset_posix = config.input.dataset_path.as_posix().strip("./")
        input_storage = (
            base_storage.child(dataset_posix) if dataset_posix else base_storage
        )
    else:
        input_storage = None

    # Log assertion generation status
    local_assertions_enabled = (
        config.assertions.local.max_assertions is None
        or config.assertions.local.max_assertions > 0
    )
    global_assertions_enabled = (
        config.assertions.global_.max_assertions is None
        or config.assertions.global_.max_assertions > 0
    )
    if local_assertions_enabled or global_assertions_enabled:
        if (
            config.assertions.local.enable_validation
            or config.assertions.global_.enable_validation
        ):
            rich_print(
                f"Assertion generation enabled with validation (local min score: {config.assertions.local.min_validation_score}/5, global min score: {config.assertions.global_.min_validation_score}/5)"
            )
        else:
            rich_print("Assertion generation enabled (validation disabled)")
    else:
        rich_print("Assertion generation disabled")

    rich_print("Creating clustered sample from the input data...")
    loop.run_until_complete(
        __create_clustered_sample(
            input_data_path=config.input.dataset_path,
            output_storage=output_storage,
            text_embedder=text_embedder,
            num_clusters=config.sampling.num_clusters,
            num_samples_per_cluster=config.sampling.num_samples_per_cluster,
            input_type=config.input.input_type,
            text_column=config.input.text_column,
            metadata_columns=config.input.metadata_columns,
            file_encoding=config.input.file_encoding,
            chunk_size=config.encoding.chunk_size,
            chunk_overlap=config.encoding.chunk_overlap,
            model_name=config.encoding.model_name,
            random_seed=config.sampling.random_seed,
            input_storage=input_storage,
        )
    )
    first_activity = True
    for generation_type in generation_types:
        rich_print(f"Generating questions for {generation_type}...")
        if generation_type in [
            GenerationType.activity_local,
            GenerationType.activity_global,
        ]:
            activity_config = (
                config.activity_local
                if generation_type == GenerationType.activity_local
                else config.activity_global
            )
            loop.run_until_complete(
                __generate_activity_context(
                    output_storage=output_storage,
                    llm=chat_model,
                    text_embedder=text_embedder,
                    token_encoder=token_encoder,
                    num_personas=activity_config.num_personas,
                    num_tasks_per_persona=activity_config.num_tasks_per_persona,
                    num_entities_per_task=activity_config.num_entities_per_task,
                    oversample_factor=activity_config.oversample_factor,
                    concurrent_requests=config.concurrent_requests,
                    config=config.activity_questions_prompt_config.activity_context_prompt_config,
                    llm_params=config.chat_model.call_args,
                    skip_warning=not first_activity,
                )
            )
            activity_fn = SCOPE_SOURCE_MAPPING[generation_type]
            activity_fn_kwargs: dict[str, Any] = {
                "output_storage": output_storage,
                "llm": chat_model,
                "text_embedder": text_embedder,
                "num_questions": activity_config.num_questions,
                "oversample_factor": activity_config.oversample_factor,
                "random_seed": config.sampling.random_seed,
                "concurrent_requests": config.concurrent_requests,
                "config": config.activity_questions_prompt_config.activity_local_prompt_config
                if generation_type == GenerationType.activity_local
                else config.activity_questions_prompt_config.activity_global_prompt_config,
                "llm_params": config.chat_model.call_args,
            }
            loop.run_until_complete(activity_fn(**activity_fn_kwargs))
            first_activity = False
        else:
            data_config = (
                config.data_local
                if generation_type == GenerationType.data_local
                else config.data_global
            )
            data_fn = SCOPE_SOURCE_MAPPING[generation_type]
            # Build kwargs for the data function
            data_kwargs: dict[str, Any] = {
                "output_storage": output_storage,
                "llm": chat_model,
                "text_embedder": text_embedder,
                "num_questions": data_config.num_questions,
                "oversample_factor": data_config.oversample_factor,
                "random_seed": config.sampling.random_seed,
                "concurrent_requests": config.concurrent_requests,
                "config": config.data_questions_prompt_config.data_local_prompt_config
                if generation_type == GenerationType.data_local
                else config.data_questions_prompt_config.data_global_prompt_config,
                "assertion_config": config.assertions,
                "assertion_prompt_config": config.assertion_prompts,
                "llm_params": config.chat_model.call_args,
            }
            # Add min_questions_in_context only for data_global
            if generation_type == GenerationType.data_global:
                data_kwargs["min_questions_in_context"] = (
                    config.data_global.min_questions_in_context
                )
            loop.run_until_complete(data_fn(**data_kwargs))

    if print_model_usage:
        rich_print("Chat Model usage statistics:")
        rich_print(chat_model.get_usage())

        rich_print("Embedding Model usage statistics:")
        rich_print(embedding_model.get_usage())
