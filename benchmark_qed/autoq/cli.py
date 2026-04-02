# Copyright (c) 2025 Microsoft Corporation.
"""Autoq CLI for generating questions."""

import asyncio
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import tiktoken
import typer
from rich import print as rich_print

from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.io.enums import InputDataType
from benchmark_qed.autod.io.text_unit import load_text_units
from benchmark_qed.autod.sampler.sample_gen import acreate_clustered_sample
from benchmark_qed.autoq.config import (
    ActivityContextPromptConfig,
    ActivityGlobalPromptConfig,
    ActivityLocalPromptConfig,
    AssertionConfig,
    AssertionPromptConfig,
    DataGlobalPromptConfig,
    DataLinkedPromptConfig,
    DataLocalPromptConfig,
    QuestionGenerationConfig,
)
from benchmark_qed.autoq.data_model.activity import ActivityContext
from benchmark_qed.autoq.io.activity import save_activity_context
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
from benchmark_qed.autoq.question_gen.data_questions.linked_question_gen import (
    DataLinkedQuestionGen,
)
from benchmark_qed.autoq.question_gen.data_questions.local_question_gen import (
    DataLocalQuestionGen,
)
from benchmark_qed.config.utils import load_config
from benchmark_qed.llm.factory import ModelFactory
from benchmark_qed.llm.type.base import ChatModel

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)


class GenerationType(StrEnum):
    """Enumeration for the scope of question generation."""

    data_local = "data_local"
    data_global = "data_global"
    data_linked = "data_linked"
    activity_local = "activity_local"
    activity_global = "activity_global"


async def __generate_data_local(
    output_data_path: Path,
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
    sample_texts_df = pd.read_parquet(f"{output_data_path}/sample_texts.parquet")
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
    save_questions(
        data_local_question_results.selected_questions,
        f"{output_data_path}/data_local_questions/",
        "selected_questions",
    )
    save_questions(
        data_local_question_results.selected_questions,
        f"{output_data_path}/data_local_questions/",
        "selected_questions_text",
        question_text_only=True,
    )
    save_questions(
        data_local_question_results.candidate_questions,
        f"{output_data_path}/data_local_questions/",
        "candidate_questions",
        save_assertions=False,  # Only save assertions for selected questions
    )


async def __generate_data_global(
    output_data_path: Path,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    num_questions: int,
    oversample_factor: float,
    min_questions_in_context: int,
    min_claim_count: int,
    min_relevant_reference_count: int,
    enable_question_validation: bool,
    random_seed: int,
    concurrent_requests: int,
    config: DataGlobalPromptConfig,
    assertion_config: AssertionConfig,
    assertion_prompt_config: AssertionPromptConfig,
    llm_params: dict[str, Any],
) -> None:
    if not (
        output_data_path / "data_local_questions" / "candidate_questions.json"
    ).exists():
        rich_print(
            "Local candidate questions not found. Please run data_local generation first."
        )
        return

    local_questions = load_questions(
        f"{output_data_path}/data_local_questions/candidate_questions.json"
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
        min_claim_count=min_claim_count,
        min_relevant_reference_count=min_relevant_reference_count,
        enable_question_validation=enable_question_validation,
    )

    data_global_question_results = await data_global_generator.agenerate(
        num_questions=num_questions,
        oversample_factor=oversample_factor,
    )

    # save both candidate questions and the final selected questions
    save_questions(
        data_global_question_results.selected_questions,
        f"{output_data_path}/data_global_questions/",
        "selected_questions",
    )
    save_questions(
        data_global_question_results.selected_questions,
        f"{output_data_path}/data_global_questions/",
        "selected_questions_text",
        question_text_only=True,
    )
    save_questions(
        data_global_question_results.candidate_questions,
        f"{output_data_path}/data_global_questions/",
        "candidate_questions",
        save_assertions=False,  # Only save assertions for selected questions
    )


async def __generate_data_linked(
    output_data_path: Path,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    num_questions: int,
    oversample_factor: float,
    min_questions_per_entity: int,
    max_questions_per_entity: int,
    type_balance_weight: float,
    random_seed: int,
    concurrent_requests: int,
    config: DataLinkedPromptConfig,
    assertion_config: AssertionConfig,
    assertion_prompt_config: AssertionPromptConfig,
    llm_params: dict[str, Any],
) -> None:
    """Generate data-linked questions from local questions sharing named entities."""
    if not (
        output_data_path / "data_local_questions" / "candidate_questions.json"
    ).exists():
        rich_print(
            "Local candidate questions not found. Please run data_local generation first."
        )
        return

    local_questions = load_questions(
        f"{output_data_path}/data_local_questions/candidate_questions.json"
    )

    data_linked_generator = DataLinkedQuestionGen(
        llm=llm,
        text_embedder=text_embedder,
        local_questions=local_questions,
        concurrent_coroutines=concurrent_requests,
        random_seed=random_seed,
        llm_params=llm_params,
        assertion_config=assertion_config,
        assertion_prompt_config=assertion_prompt_config,
        min_questions_per_entity=min_questions_per_entity,
        max_questions_per_entity=max_questions_per_entity,
        type_balance_weight=type_balance_weight,
        prompt_config=config,
    )

    data_linked_question_results = await data_linked_generator.agenerate(
        num_questions=num_questions,
        oversample_factor=oversample_factor,
    )

    # save both candidate questions and the final selected questions
    save_questions(
        data_linked_question_results.selected_questions,
        f"{output_data_path}/data_linked_questions/",
        "selected_questions",
    )
    save_questions(
        data_linked_question_results.selected_questions,
        f"{output_data_path}/data_linked_questions/",
        "selected_questions_text",
        question_text_only=True,
    )
    save_questions(
        data_linked_question_results.candidate_questions,
        f"{output_data_path}/data_linked_questions/",
        "candidate_questions",
        save_assertions=False,  # Only save assertions for selected questions
    )

    # Save question stats (includes pipeline stats)
    if hasattr(data_linked_question_results, "pipeline_stats"):
        import json

        stats = data_linked_question_results.pipeline_stats  # type: ignore[attr-defined]
        stats_path = Path(
            f"{output_data_path}/data_linked_questions/question_stats.json"
        )
        stats_path.write_text(json.dumps(stats, indent=2))

        # Print summary stats
        rich_print("\n[bold]Data-Linked Question Generation Summary:[/bold]")
        rich_print(f"  Entity groups: {stats.get('entity_groups', 'N/A')}")
        rich_print(f"  Generated: {stats.get('generated', 'N/A')}")
        rich_print(
            f"  After batch validation: {stats.get('after_batch_validation', 'N/A')} (filtered {stats.get('batch_validation_filtered', 0)})"
        )
        rich_print(
            f"  After assertion filter: {stats.get('after_assertion_filter', 'N/A')} (filtered {stats.get('assertion_filter_removed', 0)})"
        )
        rich_print(f"  Selected: {stats.get('selected', 'N/A')}")
        if "type_distribution" in stats:
            rich_print(f"  Type distribution: {stats['type_distribution']}")
        if "quality_scores" in stats:
            qs = stats["quality_scores"]
            rich_print(
                f"  Quality scores: min={qs.get('min', 'N/A'):.2f}, max={qs.get('max', 'N/A'):.2f}, avg={qs.get('avg', 'N/A'):.2f}"
            )


async def __generate_activity_context(
    output_data_path: Path,
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
    if (
        output_data_path / "context" / "activity_context_full.json"
    ).exists() and not skip_warning:
        rich_print(
            "Activity context already exists. Skipping generation.\n"
            f"[bold yellow]If you want to generate a new context, delete context folder from {output_data_path}.[/bold yellow]"
        )
        return
    sample_texts_df = pd.read_parquet(f"{output_data_path}/sample_texts.parquet")
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

    save_activity_context(activity_context, f"{output_data_path}/context/")


async def __generate_activity_local(
    output_data_path: Path,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    num_questions: int,
    oversample_factor: float,
    random_seed: int,
    concurrent_requests: int,
    config: ActivityLocalPromptConfig,
    llm_params: dict[str, Any],
) -> None:
    activity_context = ActivityContext(
        **json.loads(
            (output_data_path / "context" / "activity_context_full.json").read_text()
        )
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
    save_questions(
        activity_local_question_results.selected_questions,
        f"{output_data_path}/activity_local_questions/",
        "selected_questions",
    )
    save_questions(
        activity_local_question_results.selected_questions,
        f"{output_data_path}/activity_local_questions/",
        "selected_questions_text",
        question_text_only=True,
    )
    save_questions(
        activity_local_question_results.candidate_questions,
        f"{output_data_path}/activity_local_questions/",
        "candidate_questions",
        save_assertions=False,  # Only save assertions for selected questions
    )


async def __generate_activity_global(
    output_data_path: Path,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    num_questions: int,
    oversample_factor: float,
    random_seed: int,
    concurrent_requests: int,
    config: ActivityGlobalPromptConfig,
    llm_params: dict[str, Any],
) -> None:
    activity_context = ActivityContext(
        **json.loads(
            (output_data_path / "context" / "activity_context_full.json").read_text()
        )
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
    save_questions(
        activity_global_question_results.selected_questions,
        f"{output_data_path}/activity_global_questions/",
        "selected_questions",
    )
    save_questions(
        activity_global_question_results.selected_questions,
        f"{output_data_path}/activity_global_questions/",
        "selected_questions_text",
        question_text_only=True,
    )
    save_questions(
        activity_global_question_results.candidate_questions,
        f"{output_data_path}/activity_global_questions/",
        "candidate_questions",
        save_assertions=False,  # Only save assertions for selected questions
    )


SCOPE_SOURCE_MAPPING: dict[Any, Any] = {
    GenerationType.activity_local: __generate_activity_local,
    GenerationType.activity_global: __generate_activity_global,
    GenerationType.data_local: __generate_data_local,
    GenerationType.data_global: __generate_data_global,
    GenerationType.data_linked: __generate_data_linked,
}


async def __create_clustered_sample(
    input_data_path: Path,
    output_data_path: Path,
    text_embedder: TextEmbedder,
    num_clusters: int,
    num_samples_per_cluster: int,
    input_type: InputDataType,
    text_column: str,
    metadata_columns: list[str] | None,
    file_encoding: str,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    random_seed: int = 42,
) -> None:
    if (output_data_path / "sample_texts.parquet").exists():
        rich_print(
            "Sample files already exist. Skipping sampling step.\n"
            f"[bold yellow]If you want to generate a new sample, delete sample_texts.parquet from {output_data_path}.[/bold yellow]"
        )
        return

    await acreate_clustered_sample(
        input_path=input_data_path.as_posix(),
        output_path=output_data_path.as_posix(),
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
    config = load_config(configuration_path, QuestionGenerationConfig)

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

    # Only create clustered sample if needed
    # - data_local needs sample_texts.parquet
    # - activity_* needs sample_texts.parquet
    # - data_global and data_linked use local questions (don't need sample)
    needs_sample = any(
        gt in generation_types
        for gt in [
            GenerationType.data_local,
            GenerationType.activity_local,
            GenerationType.activity_global,
        ]
    )
    if needs_sample:
        rich_print("Creating clustered sample from the input data...")
        loop.run_until_complete(
            __create_clustered_sample(
                input_data_path=config.input.dataset_path,
                output_data_path=output_data_path,
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
                    output_data_path=output_data_path,
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
                "output_data_path": output_data_path,
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
            # Handle data question types (data_local, data_global, data_linked)
            if generation_type == GenerationType.data_local:
                data_config = config.data_local
                prompt_config = (
                    config.data_questions_prompt_config.data_local_prompt_config
                )
            elif generation_type == GenerationType.data_global:
                data_config = config.data_global
                prompt_config = (
                    config.data_questions_prompt_config.data_global_prompt_config
                )
            else:  # data_linked
                data_config = config.data_linked
                prompt_config = (
                    config.data_questions_prompt_config.data_linked_prompt_config
                )

            data_fn = SCOPE_SOURCE_MAPPING[generation_type]
            # Build kwargs for the data function
            data_kwargs: dict[str, Any] = {
                "output_data_path": output_data_path,
                "llm": chat_model,
                "text_embedder": text_embedder,
                "num_questions": data_config.num_questions,
                "oversample_factor": data_config.oversample_factor,
                "random_seed": config.sampling.random_seed,
                "concurrent_requests": config.concurrent_requests,
                "config": prompt_config,
                "assertion_config": config.assertions,
                "assertion_prompt_config": config.assertion_prompts,
                "llm_params": config.chat_model.call_args,
            }
            # Add min_questions_in_context only for data_global
            if generation_type == GenerationType.data_global:
                data_kwargs["min_questions_in_context"] = (
                    config.data_global.min_questions_in_context
                )
                data_kwargs["min_claim_count"] = config.data_global.min_claim_count
                data_kwargs["min_relevant_reference_count"] = (
                    config.data_global.min_relevant_reference_count
                )
                data_kwargs["enable_question_validation"] = (
                    config.data_global.enable_question_validation
                )
            # Add entity grouping params only for data_linked
            if generation_type == GenerationType.data_linked:
                data_kwargs["min_questions_per_entity"] = (
                    config.data_linked.min_questions_per_entity
                )
                data_kwargs["max_questions_per_entity"] = (
                    config.data_linked.max_questions_per_entity
                )
                data_kwargs["type_balance_weight"] = (
                    config.data_linked.type_balance_weight
                )
            loop.run_until_complete(data_fn(**data_kwargs))

    if print_model_usage:
        rich_print("Chat Model usage statistics:")
        rich_print(chat_model.get_usage())

        rich_print("Embedding Model usage statistics:")
        rich_print(embedding_model.get_usage())


@app.command(name="assertion-stats")
def assertion_stats(
    assertions_path: Annotated[
        Path,
        typer.Argument(
            help="Path to assertion JSON file or directory containing assertion files."
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to save stats JSON. If not specified, saves as {input}_stats.json.",
        ),
    ] = None,
    assertion_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Type of assertions: 'global', 'map', or 'local'. If not specified, inferred from filename.",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress printing stats to console.",
        ),
    ] = False,
) -> None:
    """Generate statistics for assertion files.

    Computes and saves statistics including:
    - Total assertions and questions
    - Assertions per question (mean, std, min, max)
    - Sources per assertion (mean, std, min, max)
    - Supporting assertions per global assertion (mean, std, min, max)
    - Score distribution

    Examples
    --------
        # Generate stats for a single assertion file
        benchmark-qed autoq assertion-stats output/assertions.json

        # Generate stats for all assertion files in a directory
        benchmark-qed autoq assertion-stats output/data_global_questions/

        # Specify output path
        benchmark-qed autoq assertion-stats assertions.json -o stats/my_stats.json
    """
    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.stats import (
        generate_stats_for_assertion_file,
        generate_stats_for_directory,
    )

    if assertions_path.is_dir():
        # Process directory
        output_dir = output_path or assertions_path
        results = generate_stats_for_directory(
            directory=assertions_path,
            output_dir=output_dir,
            print_stats=not quiet,
        )
        if not results:
            rich_print("[yellow]No assertion files found in directory.[/yellow]")
        else:
            rich_print(f"\n[green]Generated stats for {len(results)} file(s).[/green]")
    else:
        # Process single file
        generate_stats_for_assertion_file(
            assertions_path=assertions_path,
            output_path=output_path,
            assertion_type=assertion_type,
            print_stats=not quiet,
        )
        rich_print(
            f"[green]Stats saved to: {output_path or assertions_path.parent / f'{assertions_path.stem}_stats.json'}[/green]"
        )


class AssertionType(StrEnum):
    """Enumeration for the type of assertion generation."""

    local = "local"
    global_ = "global"
    linked = "linked"


async def __generate_assertions_for_questions(
    questions: list,
    assertion_type: AssertionType,
    llm: ChatModel,
    text_embedder: TextEmbedder,
    assertion_config: AssertionConfig,
    assertion_prompt_config: AssertionPromptConfig,
    llm_params: dict[str, Any],
) -> list:
    """Generate assertions for a list of questions.

    Args:
        questions: List of Question objects to generate assertions for.
        assertion_type: Type of assertions to generate (local, global, link).
        llm: Chat model for LLM calls.
        text_embedder: Text embedder for semantic operations.
        assertion_config: Configuration for assertion generation.
        assertion_prompt_config: Prompts for assertion generation.
        llm_params: Parameters for LLM calls.

    Returns
    -------
        List of Question objects with assertions added to attributes.
    """
    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.global_claim_assertion_gen import (
        GlobalClaimAssertionGenerator,
    )
    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.local_claim_assertion_gen import (
        LocalClaimAssertionGenerator,
    )
    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.validator import (
        AssertionValidator,
    )

    # Select the appropriate config based on assertion type
    if assertion_type == AssertionType.local:
        config = assertion_config.local
        system_prompt = assertion_prompt_config.local_assertion_gen_prompt.template
        validation_prompt = assertion_prompt_config.local_validation_prompt.template
    elif assertion_type == AssertionType.linked:
        config = assertion_config.linked
        system_prompt = assertion_prompt_config.local_assertion_gen_prompt.template
        validation_prompt = assertion_prompt_config.local_validation_prompt.template
    else:  # global
        config = assertion_config.global_
        system_prompt = None  # Global uses map/reduce prompts
        validation_prompt = assertion_prompt_config.global_validation_prompt.template

    max_assertions = config.max_assertions
    if max_assertions is not None and max_assertions == 0:
        rich_print("[yellow]Assertion generation disabled (max_assertions=0)[/yellow]")
        return questions

    # Create validator if enabled
    validator = None
    if config.enable_validation:
        validator = AssertionValidator(
            llm=llm,
            llm_params=llm_params,
            min_criterion_score=config.min_validation_score,
            validation_prompt=validation_prompt,
            concurrent_validations=config.concurrent_llm_calls,
            max_source_count=config.max_source_count,
        )

    # Create assertion generator based on type
    if assertion_type in [AssertionType.local, AssertionType.linked]:
        generator = LocalClaimAssertionGenerator(
            llm=llm,
            llm_params=llm_params,
            max_assertions=max_assertions,
            validator=validator,
            system_prompt=system_prompt,
            max_concurrent_questions=config.max_concurrent_questions,
        )
    else:  # global
        # Global needs separate map and reduce validators
        global_config = assertion_config.global_
        map_validator = None
        reduce_validator = None
        if global_config.enable_validation:
            map_validator = AssertionValidator(
                llm=llm,
                llm_params=llm_params,
                min_criterion_score=global_config.min_validation_score,
                validation_prompt=assertion_prompt_config.local_validation_prompt.template,
                concurrent_validations=global_config.concurrent_llm_calls,
                max_source_count=global_config.max_source_count,
            )
            reduce_validator = validator

        generator = GlobalClaimAssertionGenerator(
            llm=llm,
            llm_params=llm_params,
            token_encoder=tiktoken.get_encoding("cl100k_base"),
            max_assertions=global_config.max_assertions,
            map_validator=map_validator,
            reduce_validator=reduce_validator,
            batch_size=global_config.batch_size,
            reduce_data_tokens=global_config.reduce_data_tokens,
            map_data_tokens=global_config.map_data_tokens,
            concurrent_coroutines=global_config.concurrent_llm_calls,
            max_concurrent_questions=global_config.max_concurrent_questions,
            map_system_prompt=assertion_prompt_config.global_assertion_map_prompt.template,
            reduce_system_prompt=assertion_prompt_config.global_assertion_reduce_prompt.template,
            text_embedder=text_embedder,
            enable_semantic_grouping=global_config.enable_semantic_grouping,
            validate_map_assertions=global_config.validate_map_assertions,
            validate_reduce_assertions=global_config.validate_reduce_assertions,
        )

    # Generate assertions
    await generator.agenerate_assertions_for_questions(questions)
    return questions


@app.command(name="generate-assertions")
def generate_assertions(
    configuration_path: Annotated[
        Path,
        typer.Argument(help="Path to the settings.yaml configuration file."),
    ],
    questions_path: Annotated[
        Path,
        typer.Argument(
            help="Path to questions JSON file (e.g., candidate_questions.json)."
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(help="Output directory to save questions with assertions."),
    ],
    assertion_type: Annotated[
        AssertionType,
        typer.Option(
            "--type",
            "-t",
            help="Type of assertions to generate: 'local', 'global', or 'linked'.",
        ),
    ] = AssertionType.local,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics."),
    ] = False,
) -> None:
    r"""Generate assertions for existing questions.

    This command loads questions from a JSON file and generates assertions
    using the specified assertion type configuration from settings.yaml.

    Examples
    --------
        # Generate local assertions for candidate questions
        benchmark-qed autoq generate-assertions settings.yaml \
            output/data_local_questions/candidate_questions.json \
            output/data_local_questions/ --type local

        # Generate global assertions
        benchmark-qed autoq generate-assertions settings.yaml \
            output/data_global_questions/candidate_questions.json \
            output/data_global_questions/ --type global

        # Generate linked assertions
        benchmark-qed autoq generate-assertions settings.yaml \
            output/data_linked_questions/candidate_questions.json \
            output/data_linked_questions/ --type linked
    """
    config = load_config(configuration_path, QuestionGenerationConfig)

    # Load questions
    if not questions_path.exists():
        rich_print(f"[red]Questions file not found: {questions_path}[/red]")
        raise typer.Exit(1)

    questions = load_questions(str(questions_path))
    rich_print(f"Loaded {len(questions)} questions from {questions_path}")

    # Check if questions have claims (required for assertion generation)
    questions_with_claims = [
        q for q in questions if q.attributes and q.attributes.get("claims")
    ]
    if not questions_with_claims:
        rich_print(
            "[red]No questions with claims found. Assertions require claims in question attributes.[/red]"
        )
        raise typer.Exit(1)

    rich_print(f"Found {len(questions_with_claims)} questions with claims")

    # Create models
    embedding_model = ModelFactory.create_embedding_model(config.embedding_model)
    text_embedder = TextEmbedder(embedding_model)
    chat_model = ModelFactory.create_chat_model(config.chat_model)

    # Log configuration
    if assertion_type == AssertionType.local:
        assertion_cfg = config.assertions.local
    elif assertion_type == AssertionType.linked:
        assertion_cfg = config.assertions.linked
    else:
        assertion_cfg = config.assertions.global_

    rich_print(f"Generating {assertion_type} assertions:")
    rich_print(f"  - max_assertions: {assertion_cfg.max_assertions}")
    rich_print(f"  - enable_validation: {assertion_cfg.enable_validation}")
    if assertion_cfg.enable_validation:
        rich_print(f"  - min_validation_score: {assertion_cfg.min_validation_score}")

    # Generate assertions
    loop = asyncio.get_event_loop()
    questions_with_assertions = loop.run_until_complete(
        __generate_assertions_for_questions(
            questions=questions_with_claims,
            assertion_type=assertion_type,
            llm=chat_model,
            text_embedder=text_embedder,
            assertion_config=config.assertions,
            assertion_prompt_config=config.assertion_prompts,
            llm_params=config.chat_model.call_args,
        )
    )

    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    save_questions(
        questions_with_assertions,
        str(output_path),
        "questions_with_assertions",
    )

    # Count assertions generated
    total_assertions = sum(
        len(q.attributes.get("assertions", []))
        for q in questions_with_assertions
        if q.attributes
    )
    rich_print(
        f"[green]Generated {total_assertions} assertions for "
        f"{len(questions_with_assertions)} questions[/green]"
    )
    rich_print(f"[green]Results saved to: {output_path}[/green]")

    if print_model_usage:
        rich_print("Chat Model usage statistics:")
        rich_print(chat_model.get_usage())
        rich_print("Embedding Model usage statistics:")
        rich_print(embedding_model.get_usage())
