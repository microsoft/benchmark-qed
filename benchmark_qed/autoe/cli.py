# Copyright (c) 2025 Microsoft Corporation.
"""Score CLI for generating scores and significance tests for different conditions."""

import asyncio
import json
from io import StringIO
from itertools import combinations, product
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import pandas as pd
import typer
from graphrag_common.config import load_config
from graphrag_storage import Storage
from graphrag_storage.file_storage import FileStorage
from graphrag_storage.storage_config import StorageConfig
from graphrag_storage.storage_factory import create_storage
from rich import print as rich_print
from tqdm import tqdm

from benchmark_qed.autoe.assertion import (
    aggregate_hierarchical_scores,
    compute_hierarchical_eval_summary,
    get_assertion_scores,
    get_hierarchical_assertion_scores,
    run_assertion_evaluation,
    summarize_hierarchical_by_question,
    summarize_standard_scores,
)
from benchmark_qed.autoe.config import (
    AssertionConfig,
    AssertionSignificanceConfig,
    HierarchicalAssertionConfig,
    HierarchicalAssertionSignificanceConfig,
    MultiRAGAssertionConfig,
    PairwiseConfig,
    ReferenceConfig,
    RetrievalReferenceConfig,
    RetrievalScoresConfig,
)
from benchmark_qed.autoe.pairwise import analyze_criteria, get_pairwise_scores
from benchmark_qed.autoe.reference import get_reference_scores
from benchmark_qed.cli.config_resolver import (
    AccountUrlOption,
    ConnectionStringOption,
    resolve_config_path,
)
from benchmark_qed.cli.utils import print_df
from benchmark_qed.llm.factory import ModelFactory

app: typer.Typer = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Evaluate Retrieval-Augmented Generation (RAG) methods.",
)


async def _read_json_df(storage: Storage, key: str) -> pd.DataFrame:
    """Read a JSON file from storage and return as a DataFrame."""
    data = await storage.get(key)
    if data is None:
        msg = f"File not found in storage: {key}"
        raise FileNotFoundError(msg)
    return pd.read_json(StringIO(data))


async def _read_csv_df(storage: Storage, key: str) -> pd.DataFrame:
    """Read a CSV file from storage and return as a DataFrame."""
    data = await storage.get(key)
    if data is None:
        msg = f"File not found in storage: {key}"
        raise FileNotFoundError(msg)
    return pd.read_csv(StringIO(data))


async def _write_csv_df(storage: Storage, key: str, df: pd.DataFrame) -> None:
    """Write a DataFrame as CSV to storage."""
    await storage.set(key, df.to_csv(index=False))


async def _write_json(storage: Storage, key: str, data: object) -> None:
    """Write JSON data to storage."""
    await storage.set(key, json.dumps(data))


def _build_output_storage(
    storage_config: StorageConfig | None, output: Path
) -> Storage:
    """Build output storage from config or default to FileStorage."""
    if storage_config:
        storage = create_storage(storage_config)
        output_posix = output.as_posix().removeprefix("./")
        return storage.child(output_posix) if output_posix else storage
    output.mkdir(parents=True, exist_ok=True)
    return FileStorage(base_dir=str(output))


def _build_condition_storage(
    storage_config: StorageConfig | None, answer_base_path: Path, *, is_dir: bool
) -> tuple[Storage, str]:
    """Build storage for a condition's answer path.

    Always returns a non-null Storage instance: uses the provided StorageConfig when
    specified, otherwise falls back to FileStorage on the local filesystem.

    Returns (storage, filename) where filename is the key to read.
    For directory-style paths (pairwise), filename is empty.
    """
    if storage_config:
        storage = create_storage(storage_config)
        path_posix = answer_base_path.as_posix().removeprefix("./")
        if is_dir:
            return storage.child(path_posix) if path_posix else storage, ""
        parent = str(Path(path_posix).parent)
        name = Path(path_posix).name
        child = storage.child(parent) if parent != "." else storage
        return child, name
    if is_dir:
        return FileStorage(base_dir=str(answer_base_path)), ""
    return FileStorage(base_dir=str(answer_base_path.parent)), answer_base_path.name


@app.command()
def pairwise_scores(
    comparison_spec: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containing the conditions."),
    ],
    output: Annotated[
        Path, typer.Argument(help="The path to the output file for the scores.")
    ],
    *,
    alpha: Annotated[
        float, typer.Option(help="The p-value threshold for the significance test.")
    ] = 0.05,
    exclude_criteria: Annotated[
        list[str] | None,
        typer.Option(help="The criteria to exclude from the scoring."),
    ] = None,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics after scoring."),
    ] = False,
    include_score_id_in_prompt: Annotated[
        bool,
        typer.Option(
            help="Whether to include the score ID in the evaluation prompt for the LLM (might be useful to avoid cached scores)."
        ),
    ] = True,
    question_id_key: Annotated[
        str,
        typer.Option(
            help="The key in the JSON file that contains the question ID. This is used to match questions across different conditions."
        ),
    ] = "question_id",
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Generate scores for the different conditions provided in the JSON file."""
    if exclude_criteria is None:
        exclude_criteria = []
    comparison_spec = resolve_config_path(
        comparison_spec,
        account_url=account_url,
        connection_string=connection_string,
    )
    config = load_config(PairwiseConfig, comparison_spec)

    config.criteria = [
        criterion
        for criterion in config.criteria
        if criterion.name not in exclude_criteria
    ]

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    output_storage = _build_output_storage(config.output_storage, output)
    all_results = []

    all_combinations = (
        product([config.base], config.others)
        if config.base
        else combinations(config.others, 2)
    )

    for base, other in all_combinations:
        for question_set in config.question_sets:
            rich_print(f"Scoring {base.name} vs {other.name} for {question_set}")
            cache_key = f"{question_set}_{base.name}--{other.name}.csv"
            if asyncio.run(output_storage.has(cache_key)):
                rich_print(
                    f"{base.name} vs {other.name} for {question_set} already exists. Skipping generation.\n"
                    f"[bold yellow]If you want to generate a new comparison, delete {cache_key} from {output}.[/bold yellow]"
                )
                result = asyncio.run(_read_csv_df(output_storage, cache_key))
            else:
                base_storage, _ = _build_condition_storage(
                    config.input_storage, base.answer_base_path, is_dir=True
                )
                other_storage, _ = _build_condition_storage(
                    config.input_storage, other.answer_base_path, is_dir=True
                )
                result = get_pairwise_scores(
                    llm_client=llm_client,
                    llm_config=config.llm_config,
                    base_name=base.name,
                    other_name=other.name,
                    base_answers=asyncio.run(
                        _read_json_df(
                            base_storage,
                            f"{question_set}.json",
                        )
                    ),
                    other_answers=asyncio.run(
                        _read_json_df(
                            other_storage,
                            f"{question_set}.json",
                        )
                    ),
                    criteria=config.criteria,
                    assessment_user_prompt=config.prompt_config.user_prompt.template,
                    assessment_system_prompt=config.prompt_config.system_prompt.template,
                    trials=config.trials,
                    question_id_key=question_id_key,
                    include_score_id_in_prompt=include_score_id_in_prompt,
                )

                asyncio.run(_write_csv_df(output_storage, cache_key, result))
            result["question_set"] = question_set
            all_results.append(result)

    all_results = pd.concat(all_results)
    asyncio.run(_write_csv_df(output_storage, "win_rates.csv", all_results))

    all_results_p_value = analyze_criteria(
        all_results,
        alpha=alpha,
    )

    asyncio.run(
        _write_csv_df(output_storage, "winrates_sig_tests.csv", all_results_p_value)
    )

    print_df(
        cast(
            pd.DataFrame,
            all_results_p_value[
                [
                    "question_set",
                    "criteria",
                    "base_name",
                    "other_name",
                    "base_mean",
                    "other_mean",
                    "formatted_corrected_p_value",
                ]
            ],
        ),
        "Pairwise Scores Summary",
    )

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.metrics_store.get_metrics())
    asyncio.run(
        _write_json(
            output_storage, "model_usage.json", llm_client.metrics_store.get_metrics()
        )
    )


@app.command()
def reference_scores(
    comparison_spec: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containing the configuration."),
    ],
    output: Annotated[
        Path, typer.Argument(help="The path to the output file for the scores.")
    ],
    *,
    exclude_criteria: Annotated[
        list[str] | None,
        typer.Option(help="The criteria to exclude from the scoring."),
    ] = None,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics after scoring."),
    ] = False,
    include_score_id_in_prompt: Annotated[
        bool,
        typer.Option(
            help="Whether to include the score ID in the evaluation prompt for the LLM (might be useful to avoid cached scores)."
        ),
    ] = True,
    question_id_key: Annotated[
        str,
        typer.Option(
            help="The key in the JSON file that contains the question ID. This is used to match questions across different conditions."
        ),
    ] = "question_id",
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Generate scores for the generated answers provided in the JSON file."""
    if exclude_criteria is None:
        exclude_criteria = []
    comparison_spec = resolve_config_path(
        comparison_spec,
        account_url=account_url,
        connection_string=connection_string,
    )
    config = load_config(ReferenceConfig, comparison_spec)

    config.criteria = [
        criterion
        for criterion in config.criteria
        if criterion.name not in exclude_criteria
    ]

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    output_storage = _build_output_storage(config.output_storage, output)

    for generated in config.generated:
        generated_storage, generated_key = _build_condition_storage(
            config.input_storage, generated.answer_base_path, is_dir=False
        )
        reference_storage, reference_key = _build_condition_storage(
            config.input_storage, config.reference.answer_base_path, is_dir=False
        )
        result = get_reference_scores(
            llm_client=llm_client,
            llm_config=config.llm_config,
            generated_answers=asyncio.run(
                _read_json_df(generated_storage, generated_key)
            ),
            reference_answers=asyncio.run(
                _read_json_df(reference_storage, reference_key)
            ),
            criteria=config.criteria,
            assessment_user_prompt=config.prompt_config.user_prompt.template,
            assessment_system_prompt=config.prompt_config.system_prompt.template,
            score_min=config.score_min,
            score_max=config.score_max,
            trials=config.trials,
            include_score_id_in_prompt=include_score_id_in_prompt,
            question_id_key=question_id_key,
        )
        asyncio.run(
            _write_csv_df(
                output_storage,
                f"reference_scores-{generated.name}.csv",
                result,
            )
        )
        summary_df = cast(
            pd.DataFrame,
            result.drop(
                columns=[
                    "question",
                    "reference_answer",
                    "generated_answer",
                    "reasoning",
                    "trial",
                ]
            )
            .groupby("criteria")
            .agg(list),
        )

        summary_df["mean"] = summary_df["score"].apply(np.mean)
        summary_df["std"] = summary_df["score"].apply(np.std)
        summary_df = summary_df.drop(columns=["score"])
        print_df(
            summary_df.drop(columns=["score_id"]).reset_index(),
            f"Reference Scores Summary for {generated.name}",
        )

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.metrics_store.get_metrics())
    asyncio.run(
        _write_json(
            output_storage, "model_usage.json", llm_client.metrics_store.get_metrics()
        )
    )


@app.command()
def assertion_scores(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the assertion scoring config YAML."),
    ],
    output: Annotated[
        Path | None,
        typer.Argument(
            help="Output directory (required for single-RAG config, ignored for multi-RAG)."
        ),
    ] = None,
    *,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics."),
    ] = False,
    include_score_id_in_prompt: Annotated[
        bool,
        typer.Option(
            help="Whether to include the score ID in the prompt (single-RAG mode only)."
        ),
    ] = True,
    question_id_key: Annotated[
        str,
        typer.Option(help="Question ID key in JSON (single-RAG mode only)."),
    ] = "question_id",
    question_text_key: Annotated[
        str,
        typer.Option(help="Question text key in JSON (single-RAG mode only)."),
    ] = "question_text",
    answer_text_key: Annotated[
        str,
        typer.Option(help="Answer text key in JSON (single-RAG mode only)."),
    ] = "answer",
    assertions_key: Annotated[
        str,
        typer.Option(help="Assertions key in JSON (single-RAG mode only)."),
    ] = "assertions",
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Score assertions for RAG method(s).

    Supports two config formats (auto-detected):

    1. Single-RAG mode (legacy): requires 'generated' key and output argument
    2. Multi-RAG mode: requires 'rag_methods' key, includes significance testing

    Single-RAG config example:
        generated:
          name: vector_rag
          answer_base_path: input/vector_rag/answers.json
        assertions:
          assertions_path: input/assertions.json
        pass_threshold: 0.5
        trials: 4
        llm_config: ...

    Multi-RAG config example:
        input_dir: ./input
        output_dir: ./output
        rag_methods: [graphrag, vectorrag]
        question_sets: [data_global_questions]
        run_significance_test: true
        llm_config: ...
    """
    import yaml

    config_path = resolve_config_path(
        config_path,
        account_url=account_url,
        connection_string=connection_string,
    )

    # Load raw YAML to detect format
    with Path(config_path).open(encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Auto-detect config format
    is_multi_rag = "rag_methods" in raw_config

    if is_multi_rag:
        _run_multi_rag_assertion_scores(config_path, print_model_usage)
    else:
        if output is None:
            rich_print(
                "[bold red]Error: output directory is required for single-RAG config.[/bold red]"
            )
            rich_print(
                "Usage: benchmark-qed autoe assertion-scores config.yaml output_dir"
            )
            raise typer.Exit(1)
        _run_single_rag_assertion_scores(
            config_path=config_path,
            output=output,
            print_model_usage=print_model_usage,
            include_score_id_in_prompt=include_score_id_in_prompt,
            question_id_key=question_id_key,
            question_text_key=question_text_key,
            answer_text_key=answer_text_key,
            assertions_key=assertions_key,
        )


def _run_single_rag_assertion_scores(
    config_path: Path,
    output: Path,
    *,
    print_model_usage: bool,
    include_score_id_in_prompt: bool,
    question_id_key: str,
    question_text_key: str,
    answer_text_key: str,
    assertions_key: str,
) -> None:
    """Run assertion scoring for a single RAG method (legacy mode)."""
    config = load_config(AssertionConfig, config_path)
    output_storage = _build_output_storage(config.output_storage, output)

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    assertions_storage, assertions_filename = _build_condition_storage(
        config.input_storage, config.assertions.assertions_path, is_dir=False
    )
    assertions = asyncio.run(_read_json_df(assertions_storage, assertions_filename))

    if assertions.loc[:, assertions_key].isna().any():  # type: ignore
        msg = f"Some questions in the assertions file do not have assertions. Please check {config.assertions.assertions_path}, these questions will be skipped."
        rich_print(f"[bold red]{msg}[/bold red]")
    assertions = assertions[~assertions.loc[:, assertions_key].isna()]

    if assertions.loc[:, assertions_key].apply(lambda x: len(x) == 0).any():
        msg = f"Some questions in the assertions file have empty assertions. Please check {config.assertions.assertions_path}, these questions will be skipped."
        rich_print(f"[bold red]{msg}[/bold red]")
    assertions = cast(
        pd.DataFrame,
        assertions[assertions.loc[:, assertions_key].apply(lambda x: len(x) > 0)],
    )

    assertions = (
        assertions.explode(assertions_key)
        .rename(columns={assertions_key: "assertion"})
        .reset_index(drop=True)
    )

    generated_storage, generated_key = _build_condition_storage(
        config.input_storage, config.generated.answer_base_path, is_dir=False
    )
    assertion_score = get_assertion_scores(
        llm_client=llm_client,
        llm_config=config.llm_config,
        answers=asyncio.run(_read_json_df(generated_storage, generated_key)),
        assertions=assertions,
        assessment_user_prompt=config.prompt_config.user_prompt.template,
        assessment_system_prompt=config.prompt_config.system_prompt.template,
        trials=config.trials,
        include_score_id_in_prompt=include_score_id_in_prompt,
        question_id_key=question_id_key,
        question_text_key=question_text_key,
        answer_text_key=answer_text_key,
    )

    asyncio.run(_write_csv_df(output_storage, "assertion_scores.csv", assertion_score))

    # Compute summaries using shared aggregation logic
    summary_by_assertion, summary_by_question, eval_summary = summarize_standard_scores(
        assertion_score
    )

    # Save summary CSVs for consistency with multi-RAG pipeline
    asyncio.run(
        _write_csv_df(
            output_storage,
            "assertion_summary_by_question.csv",
            summary_by_question,
        )
    )
    asyncio.run(
        _write_csv_df(
            output_storage,
            "assertion_summary_by_assertion.csv",
            summary_by_assertion,
        )
    )

    print_df(
        summary_by_question,
        "Assertion Scores Summary by Question",
    )

    failed_assertions: pd.DataFrame = cast(
        pd.DataFrame,
        summary_by_assertion[summary_by_assertion["score"] == 0],
    )
    failed_assertions = failed_assertions.drop(columns=["score"])

    if len(failed_assertions) > 0:
        print_df(
            failed_assertions,
            f"[bold red]{failed_assertions.shape[0]} Failed Assertions[/bold red]",
        )
        rich_print(
            f"[bold red]{failed_assertions.shape[0]} assertions failed. See {output / 'assertion_scores.csv'} for details.[/bold red]"
        )
    else:
        rich_print("[bold green]All assertions passed.[/bold green]")

    # Save machine-readable evaluation summary
    asyncio.run(_write_json(output_storage, "eval_summary.json", eval_summary))

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.metrics_store.get_metrics())
    asyncio.run(
        _write_json(
            output_storage, "model_usage.json", llm_client.metrics_store.get_metrics()
        )
    )


def _run_multi_rag_assertion_scores(
    config_path: Path,
    print_model_usage: bool,
) -> None:
    """Run assertion scoring for multiple RAG methods with significance testing."""
    config = load_config(MultiRAGAssertionConfig, config_path)
    output_storage = _build_output_storage(config.output_storage, config.output_dir)
    input_storage = (
        create_storage(config.input_storage)
        if config.input_storage is not None
        else None
    )

    llm_client = ModelFactory.create_chat_model(config.llm_config)

    rich_print("[bold]Running multi-RAG assertion scoring[/bold]")
    rich_print(f"  Input dir: {config.input_dir}")
    rich_print(f"  Output dir: {config.output_dir}")
    rich_print(f"  RAG methods: {config.rag_methods}")
    rich_print(f"  Question sets: {config.question_sets}")
    rich_print(f"  Top-k assertions: {config.top_k_assertions or 'all'}")
    rich_print(f"  Trials: {config.trials}")
    if config.run_significance_test:
        rich_print(
            f"  Significance test: enabled "
            f"(alpha={config.significance_alpha}, "
            f"correction={config.significance_correction})"
        )
        if config.run_clustered_permutation:
            rich_print(
                "  Clustered permutation: enabled "
                f"(n={config.n_permutations}, "
                f"seed={config.permutation_seed})"
            )

    # Run the evaluation pipeline
    results_df = run_assertion_evaluation(
        llm_client=llm_client,
        llm_config=config.llm_config,
        question_sets=config.question_sets,
        generated_rags=config.rag_methods,
        input_dir=str(config.input_dir),
        output_dir=config.output_dir,
        trials=config.trials,
        top_k_assertions=config.top_k_assertions,
        pass_threshold=config.pass_threshold,
        assertions_filename_template=config.assertions_filename_template,
        answers_path_template=config.answers_path_template,
        run_significance_test=config.run_significance_test,
        significance_alpha=config.significance_alpha,
        significance_correction=config.significance_correction,
        run_clustered_permutation=config.run_clustered_permutation,
        n_permutations=config.n_permutations,
        permutation_seed=config.permutation_seed,
        question_text_key=config.question_text_key,
        answer_text_key=config.answer_text_key,
        output_storage=output_storage,
        input_storage=input_storage,
    )

    if len(results_df) > 0:
        rich_print("\n[bold green]Evaluation complete![/bold green]")
        rich_print(f"Results saved to: {config.output_dir}")
    else:
        rich_print(
            "[yellow]No results generated. Check input paths and files.[/yellow]"
        )

    if print_model_usage:
        rich_print("\nModel usage statistics:")
        rich_print(llm_client.metrics_store.get_metrics())
    asyncio.run(
        _write_json(
            output_storage, "model_usage.json", llm_client.metrics_store.get_metrics()
        )
    )


@app.command()
def hierarchical_assertion_scores(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the hierarchical assertion scoring config YAML."),
    ],
    output: Annotated[
        Path | None,
        typer.Argument(
            help="Output directory (required for single-RAG config, ignored for multi-RAG)."
        ),
    ] = None,
    *,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics after scoring."),
    ] = False,
    include_score_id_in_prompt: Annotated[
        bool,
        typer.Option(
            help="Whether to include the score ID in the evaluation prompt for the LLM."
        ),
    ] = True,
    question_id_key: Annotated[
        str,
        typer.Option(help="The key in the JSON file that contains the question ID."),
    ] = "question_id",
    question_text_key: Annotated[
        str,
        typer.Option(help="The key in the JSON file that contains the question text."),
    ] = "question_text",
    answer_text_key: Annotated[
        str,
        typer.Option(help="The key in the JSON file that contains the answer text."),
    ] = "answer",
    assertions_key: Annotated[
        str,
        typer.Option(help="The key in the JSON file that contains the assertions."),
    ] = "assertions",
    supporting_assertions_key: Annotated[
        str,
        typer.Option(
            help="The key in assertions that contains the supporting assertions list."
        ),
    ] = "supporting_assertions",
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Score hierarchical assertions with supporting assertions.

    Supports two config formats (auto-detected):

    1. Single-RAG mode (legacy): requires 'generated' key and output argument
    2. Multi-RAG mode: requires 'rag_methods' key, includes significance testing

    Single-RAG config example:
        generated:
          name: vector_rag
          answer_base_path: input/vector_rag/data_global.json
        assertions:
          assertions_path: input/data_global_assertions.json
        pass_threshold: 0.5
        trials: 4
        mode: staged
        llm_config: ...

    Multi-RAG config example:
        input_dir: ./input
        output_dir: ./output
        rag_methods: [graphrag, vectorrag]
        assertions_file: data_global_assertions.json
        run_significance_test: true
        mode: staged
        llm_config: ...

    This command evaluates global assertions along with their supporting (local)
    assertions. It computes:
    - Global assertion pass/fail
    - Support coverage (ratio of supporting assertions that passed)
    - Discovery detection (information beyond supporting assertions)
    """
    import yaml

    config_path = resolve_config_path(
        config_path,
        account_url=account_url,
        connection_string=connection_string,
    )
    # Load raw YAML to detect format
    with Path(config_path).open(encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Auto-detect config format
    is_multi_rag = "rag_methods" in raw_config

    if is_multi_rag:
        _run_multi_rag_hierarchical_assertion_scores(config_path, print_model_usage)
    else:
        if output is None:
            rich_print(
                "[bold red]Error: output directory is required for single-RAG config.[/bold red]"
            )
            rich_print(
                "Usage: benchmark-qed autoe hierarchical-assertion-scores config.yaml output_dir"
            )
            raise typer.Exit(1)
        _run_single_rag_hierarchical_assertion_scores(
            config_path=config_path,
            output=output,
            print_model_usage=print_model_usage,
            include_score_id_in_prompt=include_score_id_in_prompt,
            question_id_key=question_id_key,
            question_text_key=question_text_key,
            answer_text_key=answer_text_key,
            assertions_key=assertions_key,
            supporting_assertions_key=supporting_assertions_key,
        )


def _run_single_rag_hierarchical_assertion_scores(
    config_path: Path,
    output: Path,
    *,
    print_model_usage: bool,
    include_score_id_in_prompt: bool,
    question_id_key: str,
    question_text_key: str,
    answer_text_key: str,
    assertions_key: str,
    supporting_assertions_key: str,
) -> None:
    """Run hierarchical assertion scoring for a single RAG method."""
    from benchmark_qed.autoe.assertion import (
        load_and_normalize_hierarchical_assertions,
    )

    config = load_config(HierarchicalAssertionConfig, config_path)
    output_storage = _build_output_storage(config.output_storage, output)

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    assertions_storage, assertions_filename = _build_condition_storage(
        config.input_storage, config.assertions.assertions_path, is_dir=False
    )
    from benchmark_qed.autoe.utils.storage_io import read_json_df

    assertions = load_and_normalize_hierarchical_assertions(
        assertions_filename,
        assertions_key=assertions_key,
        supporting_assertions_key=supporting_assertions_key,
        input_storage=assertions_storage,
    )

    rich_print(f"Evaluating {len(assertions)} hierarchical assertions...")

    answers_storage, answers_filename = _build_condition_storage(
        config.input_storage, config.generated.answer_base_path, is_dir=False
    )
    answers = read_json_df(answers_storage, answers_filename)

    # Get hierarchical scores
    scores = get_hierarchical_assertion_scores(
        llm_client=llm_client,
        llm_config=config.llm_config,
        answers=answers,
        assertions=assertions,
        assessment_user_prompt=config.prompt_config.user_prompt.template,
        assessment_system_prompt=config.prompt_config.system_prompt.template,
        trials=config.trials,
        mode=config.mode,
        pass_threshold=config.pass_threshold,
        include_score_id_in_prompt=include_score_id_in_prompt,
        question_id_key=question_id_key,
        question_text_key=question_text_key,
        answer_text_key=answer_text_key,
        supporting_assertions_key=supporting_assertions_key,
    )

    # Save raw scores
    asyncio.run(
        _write_csv_df(output_storage, "hierarchical_assertion_scores.csv", scores)
    )

    # Aggregate across trials
    aggregated = aggregate_hierarchical_scores(
        scores, pass_threshold=config.pass_threshold
    )
    asyncio.run(
        _write_csv_df(output_storage, "hierarchical_assertion_summary.csv", aggregated)
    )

    # Summarize by question
    summary_by_question = summarize_hierarchical_by_question(aggregated)
    asyncio.run(
        _write_csv_df(
            output_storage,
            "hierarchical_summary_by_question.csv",
            summary_by_question,
        )
    )

    # Print summary
    print_df(summary_by_question, "Hierarchical Assertion Summary by Question")

    # Compute comprehensive evaluation metrics using shared logic
    eval_summary = compute_hierarchical_eval_summary(aggregated)

    # Print summary statistics
    rich_print("\n[bold]Overall Statistics:[/bold]")
    rich_print(
        f"  Global assertions passed: "
        f"{eval_summary['passed_assertions']}/"
        f"{eval_summary['total_assertions']} "
        f"({eval_summary['global_pass_rate'] * 100:.1f}%)"  # type: ignore[operator]
    )
    rich_print(
        f"  Average support level: {eval_summary['avg_support_level'] * 100:.1f}%"  # type: ignore[operator]
    )
    rich_print(f"  Assertions with discovery: {eval_summary['discovery_count']}")
    if eval_summary["overridden_count"]:
        rich_print(
            f"  [yellow]Overridden to fail (no support/discovery): "
            f"{eval_summary['overridden_count']}[/yellow]"
        )

    # Report failed assertions
    failed = aggregated[aggregated["global_score"] == 0]
    if len(failed) > 0:
        rich_print(f"\n[bold red]{len(failed)} global assertions failed.[/bold red]")
    else:
        rich_print("\n[bold green]All global assertions passed.[/bold green]")

    # Save machine-readable evaluation summary
    asyncio.run(_write_json(output_storage, "eval_summary.json", eval_summary))

    if print_model_usage:
        rich_print("\nModel usage statistics:")
        rich_print(llm_client.metrics_store.get_metrics())
    asyncio.run(
        _write_json(
            output_storage, "model_usage.json", llm_client.metrics_store.get_metrics()
        )
    )


def _run_multi_rag_hierarchical_assertion_scores(
    config_path: Path,
    print_model_usage: bool,
) -> None:
    """Run hierarchical assertion scoring for multiple RAG methods with significance testing."""
    from benchmark_qed.autoe.assertion import run_hierarchical_assertion_evaluation
    from benchmark_qed.autoe.config import MultiRAGHierarchicalAssertionConfig

    config = load_config(MultiRAGHierarchicalAssertionConfig, config_path)
    output_storage = _build_output_storage(config.output_storage, config.output_dir)
    input_storage = (
        create_storage(config.input_storage)
        if config.input_storage is not None
        else None
    )

    llm_client = ModelFactory.create_chat_model(config.llm_config)

    rich_print("[bold]Running multi-RAG hierarchical assertion scoring[/bold]")
    rich_print(f"  Input dir: {config.input_dir}")
    rich_print(f"  Output dir: {config.output_dir}")
    rich_print(f"  RAG methods: {config.rag_methods}")
    rich_print(f"  Assertions file: {config.assertions_file}")
    rich_print(f"  Mode: {config.mode.value}")
    rich_print(f"  Trials: {config.trials}")
    if config.run_significance_test:
        rich_print(
            f"  Significance test: enabled "
            f"(alpha={config.significance_alpha}, "
            f"correction={config.significance_correction})"
        )

    # Load and prepare assertions
    from benchmark_qed.autoe.assertion import (
        load_and_normalize_hierarchical_assertions,
    )

    assertions_path = (
        config.input_dir / config.assertions_file
        if not Path(config.assertions_file).is_absolute()
        else Path(config.assertions_file)
    )
    assertions = load_and_normalize_hierarchical_assertions(
        str(assertions_path) if input_storage is None else config.assertions_file,
        supporting_assertions_key=config.supporting_assertions_key,
        input_storage=input_storage,
    )
    rich_print(f"Loaded {len(assertions)} hierarchical assertions")

    # Run the evaluation pipeline
    results_df = run_hierarchical_assertion_evaluation(
        llm_client=llm_client,
        llm_config=config.llm_config,
        generated_rags=config.rag_methods,
        assertions=assertions,
        input_dir=str(config.input_dir),
        output_dir=config.output_dir,
        trials=config.trials,
        pass_threshold=config.pass_threshold,
        mode=config.mode,
        answers_path_template=config.answers_path_template,
        run_significance_test=config.run_significance_test,
        significance_alpha=config.significance_alpha,
        significance_correction=config.significance_correction,
        run_clustered_permutation=config.run_clustered_permutation,
        n_permutations=config.n_permutations,
        permutation_seed=config.permutation_seed,
        question_id_key=config.question_id_key,
        question_text_key=config.question_text_key,
        answer_text_key=config.answer_text_key,
        supporting_assertions_key=config.supporting_assertions_key,
        output_storage=output_storage,
        input_storage=input_storage,
    )

    if len(results_df) > 0:
        rich_print("\n[bold green]Evaluation complete![/bold green]")
        rich_print(f"Results saved to: {config.output_dir}")
    else:
        rich_print(
            "[yellow]No results generated. Check input paths and files.[/yellow]"
        )

    if print_model_usage:
        rich_print("\nModel usage statistics:")
        rich_print(llm_client.metrics_store.get_metrics())
    asyncio.run(
        _write_json(
            output_storage, "model_usage.json", llm_client.metrics_store.get_metrics()
        )
    )


@app.command()
def assertion_significance(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the assertion significance configuration YAML file."
        ),
    ],
    *,
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Run statistical significance tests on standard assertion scores.

    Compares assertion scores across multiple RAG methods using statistical
    tests (Friedman/Kruskal-Wallis for omnibus, Wilcoxon/Mann-Whitney for
    post-hoc).

    Example config:
        output_dir: ./output/assertion_scoring
        rag_methods:
          - graphrag_global
          - vectorrag
          - lazygraphrag
        question_sets:
          - data_global_questions
          - data_local_questions
        alpha: 0.05
        correction_method: holm
    """
    from benchmark_qed.autoe.assertion import compare_assertion_scores_significance

    config_path = resolve_config_path(
        config_path,
        account_url=account_url,
        connection_string=connection_string,
    )
    config = load_config(AssertionSignificanceConfig, config_path)

    rich_print("[bold]Running assertion significance tests[/bold]")
    rich_print(f"  Output dir: {config.output_dir}")
    rich_print(f"  RAG methods: {config.rag_methods}")
    rich_print(f"  Question sets: {config.question_sets}")
    rich_print(f"  Alpha: {config.alpha}, Correction: {config.correction_method}")
    if config.run_clustered_permutation:
        rich_print(
            "  Clustered permutation: enabled "
            f"(n={config.n_permutations}, seed={config.permutation_seed})"
        )

    results = compare_assertion_scores_significance(
        output_dir=config.output_dir,
        generated_rags=config.rag_methods,
        question_sets=config.question_sets,
        alpha=config.alpha,
        correction_method=config.correction_method,
        run_clustered_permutation=config.run_clustered_permutation,
        n_permutations=config.n_permutations,
        permutation_seed=config.permutation_seed,
        output_storage=_build_output_storage(config.output_storage, config.output_dir)
        if config.output_storage is not None
        else None,
    )

    # Summary
    rich_print("\n[bold]===== Summary =====[/bold]")
    for question_set, result in results.items():
        sig_marker = "✓" if result.omnibus.is_significant else "✗"
        rich_print(
            f"  {question_set}: {result.omnibus.test_name} "
            f"p={result.omnibus.p_value:.4f} [{sig_marker}]"
        )


@app.command()
def hierarchical_assertion_significance(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the hierarchical assertion significance config YAML."
        ),
    ],
    *,
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Run statistical significance tests on hierarchical assertion scores.

    Compares hierarchical assertion scores across multiple RAG methods on
    four metrics: global_pass_rate, support_level, supporting_pass_rate,
    and discovery_rate.

    Example config:
        scores_dir: ./output/hierarchical_scoring
        rag_methods:
          - graphrag_global
          - vectorrag
          - lazygraphrag
        scores_filename_template: "{rag_method}_hierarchical_scores_aggregated.csv"
        alpha: 0.05
        correction_method: holm
        output_dir: ./output/significance  # Optional
    """
    from benchmark_qed.autoe.assertion import (
        compare_hierarchical_assertion_scores_significance,
    )

    config_path = resolve_config_path(
        config_path,
        account_url=account_url,
        connection_string=connection_string,
    )
    config = load_config(HierarchicalAssertionSignificanceConfig, config_path)
    input_storage = (
        create_storage(config.input_storage)
        if config.input_storage is not None
        else FileStorage(base_dir=str(config.scores_dir))
    )
    output_storage = (
        _build_output_storage(config.output_storage, config.output_dir)
        if config.output_dir is not None
        else None
    )

    rich_print("[bold]Running hierarchical assertion significance tests[/bold]")
    rich_print(f"  Scores dir: {config.scores_dir}")
    rich_print(f"  RAG methods: {config.rag_methods}")
    rich_print(f"  Alpha: {config.alpha}, Correction: {config.correction_method}")

    # Load aggregated scores for each RAG method
    aggregated_scores: dict[str, pd.DataFrame] = {}
    for rag_method in config.rag_methods:
        filename = config.scores_filename_template.format(rag_method=rag_method)
        # When input_storage is configured, treat filename as a key relative to it.
        # Otherwise the FileStorage built above is rooted at scores_dir, so the
        # filename is also the key.
        if not asyncio.run(input_storage.has(filename)):
            rich_print(
                f"  [yellow]Warning: {filename} not found in input storage, skipping[/yellow]"
            )
            continue
        aggregated_scores[rag_method] = asyncio.run(
            _read_csv_df(input_storage, filename)
        )
        rich_print(
            f"  Loaded {len(aggregated_scores[rag_method])} rows for {rag_method}"
        )

    if len(aggregated_scores) < 2:
        rich_print("[red]Error: Need at least 2 RAG methods with data[/red]")
        raise typer.Exit(1)

    results = compare_hierarchical_assertion_scores_significance(
        aggregated_scores=aggregated_scores,
        alpha=config.alpha,
        correction_method=config.correction_method,
        output_dir=config.output_dir,
        run_clustered_permutation=config.run_clustered_permutation,
        n_permutations=config.n_permutations,
        permutation_seed=config.permutation_seed,
        output_storage=output_storage,
    )

    # Summary
    rich_print("\n[bold]===== Summary =====[/bold]")
    for metric, result in results.items():
        sig_marker = "✓" if result.omnibus.is_significant else "✗"
        rich_print(
            f"  {metric}: {result.omnibus.test_name} "
            f"p={result.omnibus.p_value:.4f} [{sig_marker}]"
        )


@app.command()
def generate_retrieval_reference(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the retrieval reference configuration JSON file."),
    ],
    *,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics."),
    ] = False,
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Generate retrieval reference data (cluster relevance) for a question set.

    This is a one-off operation that creates reference data used by retrieval_scores.
    The reference data identifies which clusters are relevant to each question.

    Supports multiple question sets and cluster counts via config:
    - question_sets: List of {name, questions_path} for multiple question sets
    - num_clusters: Single int or list of ints for multiple cluster counts
    - save_clusters: Whether to save clustering results separately for debugging

    Output structure:
      output_dir/
        clusters/
          clusters_{num_clusters}.json
        {question_set_name}/
          clusters_{num_clusters}/
            reference.json
            model_usage.json

    If clusters_path is provided, pre-computed clusters will be loaded.
    Otherwise, text units will be loaded from text_units_path and clustered.
    """
    # Run all async work in a single event loop
    config_path = resolve_config_path(
        config_path,
        account_url=account_url,
        connection_string=connection_string,
    )
    asyncio.run(_generate_retrieval_reference_async(config_path, print_model_usage))


async def _generate_retrieval_reference_async(
    config_path: Path,
    print_model_usage: bool,
) -> None:
    """Async implementation of generate_retrieval_reference."""
    from benchmark_qed.autod.data_model.text_unit import TextUnit
    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
    from benchmark_qed.autod.io.text_unit import load_text_units
    from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
    from benchmark_qed.autoe.retrieval_metrics.reference_gen.cluster_relevance import (
        ClusterRelevanceRater,
        build_cluster_references_payload,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.bing_rater import (
        BingRelevanceRater,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.rationale_rater import (
        RationaleRelevanceRater,
    )

    config = load_config(RetrievalReferenceConfig, config_path)
    output_storage = _build_output_storage(config.output_storage, config.output_dir)
    input_storage: Storage | None = (
        create_storage(config.input_storage)
        if config.input_storage is not None
        else None
    )

    # Initialize LLM client
    llm_client = ModelFactory.create_chat_model(config.llm_config)

    # Initialize relevance rater based on config
    if config.assessor_type == "bing":
        relevance_rater = BingRelevanceRater(
            llm_client=llm_client,
            llm_config=config.llm_config,
            cache_dir=config.cache_dir,
            cache_enabled=config.cache_dir is not None,
            concurrent_requests=config.concurrent_requests,
        )
        rich_print(
            f"Using BingRelevanceRater (UMBRELA DNA prompt, {config.concurrent_requests} concurrent)"
        )
    else:
        relevance_rater = RationaleRelevanceRater(
            llm_client=llm_client,
            llm_config=config.llm_config,
            cache_dir=config.cache_dir,
            cache_enabled=config.cache_dir is not None,
            concurrent_requests=config.concurrent_requests,
        )
        rich_print(
            f"Using RationaleRelevanceRater (structured JSON, {config.concurrent_requests} concurrent)"
        )

    # Initialize embedding model and text embedder
    embedding_model = ModelFactory.create_embedding_model(config.embedding_config)
    embedder = TextEmbedder(embedding_model)

    # Load text units (needed for clustering)
    text_units: list[TextUnit] = []
    if config.clusters_path is None:
        rich_print(f"Loading text units from {config.text_units_path}...")

        suffix = config.text_units_path.suffix.lower()
        if input_storage is not None:
            from io import BytesIO

            from benchmark_qed.autoe.utils.storage_io import read_bytes

            tu_key = config.text_units_path.as_posix().lstrip("/")
            if suffix == ".parquet":
                text_df = pd.read_parquet(BytesIO(read_bytes(input_storage, tu_key)))
            elif suffix == ".csv":
                text_df = pd.read_csv(
                    StringIO(asyncio.run(input_storage.get(tu_key)) or "")
                )
            elif suffix in {".json", ".jsonl"}:
                text_df = pd.read_json(
                    StringIO(asyncio.run(input_storage.get(tu_key)) or ""),
                    lines=(suffix == ".jsonl"),
                )
            else:
                msg = f"Unsupported file format: {suffix}. Supported: .parquet, .csv, .json, .jsonl"
                raise ValueError(msg)
        elif suffix == ".parquet":
            text_df = pd.read_parquet(config.text_units_path)
        elif suffix == ".csv":
            text_df = pd.read_csv(config.text_units_path)
        elif suffix in {".json", ".jsonl"}:
            text_df = pd.read_json(config.text_units_path, lines=(suffix == ".jsonl"))
        else:
            msg = f"Unsupported file format: {suffix}. Supported: .parquet, .csv, .json, .jsonl"
            raise ValueError(msg)

        fields = config.text_unit_fields

        if fields.id_col not in text_df.columns:
            msg = f"Required column '{fields.id_col}' not found. Available: {list(text_df.columns)}"
            raise ValueError(msg)
        if fields.text_col not in text_df.columns:
            msg = f"Required column '{fields.text_col}' not found. Available: {list(text_df.columns)}"
            raise ValueError(msg)

        if fields.short_id_col and fields.short_id_col not in text_df.columns:
            rich_print(
                f"[yellow]Column '{fields.short_id_col}' not found, will auto-generate short_id[/yellow]"
            )
        if fields.embedding_col and fields.embedding_col not in text_df.columns:
            rich_print(
                f"[yellow]Column '{fields.embedding_col}' not found, will generate embeddings[/yellow]"
            )

        text_units = load_text_units(
            text_df,
            id_col=fields.id_col,
            text_col=fields.text_col,
            short_id_col=fields.short_id_col
            if fields.short_id_col and fields.short_id_col in text_df.columns
            else None,
            embedding_col=fields.embedding_col
            if fields.embedding_col and fields.embedding_col in text_df.columns
            else None,
        )
        rich_print(f"Loaded {len(text_units)} text units")

        units_with_embeddings = sum(
            1 for tu in text_units if tu.text_embedding is not None
        )
        if units_with_embeddings < len(text_units):
            units_without = len(text_units) - units_with_embeddings
            rich_print(
                f"[yellow]{units_without}/{len(text_units)} text units missing embeddings. Generating...[/yellow]"
            )

            text_units = await embedder.embed_batch(
                text_units=text_units, batch_size=32
            )
            rich_print(f"[green]Embedded {len(text_units)} text units[/green]")
        else:
            rich_print(f"All {len(text_units)} text units have embeddings")

    # Get question sets and cluster counts from config
    question_sets = config.get_question_sets()
    cluster_counts = config.get_cluster_counts()

    rich_print(
        f"\n[bold]Processing {len(question_sets)} question set(s) x {len(cluster_counts)} cluster count(s)[/bold]"
    )
    for qs in question_sets:
        rich_print(f"  - Question set: {qs.name} ({qs.questions_path})")
    rich_print(f"  - Cluster counts: {cluster_counts}")

    # Cache for clusters by num_clusters to avoid re-clustering
    clusters_cache: dict[int | None, list[TextCluster]] = {}

    # Storage for clusters subdirectory if saving clusters
    clusters_storage: Storage | None = (
        output_storage.child("clusters") if config.save_clusters else None
    )

    # Calculate total combinations for progress tracking
    total_combinations = len(cluster_counts) * len(question_sets)
    combination_count = 0

    # Process each combination of question set and cluster count
    for num_clusters in tqdm(cluster_counts, desc="Cluster counts", unit="count"):
        cluster_label = f"clusters_{num_clusters}" if num_clusters else "clusters_auto"
        rich_print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        rich_print(
            f"[bold cyan]Processing with {num_clusters or 'auto'} clusters[/bold cyan]"
        )
        rich_print(f"[bold cyan]{'=' * 60}[/bold cyan]")

        # Get or create clusters for this cluster count
        if num_clusters in clusters_cache:
            clusters = clusters_cache[num_clusters]
            rich_print(f"Using cached clusters ({len(clusters)} clusters)")
        elif config.clusters_path is not None:
            # Load pre-computed clusters
            rich_print(f"Loading pre-computed clusters from {config.clusters_path}...")
            if input_storage is not None:
                clusters_data = json.loads(
                    asyncio.run(
                        input_storage.get(config.clusters_path.as_posix().lstrip("/"))
                    )
                    or "[]"
                )
            else:
                with config.clusters_path.open(encoding="utf-8") as f:
                    clusters_data = json.load(f)

            clusters = []
            for cluster_data in clusters_data:
                tus = [
                    TextUnit(
                        id=tu.get("id", ""),
                        short_id=tu.get("short_id", tu.get("id", "")),
                        text=tu.get("text", ""),
                        text_embedding=tu.get("text_embedding"),
                    )
                    for tu in cluster_data.get("text_units", [])
                ]
                cluster = TextCluster(
                    id=cluster_data.get("cluster_id", cluster_data.get("id", "")),
                    text_units=tus,
                )
                clusters.append(cluster)

            rich_print(f"Loaded {len(clusters)} pre-computed clusters")
            clusters_cache[num_clusters] = clusters
        else:
            # Create ClusterRelevanceRater to perform clustering
            cluster_rater = ClusterRelevanceRater(
                text_embedder=embedder,
                relevance_rater=relevance_rater,
                corpus=text_units,
                semantic_neighbors=config.semantic_neighbors,
                centroid_neighbors=config.centroid_neighbors,
                num_clusters=num_clusters,
            )
            clusters = cluster_rater.clusters
            clusters_cache[num_clusters] = clusters
            rich_print(f"Created {len(clusters)} clusters")

            # Save clusters if requested
            if config.save_clusters and clusters_storage is not None:
                clusters_data_out = [
                    {
                        "cluster_id": c.id,
                        "num_text_units": len(c.text_units),
                        "text_unit_ids": [tu.id for tu in c.text_units],
                        "text_unit_short_ids": [tu.short_id for tu in c.text_units],
                    }
                    for c in clusters
                ]
                await clusters_storage.set(
                    f"{cluster_label}.json",
                    json.dumps(clusters_data_out, indent=2),
                )
                rich_print(
                    f"[green]Saved clustering results to clusters/{cluster_label}.json[/green]"
                )

        # Process each question set
        for question_set in tqdm(
            question_sets, desc="Question sets", unit="set", leave=False
        ):
            combination_count += 1
            rich_print(
                f"\n[bold]Processing question set: {question_set.name} ({combination_count}/{total_combinations})[/bold]"
            )

            # Create output sub-storage for this combination
            subdir_key = f"{question_set.name}/{cluster_label}"
            output_subdir = config.output_dir / question_set.name / cluster_label
            subdir_storage = output_storage.child(subdir_key)

            # Load questions
            rich_print(f"Loading questions from {question_set.questions_path}...")
            if input_storage is not None:
                questions_data = json.loads(
                    asyncio.run(
                        input_storage.get(
                            question_set.questions_path.as_posix().lstrip("/")
                        )
                    )
                    or "[]"
                )
            else:
                with question_set.questions_path.open(encoding="utf-8") as f:
                    questions_data = json.load(f)

            if config.max_questions is not None and config.max_questions < len(
                questions_data
            ):
                questions_data = questions_data[: config.max_questions]
                rich_print(
                    f"Limited to {len(questions_data)} questions (max_questions={config.max_questions})"
                )
            else:
                rich_print(f"Loaded {len(questions_data)} questions")

            # Process questions
            from benchmark_qed.autoq.data_model.question import Question

            questions = [
                Question(
                    id=q.get("question_id", q.get("id", str(i))),
                    text=q.get("text", q.get("question_text", "")),
                )
                for i, q in enumerate(questions_data)
            ]

            # Create cluster rater with cached clusters
            cluster_rater = ClusterRelevanceRater(
                text_embedder=embedder,
                relevance_rater=relevance_rater,
                corpus=clusters,  # Pass pre-computed clusters
                semantic_neighbors=config.semantic_neighbors,
                centroid_neighbors=config.centroid_neighbors,
                num_clusters=num_clusters,
            )

            rich_print(f"Assessing cluster relevance for {len(questions)} questions...")
            results = await cluster_rater.assess_batch(questions)

            # Save results
            reference_payload = build_cluster_references_payload(
                results,
                include_clusters=True,
                clusters=cluster_rater.clusters,
            )
            await subdir_storage.set(
                "reference.json",
                json.dumps(reference_payload, indent=4, ensure_ascii=False),
            )

            rich_print(
                f"[green]Saved reference data to {output_subdir / 'reference.json'}[/green]"
            )

            # Save usage for this run
            await _write_json(
                subdir_storage,
                "model_usage.json",
                llm_client.metrics_store.get_metrics(),
            )

    # Print final summary
    rich_print(f"\n[bold green]{'=' * 60}[/bold green]")
    rich_print("[bold green]All reference data generated successfully![/bold green]")
    rich_print(f"[bold green]{'=' * 60}[/bold green]")
    rich_print(f"Output directory: {config.output_dir}")

    cache_stats = relevance_rater.get_cache_stats()
    if cache_stats.get("caching_enabled"):
        rich_print(
            f"Cache stats: {cache_stats['cache_hits']} hits, {cache_stats['cache_misses']} misses"
        )

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.metrics_store.get_metrics())


@app.command()
def retrieval_scores(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the retrieval scores configuration JSON file."),
    ],
    *,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics."),
    ] = False,
    max_concurrent: Annotated[
        int,
        typer.Option(help="Maximum concurrent relevance assessments."),
    ] = 8,
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Evaluate retrieval metrics (precision, recall, fidelity) for RAG methods.

    Compares multiple RAG methods on retrieval quality metrics and runs
    statistical significance tests.
    """
    from benchmark_qed.autoe.retrieval import (
        FidelityMetric,
        load_clusters_from_json,
        run_retrieval_evaluation,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.bing_rater import (
        BingRelevanceRater,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.rationale_rater import (
        RationaleRelevanceRater,
    )

    config_path = resolve_config_path(
        config_path,
        account_url=account_url,
        connection_string=connection_string,
    )
    config = load_config(RetrievalScoresConfig, config_path)
    output_storage = _build_output_storage(config.output_storage, config.output_dir)
    input_storage: Storage | None = (
        create_storage(config.input_storage)
        if config.input_storage is not None
        else None
    )

    # Parse fidelity metric
    fidelity_metric = (
        FidelityMetric.JENSEN_SHANNON
        if config.fidelity_metric == "js"
        else FidelityMetric.TOTAL_VARIATION
    )

    # Initialize LLM client
    llm_client = ModelFactory.create_chat_model(config.llm_config)

    # Initialize relevance rater based on config (must match reference generation)
    if config.assessor_type == "bing":
        relevance_rater = BingRelevanceRater(
            llm_client=llm_client,
            llm_config=config.llm_config,
            cache_dir=config.cache_dir,
            cache_enabled=config.cache_dir is not None,
        )
        rich_print("Using BingRelevanceRater (UMBRELA DNA prompt)")
    else:
        relevance_rater = RationaleRelevanceRater(
            llm_client=llm_client,
            llm_config=config.llm_config,
            cache_dir=config.cache_dir,
            cache_enabled=config.cache_dir is not None,
        )
        rich_print("Using RationaleRelevanceRater (structured JSON)")

    # Load clusters
    rich_print(f"Loading clusters from {config.clusters_path}...")
    clusters = load_clusters_from_json(
        config.clusters_path,
        text_units_path=config.text_units_path,
        input_storage=input_storage,
    )
    rich_print(f"Loaded {len(clusters)} clusters")

    # Prepare RAG methods list
    rag_methods = [
        {
            "name": method.name,
            "retrieval_results_path": str(method.retrieval_results_path),
        }
        for method in config.rag_methods
    ]

    # Run evaluation
    asyncio.run(
        run_retrieval_evaluation(
            relevance_rater=relevance_rater,
            rag_methods=rag_methods,
            question_sets=config.question_sets,
            reference_dir=config.reference_dir,
            clusters=clusters,
            output_dir=config.output_dir,
            relevance_threshold=config.relevance_threshold,
            context_id_key=config.context_id_key,
            context_text_key=config.context_text_key,
            run_significance_test=config.run_significance_test,
            significance_alpha=config.significance_alpha,
            significance_correction=config.significance_correction,
            fidelity_metric=fidelity_metric,
            max_concurrent=max_concurrent,
            reference_filename=config.reference_filename,
            cluster_match_by=config.cluster_match_by,
            output_storage=output_storage,
            input_storage=input_storage,
        )
    )

    rich_print(f"\n[green]Results saved to {config.output_dir}[/green]")

    # Print cache stats
    cache_stats = relevance_rater.get_cache_stats()
    if cache_stats.get("caching_enabled"):
        rich_print(
            f"Cache stats: {cache_stats['cache_hits']} hits, "
            f"{cache_stats['cache_misses']} misses "
            f"({cache_stats['hit_rate_percent']}% hit rate)"
        )

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.metrics_store.get_metrics())

    asyncio.run(
        _write_json(
            output_storage, "model_usage.json", llm_client.metrics_store.get_metrics()
        )
    )


def _normalize_assertion_record(record: dict[str, Any]) -> dict[str, Any]:
    """Ensure a single assertion record has an "assertions" array of dicts.

    Handles backwards-compatible single-assertion fields and string assertions,
    making sure every assertion is a dict with a "statement" key.
    """
    if "assertions" not in record:
        if "assertion" in record:
            record["assertions"] = [{"statement": record.pop("assertion")}]
        elif "assertion_text" in record:
            record["assertions"] = [{"statement": record.pop("assertion_text")}]
        return record

    assertions_arr = record["assertions"]
    if not isinstance(assertions_arr, list):
        return record

    converted: list[Any] = []
    for assertion in assertions_arr:
        if isinstance(assertion, str):
            converted.append({"statement": assertion})
        elif isinstance(assertion, dict):
            if "statement" not in assertion and "assertion" in assertion:
                assertion["statement"] = assertion.pop("assertion")
            elif "statement" not in assertion and "assertion_text" in assertion:
                assertion["statement"] = assertion.pop("assertion_text")
            elif "statement" not in assertion:
                assertion["statement"] = assertion.get("text", str(assertion))
            converted.append(assertion)
        else:
            converted.append(assertion)
    record["assertions"] = converted
    return record


def _load_chunk_assertion_prompts(config: Any) -> tuple[str, str]:
    """Load system and user prompts, falling back to package defaults."""
    system_prompt = ""
    user_prompt = ""

    if config.prompt_config:
        if config.prompt_config.system_prompt:
            prompt_path = Path(
                config.prompt_config.system_prompt.prompt
                or config.prompt_config.system_prompt.template
                or ""
            )
            if prompt_path.exists():
                system_prompt = prompt_path.read_text()
        if config.prompt_config.user_prompt:
            prompt_path = Path(
                config.prompt_config.user_prompt.prompt
                or config.prompt_config.user_prompt.template
                or ""
            )
            if prompt_path.exists():
                user_prompt = prompt_path.read_text()

    prompts_dir = Path(__file__).parent / "prompts" / "chunk_assertion"
    if not system_prompt:
        default_system_prompt = prompts_dir / "system_prompt.txt"
        if default_system_prompt.exists():
            system_prompt = default_system_prompt.read_text()
    if not user_prompt:
        default_user_prompt = prompts_dir / "user_prompt.txt"
        if default_user_prompt.exists():
            user_prompt = default_user_prompt.read_text()

    return system_prompt, user_prompt


@app.command()
def chunk_assertion_scores(
    config_path: Annotated[
        Path,
        typer.Argument(help="The path to the configuration file."),
    ],
    output: Annotated[
        Path | None,
        typer.Argument(help="The path to the output directory."),
    ] = None,
    *,
    print_model_usage: Annotated[
        bool,
        typer.Option(help="Whether to print the model usage statistics."),
    ] = False,
    account_url: AccountUrlOption = None,
    connection_string: ConnectionStringOption = None,
) -> None:
    """Score retrieved chunks against assertions using chunk-level evaluation.

    This command evaluates retrieved chunks directly (not synthesized answers)
    against per-question assertions. Results are cached at (assertion, chunk)
    granularity using SHA256 content-addressing, enabling efficient re-runs
    with different k values or retriever configurations.

    Three metrics are reported at each k:
    - Coverage: Macro-averaged pass rate (full+partial support)
    - Strict Coverage: Macro-averaged full-support rate only
    - Coverage Strength: Mean score across assertions

    Example usage:
        benchmark-qed autoe chunk-assertion-scores config.yaml output/

    Config format (YAML):
        generated:
          name: vector_rag
          chunks_path: input/chunks.json  # or embedded in answer_base_path
        assertions:
          assertions_path: input/assertions.json
        k_list: [5, 10, 20, 50]
        pass_threshold: 0.5
        llm_config: ...
        prompt_config:
          system_prompt: {prompt: prompts/chunk_assertion/system_prompt.txt}
          user_prompt: {prompt: prompts/chunk_assertion/user_prompt.txt}
    """
    config_path = resolve_config_path(
        config_path,
        account_url=account_url,
        connection_string=connection_string,
    )

    if output is None:
        output = Path.cwd() / "output"

    from benchmark_qed.autoe.chunk_assertion import run_assertion_eval_chunk_mode
    from benchmark_qed.autoe.data_model.chunk_assertion import ChunkAssertionConfig

    config = load_config(ChunkAssertionConfig, config_path)
    output.mkdir(parents=True, exist_ok=True)

    llm_client = ModelFactory.create_chat_model(config.llm_config)

    # Load assertions (input_storage selects local filesystem vs. Azure Blob)
    assertions_storage, assertions_filename = _build_condition_storage(
        config.input_storage, config.assertions.assertions_path, is_dir=False
    )
    assertions_data = asyncio.run(
        _read_json_df(assertions_storage, assertions_filename)
    )

    # Convert to list of dicts (preserving assertion structure: each record has "assertions" array)
    if isinstance(assertions_data, pd.DataFrame):
        assertions_list = assertions_data.to_dict(orient="records")
    elif isinstance(assertions_data, list):
        assertions_list = assertions_data
    else:
        assertions_list = [assertions_data]

    # Ensure each record has the "assertions" field (expected by chunk evaluation)
    # If missing, create it from "assertion" or "assertion_text" field for backwards compatibility
    # Also convert string assertions to dict format with "statement" field
    processed_assertions = [
        _normalize_assertion_record(record) for record in assertions_list
    ]

    # Convert to question set format expected by chunk evaluation
    question_set = {"assertions": processed_assertions}

    rich_print(f"✓ Loaded {len(processed_assertions)} question records")
    total_assertions = sum(len(r.get("assertions", [])) for r in processed_assertions)
    rich_print(f"✓ Total assertions: {total_assertions}")

    # Load chunks (support both embedded in answers and separate chunks file)
    eval_results = []
    if config.generated.chunks_path:
        chunks_path = config.generated.chunks_path
        chunks_storage, chunks_filename = _build_condition_storage(
            config.input_storage, chunks_path, is_dir=False
        )
        eval_results = asyncio.run(_read_json_df(chunks_storage, chunks_filename))
        eval_results = (
            eval_results.to_dict(orient="records")
            if isinstance(eval_results, pd.DataFrame)
            else eval_results
        )
        rich_print(f"✓ Loaded {len(eval_results)} chunk records from {chunks_path}")
        if config.max_chunks_per_question:
            rich_print(
                f"✓ Capping evaluation to the top {config.max_chunks_per_question} "
                "chunks per question (max_chunks_per_question)"
            )
    elif config.generated.answer_base_path:
        answer_path = config.generated.answer_base_path
        answers_storage, answers_filename = _build_condition_storage(
            config.input_storage, answer_path, is_dir=False
        )
        eval_results = asyncio.run(_read_json_df(answers_storage, answers_filename))
        eval_results = (
            eval_results.to_dict(orient="records")
            if isinstance(eval_results, pd.DataFrame)
            else eval_results
        )
        rich_print(f"✓ Loaded {len(eval_results)} answer records from {answer_path}")
    else:
        msg = "Must specify either chunks_path or answer_base_path"
        raise typer.BadParameter(msg)

    # Load prompts (use defaults from package if not configured)
    system_prompt, user_prompt = _load_chunk_assertion_prompts(config)

    # Set cache directory
    cache_path = None
    if config.cache_dir:
        cache_path = Path(config.cache_dir) / "chunk_assertions.jsonl"
    else:
        cache_path = Path.cwd() / ".benchmark_qed_cache" / "chunk_assertions.jsonl"

    # Run chunk-level evaluation
    rich_print("\n[bold]Running chunk-level assertion evaluation...[/bold]")
    summaries = asyncio.run(
        run_assertion_eval_chunk_mode(
            eval_results,
            question_set,
            llm_client=llm_client,
            llm_config=config.llm_config,
            pass_threshold=config.pass_threshold,
            debug_dir=output / "debug",
            cache_path=cache_path,
            k_list=config.k_list,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_chunks_per_question=config.max_chunks_per_question,
        )
    )

    # Report results
    rich_print("\n[bold cyan]Chunk-Level Assertion Evaluation Results[/bold cyan]")
    rich_print(
        f"{'k':>6}  {'Coverage':>10}  {'Strict':>10}  {'Strength':>10}  {'Chunks':>8}"
    )
    rich_print("-" * 60)

    results_rows = []
    for k_label, summary in sorted(summaries.items()):
        k_display = "all" if k_label == "all" else k_label
        rich_print(
            f"{k_display:>6}  {summary.coverage:>10.1%}  {summary.strict_coverage:>10.1%}  "
            f"{summary.mean_score:>10.3f}  {summary.mean_retrieved_chunks:>8.1f}"
        )
        results_rows.append({
            "k_label": k_label,
            "k": summary.k,
            "coverage": round(summary.coverage, 3),
            "strict_coverage": round(summary.strict_coverage, 3),
            "mean_score": round(summary.mean_score, 3),
            "mean_retrieved_chunks": round(summary.mean_retrieved_chunks, 3),
            "questions_evaluated": summary.n_questions,
            "assertions_evaluated": summary.n_assertions,
            "successful_calls": summary.successful_calls,
            "failed_calls": summary.failed_calls,
            "total_calls": summary.total_calls,
        })

    # Save results
    results_file = output / "chunk_assertion_results.json"
    results_file.write_text(json.dumps(results_rows, indent=2))
    try:
        display_path = results_file.relative_to(Path.cwd())
    except ValueError:
        display_path = results_file.absolute()
    rich_print(f"\nResults saved to {display_path}")

    # Save per-query metrics for each k
    for k_label, summary in summaries.items():
        if summary.per_query_metrics:
            per_query_file = output / f"per_query_metrics_{k_label}.json"
            per_query_file.write_text(
                json.dumps(
                    {
                        "k_label": k_label,
                        "k": summary.k,
                        "metrics": ["coverage", "strict_coverage", "mean_score"],
                        "per_query": summary.per_query_metrics,
                    },
                    indent=2,
                )
            )

    if print_model_usage:
        rich_print("\n[bold]Model usage statistics:[/bold]")
        rich_print(llm_client.metrics_store.get_metrics())
