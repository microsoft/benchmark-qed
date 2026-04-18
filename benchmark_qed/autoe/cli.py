# Copyright (c) 2025 Microsoft Corporation.
"""Score CLI for generating scores and significance tests for different conditions."""

import asyncio
import json
from io import StringIO
from itertools import combinations, product
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pandas as pd
import typer
from graphrag_common.config import load_config
from graphrag_storage import Storage
from graphrag_storage.file_storage import FileStorage
from graphrag_storage.storage_config import StorageConfig
from graphrag_storage.storage_factory import create_storage
from rich import print as rich_print

from benchmark_qed.autoe.assertion_scores import get_assertion_scores
from benchmark_qed.autoe.config import AssertionConfig, PairwiseConfig, ReferenceConfig
from benchmark_qed.autoe.pairwise_scores import analyze_criteria, get_pairwise_scores
from benchmark_qed.autoe.reference_scores import get_reference_scores
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
        output_posix = output.as_posix().strip("./")
        return storage.child(output_posix) if output_posix else storage
    output.mkdir(parents=True, exist_ok=True)
    return FileStorage(base_dir=str(output))


def _build_condition_storage(
    storage_config: StorageConfig | None, answer_base_path: Path, *, is_dir: bool
) -> tuple[Storage, str]:
    """Build storage for a condition's answer path.

    Returns (storage, filename) where filename is the key to read.
    For directory-style paths (pairwise), filename is empty.
    """
    if storage_config:
        storage = create_storage(storage_config)
        path_posix = answer_base_path.as_posix().strip("./")
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
) -> None:
    """Generate scores for the different conditions provided in the JSON file."""
    if exclude_criteria is None:
        exclude_criteria = []
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

    loop = asyncio.get_event_loop()

    for base, other in all_combinations:
        for question_set in config.question_sets:
            rich_print(f"Scoring {base.name} vs {other.name} for {question_set}")
            cache_key = f"{question_set}_{base.name}--{other.name}.csv"
            if loop.run_until_complete(output_storage.has(cache_key)):
                rich_print(
                    f"{base.name} vs {other.name} for {question_set} already exists. Skipping generation.\n"
                    f"[bold yellow]If you want to generate a new comparison, delete {cache_key} from {output}.[/bold yellow]"
                )
                result = loop.run_until_complete(
                    _read_csv_df(output_storage, cache_key)
                )
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
                    base_answers=loop.run_until_complete(
                        _read_json_df(
                            base_storage,
                            f"{question_set}.json",
                        )
                    ),
                    other_answers=loop.run_until_complete(
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

                loop.run_until_complete(
                    _write_csv_df(output_storage, cache_key, result)
                )
            result["question_set"] = question_set
            all_results.append(result)

    all_results = pd.concat(all_results)
    loop.run_until_complete(_write_csv_df(output_storage, "win_rates.csv", all_results))

    all_results_p_value = analyze_criteria(
        all_results,
        alpha=alpha,
    )

    loop.run_until_complete(
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
        rich_print(llm_client.get_usage())
    loop.run_until_complete(
        _write_json(output_storage, "model_usage.json", llm_client.get_usage())
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
) -> None:
    """Generate scores for the generated answers provided in the JSON file."""
    if exclude_criteria is None:
        exclude_criteria = []
    config = load_config(ReferenceConfig, comparison_spec)

    config.criteria = [
        criterion
        for criterion in config.criteria
        if criterion.name not in exclude_criteria
    ]

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    output_storage = _build_output_storage(config.output_storage, output)
    loop = asyncio.get_event_loop()

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
            generated_answers=loop.run_until_complete(
                _read_json_df(generated_storage, generated_key)
            ),
            reference_answers=loop.run_until_complete(
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
        loop.run_until_complete(
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
        rich_print(llm_client.get_usage())
    loop.run_until_complete(
        _write_json(output_storage, "model_usage.json", llm_client.get_usage())
    )


@app.command()
def assertion_scores(
    comparison_spec: Annotated[
        Path,
        typer.Argument(help="The path to the JSON file containing the configuration."),
    ],
    output: Annotated[
        Path, typer.Argument(help="The path to the output file for the scores.")
    ],
    *,
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
            help="The key in the JSON file that contains the question ID. This is used to match questions with assertions."
        ),
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
        typer.Option(
            help="The key in the JSON file that contains the assertions. This should be a list of assertions for each question."
        ),
    ] = "assertions",
) -> None:
    """Generate assertion for the generated answers provided in the JSON file."""
    config = load_config(AssertionConfig, comparison_spec)
    output_storage = _build_output_storage(config.output_storage, output)
    loop = asyncio.get_event_loop()

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    assertions_storage, assertions_key = _build_condition_storage(
        config.input_storage, config.assertions.assertions_path, is_dir=False
    )
    assertions = loop.run_until_complete(
        _read_json_df(assertions_storage, assertions_key)
    )

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
        answers=loop.run_until_complete(
            _read_json_df(generated_storage, generated_key)
        ),
        assertions=assertions,
        assessment_user_prompt=config.prompt_config.user_prompt.template,
        assessment_system_prompt=config.prompt_config.system_prompt.template,
        trials=config.trials,
        include_score_id_in_prompt=include_score_id_in_prompt,
        question_id_key=question_id_key,
        question_text_key=question_text_key,
        answer_text_key=answer_text_key,
    )

    loop.run_until_complete(
        _write_csv_df(output_storage, "assertion_scores.csv", assertion_score)
    )

    summary_by_assertion = (
        assertion_score.groupby(["question", "assertion"])
        .agg(score=("score", lambda x: int(x.mean() > 0.5)), scores=("score", list))
        .reset_index()
    )

    summary_by_question = (
        summary_by_assertion.groupby(["question"])
        .agg(
            success=("score", lambda x: (x == 1).sum()),
            fail=("score", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    summary_by_assertion["score_mean"] = summary_by_assertion["scores"].apply(
        lambda x: np.mean(x) if len(x) > 0 else 0.0
    )
    summary_by_assertion["score_std"] = summary_by_assertion["scores"].apply(
        lambda x: np.std(x) if len(x) > 0 else 0.0
    )
    summary_by_assertion = summary_by_assertion.drop(columns=["scores"])

    print_df(
        summary_by_question,
        "Assertion Scores Summary by Question",
    )

    failed_assertions: pd.DataFrame = cast(
        pd.DataFrame, summary_by_assertion[summary_by_assertion["score"] == 0]
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

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.get_usage())
    loop.run_until_complete(
        _write_json(output_storage, "model_usage.json", llm_client.get_usage())
    )
