# Copyright (c) 2025 Microsoft Corporation.
"""Score CLI for generating scores and significance tests for different conditions."""

import json
from itertools import combinations, product
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pandas as pd
import typer
from rich import print as rich_print

from benchmark_qed.autoe.assertion_scores import get_assertion_scores
from benchmark_qed.autoe.config import AssertionConfig, PairwiseConfig, ReferenceConfig
from benchmark_qed.autoe.pairwise_scores import analyze_criteria, get_pairwise_scores
from benchmark_qed.autoe.reference_scores import get_reference_scores
from benchmark_qed.cli.utils import print_df
from benchmark_qed.config.utils import load_config
from benchmark_qed.llm.factory import ModelFactory

app: typer.Typer = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Evaluate Retrieval-Augmented Generation (RAG) methods.",
)


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
    config = load_config(comparison_spec, PairwiseConfig)

    config.criteria = [
        criterion
        for criterion in config.criteria
        if criterion.name not in exclude_criteria
    ]

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    output.mkdir(parents=True, exist_ok=True)
    all_results = []

    all_combinations = (
        product([config.base], config.others)
        if config.base
        else combinations(config.others, 2)
    )

    for base, other in all_combinations:
        for question_set in config.question_sets:
            rich_print(f"Scoring {base.name} vs {other.name} for {question_set}")
            if (output / f"{question_set}_{base.name}--{other.name}.csv").exists():
                rich_print(
                    f"{base.name} vs {other.name} for {question_set} already exists. Skipping generation.\n"
                    f"[bold yellow]If you want to generate a new comparison, delete {question_set}_{base.name}--{other.name}.csv from {output}.[/bold yellow]"
                )
                result = pd.read_csv(
                    output / f"{question_set}_{base.name}--{other.name}.csv"
                )
            else:
                result = get_pairwise_scores(
                    llm_client=llm_client,
                    llm_config=config.llm_config,
                    base_name=base.name,
                    other_name=other.name,
                    base_answers=pd.read_json(
                        (base.answer_base_path / f"{question_set}.json"),
                        encoding="utf-8",
                    ),
                    other_answers=pd.read_json(
                        (other.answer_base_path / f"{question_set}.json"),
                        encoding="utf-8",
                    ),
                    criteria=config.criteria,
                    assessment_user_prompt=config.prompt_config.user_prompt.template,
                    assessment_system_prompt=config.prompt_config.system_prompt.template,
                    trials=config.trials,
                    question_id_key=question_id_key,
                    include_score_id_in_prompt=include_score_id_in_prompt,
                )

                result.to_csv(
                    output / f"{question_set}_{base.name}--{other.name}.csv",
                    index=False,
                )
            result["question_set"] = question_set
            all_results.append(result)

    all_results = pd.concat(all_results)
    all_results.to_csv(output / "win_rates.csv", index=False)

    all_results_p_value = analyze_criteria(
        all_results,
        alpha=alpha,
    )

    all_results_p_value.to_csv(output / "winrates_sig_tests.csv", index=False)

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
    usage_file = output / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


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
    config = load_config(comparison_spec, ReferenceConfig)

    config.criteria = [
        criterion
        for criterion in config.criteria
        if criterion.name not in exclude_criteria
    ]

    llm_client = ModelFactory.create_chat_model(config.llm_config)

    for generated in config.generated:
        result = get_reference_scores(
            llm_client=llm_client,
            llm_config=config.llm_config,
            generated_answers=pd.read_json(
                generated.answer_base_path, encoding="utf-8"
            ),
            reference_answers=pd.read_json(
                config.reference.answer_base_path, encoding="utf-8"
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
        output.mkdir(parents=True, exist_ok=True)
        result.to_csv(output / f"reference_scores-{generated.name}.csv", index=False)
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
    usage_file = output / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


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
    config = load_config(comparison_spec, AssertionConfig)
    output.mkdir(parents=True, exist_ok=True)

    llm_client = ModelFactory.create_chat_model(config.llm_config)

    assertions = (
        pd.read_json(config.assertions.assertions_path, encoding="utf-8")
        .explode(assertions_key)
        .rename(columns={f"{assertions_key}": "assertion"})
        .reset_index(drop=True)
    )

    assertion_score = get_assertion_scores(
        llm_client=llm_client,
        llm_config=config.llm_config,
        answers=pd.read_json(config.generated.answer_base_path, encoding="utf-8"),
        assertions=assertions,
        assessment_user_prompt=config.prompt_config.user_prompt.template,
        assessment_system_prompt=config.prompt_config.system_prompt.template,
        trials=config.trials,
        include_score_id_in_prompt=include_score_id_in_prompt,
        question_id_key=question_id_key,
        question_text_key=question_text_key,
        answer_text_key=answer_text_key,
    )

    assertion_score.to_csv(output / "assertion_scores.csv", index=False)

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

    failed_assertions["trials"] = config.trials
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
    usage_file = output / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")
