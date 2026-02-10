# Copyright (c) 2025 Microsoft Corporation.
"""Score CLI for generating scores and significance tests for different conditions."""

import asyncio
import json
from itertools import combinations, product
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pandas as pd
import typer
from rich import print as rich_print
from tqdm import tqdm

from benchmark_qed.autoe.assertion import (
    HierarchicalMode,
    aggregate_hierarchical_scores,
    get_assertion_scores,
    get_hierarchical_assertion_scores,
    run_assertion_evaluation,
    summarize_hierarchical_by_question,
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
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the assertion scoring config YAML."),
    ],
    output: Annotated[
        Path | None,
        typer.Argument(help="Output directory (required for single-RAG config, ignored for multi-RAG)."),
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

    # Load raw YAML to detect format
    with open(config_path, encoding="utf-8") as f:
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
            rich_print("Usage: benchmark-qed autoe assertion-scores config.yaml output_dir")
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
    config = load_config(config_path, AssertionConfig)
    output.mkdir(parents=True, exist_ok=True)

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    assertions = pd.read_json(config.assertions.assertions_path, encoding="utf-8")

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


def _run_multi_rag_assertion_scores(
    config_path: Path,
    print_model_usage: bool,
) -> None:
    """Run assertion scoring for multiple RAG methods with significance testing."""
    config = load_config(config_path, MultiRAGAssertionConfig)
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
        question_text_key=config.question_text_key,
        answer_text_key=config.answer_text_key,
    )

    if len(results_df) > 0:
        rich_print("\n[bold green]Evaluation complete![/bold green]")
        rich_print(f"Results saved to: {config.output_dir}")
    else:
        rich_print("[yellow]No results generated. Check input paths and files.[/yellow]")

    if print_model_usage:
        rich_print("\nModel usage statistics:")
        rich_print(llm_client.get_usage())
    usage_file = config.output_dir / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


@app.command()
def hierarchical_assertion_scores(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the hierarchical assertion scoring config YAML."),
    ],
    output: Annotated[
        Path | None,
        typer.Argument(help="Output directory (required for single-RAG config, ignored for multi-RAG)."),
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
        typer.Option(
            help="The key in the JSON file that contains the question ID."
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
            help="The key in the JSON file that contains the assertions."
        ),
    ] = "assertions",
    supporting_assertions_key: Annotated[
        str,
        typer.Option(
            help="The key in assertions that contains the supporting assertions list."
        ),
    ] = "supporting_assertions",
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

    # Load raw YAML to detect format
    with open(config_path, encoding="utf-8") as f:
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
            rich_print("Usage: benchmark-qed autoe hierarchical-assertion-scores config.yaml output_dir")
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

    config = load_config(config_path, HierarchicalAssertionConfig)
    output.mkdir(parents=True, exist_ok=True)

    llm_client = ModelFactory.create_chat_model(config.llm_config)
    assertions = load_and_normalize_hierarchical_assertions(
        config.assertions.assertions_path,
        assertions_key=assertions_key,
        supporting_assertions_key=supporting_assertions_key,
    )

    rich_print(f"Evaluating {len(assertions)} hierarchical assertions...")

    # Get hierarchical scores
    scores = get_hierarchical_assertion_scores(
        llm_client=llm_client,
        llm_config=config.llm_config,
        answers=pd.read_json(config.generated.answer_base_path, encoding="utf-8"),
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
    scores.to_csv(output / "hierarchical_assertion_scores.csv", index=False)

    # Aggregate across trials
    aggregated = aggregate_hierarchical_scores(
        scores, pass_threshold=config.pass_threshold
    )
    aggregated.to_csv(output / "hierarchical_assertion_summary.csv", index=False)

    # Summarize by question
    summary_by_question = summarize_hierarchical_by_question(aggregated)
    summary_by_question.to_csv(
        output / "hierarchical_summary_by_question.csv", index=False
    )

    # Print summary
    print_df(summary_by_question, "Hierarchical Assertion Summary by Question")

    # Report statistics
    total_assertions = len(aggregated)
    passed_assertions = (aggregated["global_score"] == 1).sum()
    avg_support_coverage = aggregated["support_coverage"].mean()
    discovery_count = aggregated["has_discovery"].sum()

    # Count overridden assertions (pass forced to fail due to no support/discovery)
    overridden_count = (
        aggregated["global_score_overridden"].sum()
        if "global_score_overridden" in aggregated.columns
        else 0
    )

    rich_print("\n[bold]Overall Statistics:[/bold]")
    rich_print(
        f"  Global assertions passed: {passed_assertions}/{total_assertions} "
        f"({passed_assertions/total_assertions*100:.1f}%)"
    )
    rich_print(f"  Average support coverage: {avg_support_coverage*100:.1f}%")
    rich_print(f"  Assertions with discovery: {discovery_count}")
    if overridden_count > 0:
        rich_print(
            f"  [yellow]Overridden to fail (no support/discovery): "
            f"{overridden_count}[/yellow]"
        )

    # Report failed assertions
    failed = aggregated[aggregated["global_score"] == 0]
    if len(failed) > 0:
        rich_print(
            f"\n[bold red]{len(failed)} global assertions failed.[/bold red]"
        )
    else:
        rich_print("\n[bold green]All global assertions passed.[/bold green]")

    if print_model_usage:
        rich_print("\nModel usage statistics:")
        rich_print(llm_client.get_usage())
    usage_file = output / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


def _run_multi_rag_hierarchical_assertion_scores(
    config_path: Path,
    print_model_usage: bool,
) -> None:
    """Run hierarchical assertion scoring for multiple RAG methods with significance testing."""
    from benchmark_qed.autoe.assertion import run_hierarchical_assertion_evaluation
    from benchmark_qed.autoe.config import MultiRAGHierarchicalAssertionConfig

    config = load_config(config_path, MultiRAGHierarchicalAssertionConfig)
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
        assertions_path,
        supporting_assertions_key=config.supporting_assertions_key,
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
    )

    if len(results_df) > 0:
        rich_print("\n[bold green]Evaluation complete![/bold green]")
        rich_print(f"Results saved to: {config.output_dir}")
    else:
        rich_print("[yellow]No results generated. Check input paths and files.[/yellow]")

    if print_model_usage:
        rich_print("\nModel usage statistics:")
        rich_print(llm_client.get_usage())
    usage_file = config.output_dir / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")


@app.command()
def assertion_significance(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to the assertion significance configuration YAML file."),
    ],
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

    config = load_config(config_path, AssertionSignificanceConfig)

    rich_print(f"[bold]Running assertion significance tests[/bold]")
    rich_print(f"  Output dir: {config.output_dir}")
    rich_print(f"  RAG methods: {config.rag_methods}")
    rich_print(f"  Question sets: {config.question_sets}")
    rich_print(f"  Alpha: {config.alpha}, Correction: {config.correction_method}")

    results = compare_assertion_scores_significance(
        output_dir=config.output_dir,
        generated_rags=config.rag_methods,
        question_sets=config.question_sets,
        alpha=config.alpha,
        correction_method=config.correction_method,
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
        typer.Argument(help="Path to the hierarchical assertion significance config YAML."),
    ],
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

    config = load_config(config_path, HierarchicalAssertionSignificanceConfig)

    rich_print(f"[bold]Running hierarchical assertion significance tests[/bold]")
    rich_print(f"  Scores dir: {config.scores_dir}")
    rich_print(f"  RAG methods: {config.rag_methods}")
    rich_print(f"  Alpha: {config.alpha}, Correction: {config.correction_method}")

    # Load aggregated scores for each RAG method
    aggregated_scores: dict[str, pd.DataFrame] = {}
    for rag_method in config.rag_methods:
        filename = config.scores_filename_template.format(rag_method=rag_method)
        filepath = config.scores_dir / filename
        if not filepath.exists():
            rich_print(f"  [yellow]Warning: {filepath} not found, skipping[/yellow]")
            continue
        aggregated_scores[rag_method] = pd.read_csv(filepath)
        rich_print(f"  Loaded {len(aggregated_scores[rag_method])} rows for {rag_method}")

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
        save_cluster_references_to_json,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.bing_rater import (
        BingRelevanceRater,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.rationale_rater import (
        RationaleRelevanceRater,
    )

    config = load_config(config_path, RetrievalReferenceConfig)
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
        rich_print(f"Using BingRelevanceRater (UMBRELA DNA prompt, {config.concurrent_requests} concurrent)")
    else:
        relevance_rater = RationaleRelevanceRater(
            llm_client=llm_client,
            llm_config=config.llm_config,
            cache_dir=config.cache_dir,
            cache_enabled=config.cache_dir is not None,
            concurrent_requests=config.concurrent_requests,
        )
        rich_print(f"Using RationaleRelevanceRater (structured JSON, {config.concurrent_requests} concurrent)")

    # Initialize embedding model and text embedder
    embedding_model = ModelFactory.create_embedding_model(config.embedding_config)
    embedder = TextEmbedder(embedding_model)

    # Load text units (needed for clustering)
    text_units: list[TextUnit] = []
    if config.clusters_path is None:
        rich_print(f"Loading text units from {config.text_units_path}...")

        suffix = config.text_units_path.suffix.lower()
        if suffix == ".parquet":
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
            rich_print(f"[yellow]Column '{fields.short_id_col}' not found, will auto-generate short_id[/yellow]")
        if fields.embedding_col and fields.embedding_col not in text_df.columns:
            rich_print(f"[yellow]Column '{fields.embedding_col}' not found, will generate embeddings[/yellow]")

        text_units = load_text_units(
            text_df,
            id_col=fields.id_col,
            text_col=fields.text_col,
            short_id_col=fields.short_id_col if fields.short_id_col and fields.short_id_col in text_df.columns else None,
            embedding_col=fields.embedding_col if fields.embedding_col and fields.embedding_col in text_df.columns else None,
        )
        rich_print(f"Loaded {len(text_units)} text units")

        units_with_embeddings = sum(1 for tu in text_units if tu.text_embedding is not None)
        if units_with_embeddings < len(text_units):
            units_without = len(text_units) - units_with_embeddings
            rich_print(f"[yellow]{units_without}/{len(text_units)} text units missing embeddings. Generating...[/yellow]")

            text_units = await embedder.embed_batch(text_units=text_units, batch_size=32)
            rich_print(f"[green]Embedded {len(text_units)} text units[/green]")
        else:
            rich_print(f"All {len(text_units)} text units have embeddings")

    # Get question sets and cluster counts from config
    question_sets = config.get_question_sets()
    cluster_counts = config.get_cluster_counts()

    rich_print(f"\n[bold]Processing {len(question_sets)} question set(s) x {len(cluster_counts)} cluster count(s)[/bold]")
    for qs in question_sets:
        rich_print(f"  - Question set: {qs.name} ({qs.questions_path})")
    rich_print(f"  - Cluster counts: {cluster_counts}")

    # Cache for clusters by num_clusters to avoid re-clustering
    clusters_cache: dict[int | None, list[TextCluster]] = {}

    # Create clusters directory if saving clusters
    clusters_dir: Path | None = None
    if config.save_clusters:
        clusters_dir = config.output_dir / "clusters"
        clusters_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total combinations for progress tracking
    total_combinations = len(cluster_counts) * len(question_sets)
    combination_count = 0

    # Process each combination of question set and cluster count
    for num_clusters in tqdm(cluster_counts, desc="Cluster counts", unit="count"):
        cluster_label = f"clusters_{num_clusters}" if num_clusters else "clusters_auto"
        rich_print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        rich_print(f"[bold cyan]Processing with {num_clusters or 'auto'} clusters[/bold cyan]")
        rich_print(f"[bold cyan]{'='*60}[/bold cyan]")

        # Get or create clusters for this cluster count
        if num_clusters in clusters_cache:
            clusters = clusters_cache[num_clusters]
            rich_print(f"Using cached clusters ({len(clusters)} clusters)")
        elif config.clusters_path is not None:
            # Load pre-computed clusters
            rich_print(f"Loading pre-computed clusters from {config.clusters_path}...")
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
            if config.save_clusters and clusters_dir is not None:
                clusters_file = clusters_dir / f"{cluster_label}.json"
                clusters_data_out = [
                    {
                        "cluster_id": c.id,
                        "num_text_units": len(c.text_units),
                        "text_unit_ids": [tu.id for tu in c.text_units],
                        "text_unit_short_ids": [tu.short_id for tu in c.text_units],
                    }
                    for c in clusters
                ]
                with clusters_file.open("w", encoding="utf-8") as f:
                    json.dump(clusters_data_out, f, indent=2)
                rich_print(f"[green]Saved clustering results to {clusters_file}[/green]")

        # Process each question set
        for question_set in tqdm(question_sets, desc="Question sets", unit="set", leave=False):
            combination_count += 1
            rich_print(f"\n[bold]Processing question set: {question_set.name} ({combination_count}/{total_combinations})[/bold]")

            # Create output directory for this combination
            output_subdir = config.output_dir / question_set.name / cluster_label
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Load questions
            rich_print(f"Loading questions from {question_set.questions_path}...")
            with question_set.questions_path.open(encoding="utf-8") as f:
                questions_data = json.load(f)

            if config.max_questions is not None and config.max_questions < len(questions_data):
                questions_data = questions_data[:config.max_questions]
                rich_print(f"Limited to {len(questions_data)} questions (max_questions={config.max_questions})")
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
            output_file = output_subdir / "reference.json"
            save_cluster_references_to_json(
                results,
                output_file,
                include_clusters=True,
                clusters=cluster_rater.clusters,
            )

            rich_print(f"[green]Saved reference data to {output_file}[/green]")

            # Save usage for this run
            usage_file = output_subdir / "model_usage.json"
            usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")

    # Print final summary
    rich_print(f"\n[bold green]{'='*60}[/bold green]")
    rich_print("[bold green]All reference data generated successfully![/bold green]")
    rich_print(f"[bold green]{'='*60}[/bold green]")
    rich_print(f"Output directory: {config.output_dir}")

    cache_stats = relevance_rater.get_cache_stats()
    if cache_stats.get("caching_enabled"):
        rich_print(f"Cache stats: {cache_stats['cache_hits']} hits, {cache_stats['cache_misses']} misses")

    if print_model_usage:
        rich_print("Model usage statistics:")
        rich_print(llm_client.get_usage())


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
) -> None:
    """Evaluate retrieval metrics (precision, recall, fidelity) for RAG methods.

    Compares multiple RAG methods on retrieval quality metrics and runs
    statistical significance tests.
    """
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.bing_rater import (
        BingRelevanceRater,
    )
    from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.rationale_rater import (
        RationaleRelevanceRater,
    )
    from benchmark_qed.autoe.retrieval import (
        FidelityMetric,
        load_clusters_from_json,
        run_retrieval_evaluation,
    )

    config = load_config(config_path, RetrievalScoresConfig)
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
        rich_print(f"Using BingRelevanceRater (UMBRELA DNA prompt)")
    else:
        relevance_rater = RationaleRelevanceRater(
            llm_client=llm_client,
            llm_config=config.llm_config,
            cache_dir=config.cache_dir,
            cache_enabled=config.cache_dir is not None,
        )
        rich_print(f"Using RationaleRelevanceRater (structured JSON)")

    # Load clusters
    rich_print(f"Loading clusters from {config.clusters_path}...")
    clusters = load_clusters_from_json(
        config.clusters_path,
        text_units_path=config.text_units_path,
    )
    rich_print(f"Loaded {len(clusters)} clusters")

    # Prepare RAG methods list
    rag_methods = [
        {"name": method.name, "retrieval_results_path": str(method.retrieval_results_path)}
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
        rich_print(llm_client.get_usage())

    usage_file = config.output_dir / "model_usage.json"
    usage_file.write_text(json.dumps(llm_client.get_usage()), encoding="utf-8")
