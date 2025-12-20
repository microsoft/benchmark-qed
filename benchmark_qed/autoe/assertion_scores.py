# Copyright (c) 2025 Microsoft Corporation.
"""Evaluate assertions using a language model."""

import asyncio
import functools
import itertools
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
from rich import print as rich_print
from rich.progress import Progress, TaskID

from benchmark_qed.autoe.data_model.assertion import Assertion, AssertionLLMResponse
from benchmark_qed.autoe.prompts import assertion as assertion_prompts
from benchmark_qed.cli.utils import print_df
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel
from benchmark_qed.llm.utils import chat_typed_response

ASSERTION_PROMPTS = Path(assertion_prompts.__file__).parent


def get_assertion_scores(
    *,
    llm_client: ChatModel,
    llm_config: LLMConfig,
    answers: pd.DataFrame,
    assertions: pd.DataFrame,
    trials: int,
    top_k: int | None = None,
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    include_score_id_in_prompt: bool = True,
    question_id_key: str = "question_id",
    question_text_key: str = "question_text",
    answer_text_key: str = "answer",
) -> pd.DataFrame:
    """
    Score assertions based on the provided answers using a language model.

    Args:
        llm_client (ChatModel): The LLM client to use for scoring.
        llm_config (LLMConfig): The LLM configuration to use for scoring.
        answers (pd.DataFrame): DataFrame containing answers with columns 'question', 'answer'.
        assertions (pd.DataFrame): DataFrame containing assertions with column 'assertion'.
        trials (int): Number of trials to run for each assertion.
        top_k (int | None): If specified, only evaluate the top-k assertions per question
                            (ranked by rank if available, where lower rank = higher importance,
                            otherwise uses first k assertions).
        assessment_system_prompt (Template | None): Optional system prompt template for the assessment.
        assessment_user_prompt (Template | None): Optional user prompt template for the assessment.
        include_score_id_in_prompt (bool): Whether to include the score ID in the user prompt.
        question_id_key (str): Column name for question ID.
        question_text_key (str): Column name for question text.
        answer_text_key (str): Column name for answer text.

    Returns
    -------
        pd.DataFrame: Results with assertion scores and metadata.
    """
    pairs = (
        answers.merge(
            assertions,
            how="inner",
            on=[question_id_key],
            suffixes=("_base", "_other"),
        )
        .drop(columns=[f"{question_text_key}_other"])
        .rename(
            columns={
                f"{question_id_key}": "question_id",
                f"{question_text_key}_base": "question_text",
                f"{answer_text_key}": "answer_text",
            }
        )
    )
    pairs = pairs[["question_id", "question_text", "answer_text", "assertion"]]

    # Apply top-k filtering if specified
    if top_k is not None and top_k > 0:
        # Check if assertions have a 'rank' column for ranking
        if "rank" in assertions.columns:
            # Rank by rank (ascending - lower rank = higher importance) and take top-k per question
            pairs_with_rank = pairs.merge(
                assertions[["assertion", "rank"]], on="assertion", how="left"
            )
            pairs = (
                pairs_with_rank.sort_values(
                    ["question_id", "rank"], ascending=[True, True]
                )
                .groupby("question_id")
                .head(top_k)
                .drop(columns=["rank"])
                .reset_index(drop=True)
            )
        else:
            # If no rank column, just take first k assertions per question
            pairs = pairs.groupby("question_id").head(top_k).reset_index(drop=True)

    with Progress() as progress:

        def on_complete_callback(progress_task: TaskID) -> None:
            progress.update(progress_task, advance=1, refresh=True)

        progress_task = progress.add_task("Scoring...", total=len(pairs) * trials)
        tasks = [
            evaluate_assertion(
                llm_client=llm_client,
                assertion=assertion.assertion,
                question=assertion.question_text,
                answer=assertion.answer_text,
                assessment_system_prompt=assessment_system_prompt,
                assessment_user_prompt=assessment_user_prompt,
                complete_callback=functools.partial(
                    on_complete_callback, progress_task
                ),
                trial=n,
                include_score_id_in_prompt=include_score_id_in_prompt,
                additional_call_args=llm_config.call_args,
            )
            for assertion in itertools.starmap(Assertion, pairs.itertuples(index=False))
            for n in range(trials)
        ]

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))

        return pd.DataFrame(results)


async def evaluate_assertion(
    llm_client: ChatModel,
    assertion: str,
    question: str,
    answer: str,
    trial: int = 0,
    *,
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    include_score_id_in_prompt: bool = True,
    additional_call_args: dict[str, Any] | None = None,
    complete_callback: Callable | None = None,
) -> dict[str, Any]:
    """Evaluate an assertion based on the provided criteria and conditions."""
    assessment_system_prompt = assessment_system_prompt or load_template_file(
        ASSERTION_PROMPTS / "assertion_system_prompt.txt"
    )

    assessment_user_prompt = assessment_user_prompt or load_template_file(
        ASSERTION_PROMPTS / "assertion_user_prompt.txt"
    )
    score_id = uuid4().hex

    messages = [
        {
            "role": "system",
            "content": assessment_system_prompt.substitute(assertion=assertion),
        },
        {
            "role": "user",
            "content": assessment_user_prompt.substitute(
                score_id=score_id if include_score_id_in_prompt else "",
                assertion=assertion,
                question=question,
                answer=answer,
            ),
        },
    ]

    response = await chat_typed_response(
        llm=llm_client,
        messages=messages,
        data_model=AssertionLLMResponse,
        **(additional_call_args or {}),
    )

    if complete_callback:
        complete_callback()

    return {
        "score_id": score_id,
        "reasoning": response.reasoning,
        "score": response.score,
        "question": question,
        "answer": answer,
        "assertion": assertion,
        "trial": trial,
    }


def load_and_normalize_assertions(
    input_dir: str,
    question_set: str,
    assertions_filename_template: str = "{question_set}_assertions.json",
) -> pd.DataFrame:
    """
    Load assertions from JSON file and normalize nested dictionaries.

    Args:
        input_dir: Directory containing assertion files
        question_set: Name of the question set
        assertions_filename_template: Template for assertion filename (default: "{question_set}_assertions.json")

    Returns
    -------
        DataFrame with normalized assertion data containing question_id, question_text, assertion, rank
    """
    assertions_file = assertions_filename_template.format(question_set=question_set)
    assertions_raw = pd.read_json(f"{input_dir}/{assertions_file}")

    # Explode assertions and normalize the nested dictionaries
    assertions = assertions_raw.explode("assertions").reset_index(drop=True)

    # Normalize the assertion dictionaries into separate columns
    assertion_normalized = pd.json_normalize(assertions["assertions"])
    assertions = pd.concat(
        [
            assertions.drop("assertions", axis=1),
            assertion_normalized[["statement", "rank"]],  # Keep only statement and rank
        ],
        axis=1,
    )

    # Rename the statement column to assertion for compatibility
    return assertions.rename(columns={"statement": "assertion"})



def evaluate_rag_method(
    llm_client: ChatModel,
    llm_config: LLMConfig,
    generated_rag: str,
    question_set: str,
    assertions: pd.DataFrame,
    input_dir: str,
    output_dir: Path,
    trials: int,
    top_k_assertions: int | None,
    pass_threshold: float,
    answers_path_template: str = "{input_dir}/{generated_rag}/{question_set}.json",
) -> dict[str, Any] | None:
    """
    Evaluate a single RAG method against assertions for a question set.

    Args:
        llm_client: LLM client for evaluation
        llm_config: LLM configuration
        generated_rag: Name of the RAG method
        question_set: Name of the question set
        assertions: DataFrame with assertions
        input_dir: Input directory path
        output_dir: Output directory path
        trials: Number of evaluation trials
        top_k_assertions: Number of top assertions to evaluate (None for all)
        pass_threshold: Threshold for assertion pass/fail
        answers_path_template: Template for answers file path (default: "{input_dir}/{generated_rag}/{question_set}.json")

    Returns
    -------
        Dictionary with evaluation results or None if evaluation failed
    """
    question_set_output_dir = output_dir / question_set
    if not question_set_output_dir.exists():
        question_set_output_dir.mkdir(parents=True)

    try:
        # Load answers for this RAG method and question set
        answers_path = answers_path_template.format(
            input_dir=input_dir, generated_rag=generated_rag, question_set=question_set
        )
        answers = pd.read_json(answers_path)

        # Get assertion scores
        assertion_score = get_assertion_scores(
            llm_client=llm_client,
            llm_config=llm_config,
            answers=answers,
            assertions=assertions,
            trials=trials,
            top_k=top_k_assertions,
            question_id_key="question_id",
            question_text_key="question_text",
            answer_text_key="answer",
        )

        # Save detailed scores for this RAG method and question set
        assertion_score.to_csv(
            question_set_output_dir / f"{generated_rag}_assertion_scores.csv",
            index=False,
        )

        # Calculate summary statistics
        summary_by_assertion = (
            assertion_score.groupby(["question", "assertion"])
            .agg(
                score=("score", lambda x: int(x.mean() > pass_threshold)),
                scores=("score", list),
            )
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

        # Calculate overall accuracy score
        total_success = summary_by_question["success"].sum()
        total_fail = summary_by_question["fail"].sum()
        total_assertions = total_success + total_fail
        overall_accuracy = (
            total_success / total_assertions if total_assertions > 0 else 0.0
        )

        # Calculate per-assertion statistics
        summary_by_assertion["score_mean"] = summary_by_assertion["scores"].apply(
            lambda x: np.mean(x) if len(x) > 0 else 0.0
        )
        summary_by_assertion["score_std"] = summary_by_assertion["scores"].apply(
            lambda x: np.std(x) if len(x) > 0 else 0.0
        )
        summary_by_assertion = summary_by_assertion.drop(columns=["scores"])

        # Save detailed summary for this RAG method and question set
        summary_by_question.to_csv(
            question_set_output_dir / f"{generated_rag}_summary_by_question.csv",
            index=False,
        )
        summary_by_assertion.to_csv(
            question_set_output_dir / f"{generated_rag}_summary_by_assertion.csv",
            index=False,
        )

        # Report failed assertions for this method
        failed_assertions: pd.DataFrame = cast(
            pd.DataFrame, summary_by_assertion[summary_by_assertion["score"] == 0]
        )

        if len(failed_assertions) > 0:
            rich_print(
                f"    [bold red]{generated_rag} ({question_set}): {len(failed_assertions)} assertions failed[/bold red]"
            )
        else:
            rich_print(
                f"    [bold green]{generated_rag} ({question_set}): All assertions passed[/bold green]"
            )

        rich_print(
            f"    {generated_rag} ({question_set}) - Overall accuracy: {overall_accuracy:.3f} ({total_success}/{total_assertions})"
        )
        if top_k_assertions is not None:
            rich_print(
                f"    [dim]Using top-{top_k_assertions} assertions per question[/dim]"
            )

        # Return results for summary
        return {
            "question_set": question_set,
            "rag_method": generated_rag,
            "total_assertions": total_assertions,
            "successful_assertions": total_success,
            "failed_assertions": total_fail,
            "overall_accuracy": overall_accuracy,
            "total_questions": len(summary_by_question),
            "top_k_used": top_k_assertions if top_k_assertions is not None else "all",
        }

    except FileNotFoundError as e:
        rich_print(
            f"    [bold yellow]Warning: Could not find answers file at {answers_path}: {e}[/bold yellow]"
        )
        return None
    except (OSError, ValueError, KeyError) as e:
        rich_print(
            f"    [bold red]Error processing {generated_rag}/{question_set}: {e}[/bold red]"
        )
        return None


def run_assertion_evaluation(
    llm_client: ChatModel,
    llm_config: LLMConfig,
    question_sets: list[str],
    generated_rags: list[str],
    input_dir: str,
    output_dir: Path,
    trials: int,
    top_k_assertions: int | None,
    pass_threshold: float,
    assertions_filename_template: str = "{question_set}_assertions.json",
    answers_path_template: str = "{input_dir}/{generated_rag}/{question_set}.json",
) -> pd.DataFrame:
    """
    Run assertion-based evaluation for multiple question sets and RAG methods.

    Args:
        llm_client: LLM client for evaluation
        llm_config: LLM configuration
        question_sets: List of question set names
        generated_rags: List of RAG method names
        input_dir: Input directory path
        output_dir: Output directory path
        trials: Number of evaluation trials
        top_k_assertions: Number of top assertions to evaluate (None for all)
        pass_threshold: Threshold for assertion pass/fail
        assertions_filename_template: Template for assertion filename (default: "{question_set}_assertions.json")
        answers_path_template: Template for answers file path (default: "{input_dir}/{generated_rag}/{question_set}.json")

    Returns
    -------
        DataFrame with overall results summary
    """
    overall_results = []

    # Loop through each question set
    for question_set in question_sets:
        rich_print(f"Processing question set: {question_set}")

        # Load and normalize assertions
        assertions = load_and_normalize_assertions(
            input_dir, question_set, assertions_filename_template
        )

        # Display assertion filtering info
        if top_k_assertions is not None:
            rich_print(f"  Filtering to top {top_k_assertions} assertions per question")
        else:
            rich_print("  Using all assertions (no filtering)")

        # Loop through each RAG method for this question set
        for generated_rag in generated_rags:
            rich_print(f"  Processing {generated_rag} for {question_set}")

            result = evaluate_rag_method(
                llm_client=llm_client,
                llm_config=llm_config,
                generated_rag=generated_rag,
                question_set=question_set,
                assertions=assertions,
                input_dir=input_dir,
                output_dir=output_dir,
                trials=trials,
                top_k_assertions=top_k_assertions,
                pass_threshold=pass_threshold,
                answers_path_template=answers_path_template,
            )

            if result is not None:
                overall_results.append(result)

    # Create and save overall summary
    overall_summary_df = pd.DataFrame(overall_results)
    overall_summary_df = overall_summary_df.sort_values(
        ["question_set", "overall_accuracy"], ascending=[True, False]
    )
    overall_summary_df.to_csv(
        output_dir / "assertion_scores_overall_summary.csv", index=False
    )

    # Display summary table
    print_df(
        overall_summary_df,
        "Overall Assertion Scores Summary by Question Set and RAG Method",
    )

    # Also create a pivot table for easier comparison
    pivot_summary = overall_summary_df.pivot_table(
        index="rag_method", columns="question_set", values="overall_accuracy"
    )
    pivot_summary.to_csv(output_dir / "assertion_scores_pivot_summary.csv")
    print_df(
        pivot_summary.reset_index(),
        "Assertion Accuracy Comparison (Pivot View)",
    )

    return overall_summary_df
