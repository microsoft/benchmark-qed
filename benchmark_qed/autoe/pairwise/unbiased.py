# Copyright (c) 2025 Microsoft Corporation.
"""Unbiased pairwise scoring (Extract-common-and-unique method).

Standard pairwise judging compares full answers, which makes the verdict sensitive
to confounds such as answer length and formatting. This module reduces that bias by
first extracting the content that is COMMON to both answers and the content that is
UNIQUE to each, then judging only the unique content on two criteria: relevance and
diversity.

The output DataFrame uses the exact same schema as
``benchmark_qed.autoe.pairwise.scores.get_pairwise_scores`` so that the existing
``analyze_criteria`` significance pipeline and ``win_rates.csv`` output work unchanged.
"""

import asyncio
import functools
import itertools
import uuid
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
from graphrag_llm.completion import LLMCompletion
from rich.progress import Progress, TaskID

from benchmark_qed.autoe.data_model import (
    ConditionPair,
    PairwiseExtractionLLMResponse,
    UnbiasedPairwiseLLMResponse,
)
from benchmark_qed.autoe.pairwise.scores import SCORE_MAPPING
from benchmark_qed.autoe.prompts import pairwise as pairwise_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm import chat

PAIRWISE_PROMPTS_PATH = Path(pairwise_prompts.__file__).parent

# The unbiased judge always evaluates these two criteria (hardcoded, judged together
# in a single call based only on the unique content of each answer).
UNBIASED_CRITERIA: tuple[str, ...] = ("relevance", "diversity")


def get_unbiased_pairwise_scores(
    *,
    llm_client: LLMCompletion,
    llm_config: LLMConfig,
    base_name: str,
    other_name: str,
    base_answers: pd.DataFrame,
    other_answers: pd.DataFrame,
    extract_system_prompt: Template | None = None,
    extract_user_prompt: Template | None = None,
    judge_system_prompt: Template | None = None,
    judge_user_prompt: Template | None = None,
    trials: int,
    include_score_id_in_prompt: bool = True,
    question_id_key: str = "question_id",
    question_text_key: str = "question_text",
) -> pd.DataFrame:
    """Score a pair of conditions with the unbiased extract-and-judge method.

    For each question the answers are first reduced to their common and unique parts,
    then only the unique parts are judged on relevance and diversity. Answer order is
    counterbalanced across trials (as in standard pairwise scoring) to control for
    position bias.

    Args:
        llm_client: The LLM client to use for scoring.
        llm_config: The LLM configuration to use for scoring.
        base_name: Name of the base/reference condition.
        other_name: Name of the other condition to compare.
        base_answers: DataFrame with base condition answers.
        other_answers: DataFrame with other condition answers.
        extract_system_prompt: Optional custom extraction system prompt template.
        extract_user_prompt: Optional custom extraction user prompt template.
        judge_system_prompt: Optional custom judge system prompt template.
        judge_user_prompt: Optional custom judge user prompt template.
        trials: The number of trials to run for each comparison (must be even).
        include_score_id_in_prompt: Whether to include score ID in the prompt.
        question_id_key: The column name for question ID in the DataFrames.
        question_text_key: The column name for question text in the DataFrames.

    Returns
    -------
        DataFrame containing per-criterion win scores for each condition, in the same
        schema produced by ``get_pairwise_scores``.
    """
    pairs = (
        base_answers
        .merge(
            other_answers,
            how="inner",
            on=[question_id_key],
            suffixes=("_base", "_other"),
        )
        .drop(columns=[f"{question_text_key}_other"])
        .rename(
            columns={
                question_id_key: "question_id",
                f"{question_text_key}_base": "question_text",
            }
        )
    )
    pairs = pairs[["question_id", "question_text", "answer_base", "answer_other"]]

    with Progress() as progress:

        def on_complete_callback(progress_task: TaskID) -> None:
            progress.update(progress_task, advance=1, refresh=True)

        progress_task = progress.add_task(
            "Scoring relevance & diversity (unbiased)...",
            total=len(pairs) * trials,
        )

        tasks = [
            get_unbiased_pairwise_score(
                llm=llm_client,
                question=pair.question_text,
                answer_1_name=base_name,
                answer_1=pair.answer_base,
                answer_2_name=other_name,
                answer_2=pair.answer_other,
                extract_system_prompt=extract_system_prompt,
                extract_user_prompt=extract_user_prompt,
                judge_system_prompt=judge_system_prompt,
                judge_user_prompt=judge_user_prompt,
                complete_callback=functools.partial(
                    on_complete_callback, progress_task
                ),
                trial=n,
                include_score_id_in_prompt=include_score_id_in_prompt,
                additional_call_args=llm_config.call_args,
            )
            for pair in itertools.starmap(ConditionPair, pairs.itertuples(index=False))
            for n in range(trials)
        ]

        async def _run_tasks() -> list[list[dict[str, Any]]]:
            return await asyncio.gather(*tasks)

        nested_results = asyncio.run(_run_tasks())

        results = [row for rows in nested_results for row in rows]
        result = pd.DataFrame(results)
        result["base_name"] = base_name
        result["other_name"] = other_name
        return result


async def get_unbiased_pairwise_score(
    llm: LLMCompletion,
    *,
    question: str,
    answer_1_name: str,
    answer_1: str,
    answer_2_name: str,
    answer_2: str,
    extract_system_prompt: Template | None = None,
    extract_user_prompt: Template | None = None,
    judge_system_prompt: Template | None = None,
    judge_user_prompt: Template | None = None,
    complete_callback: Callable | None = None,
    trial: int = 0,
    include_score_id_in_prompt: bool = True,
    additional_call_args: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract common/unique content then judge the unique content for one pair.

    Returns one row per criterion (relevance, diversity), each in the same schema as
    ``get_pairwise_score`` so results can flow through ``analyze_criteria`` unchanged.
    """
    extract_system_prompt = extract_system_prompt or load_template_file(
        PAIRWISE_PROMPTS_PATH / "pairwise_extract_system_prompt.txt"
    )
    extract_user_prompt = extract_user_prompt or load_template_file(
        PAIRWISE_PROMPTS_PATH / "pairwise_extract_user_prompt.txt"
    )
    judge_system_prompt = judge_system_prompt or load_template_file(
        PAIRWISE_PROMPTS_PATH / "pairwise_unique_judge_system_prompt.txt"
    )
    judge_user_prompt = judge_user_prompt or load_template_file(
        PAIRWISE_PROMPTS_PATH / "pairwise_unique_judge_user_prompt.txt"
    )

    answers_text = {
        answer_1_name: answer_1,
        answer_2_name: answer_2,
    }

    # Counterbalance the presentation order across trials to control position bias.
    answers_order = (
        list(answers_text.keys())
        if trial % 2 == 0
        else list(reversed(answers_text.keys()))
    )

    score_id = uuid.uuid4().hex
    score_id_text = f"Score ID: {score_id}\n" if include_score_id_in_prompt else ""

    # --- Step 1: extract common and unique content ---
    extract_user = extract_user_prompt.substitute(
        score_id=score_id_text,
        question=question,
        answer1=answers_text[answers_order[0]],
        answer2=answers_text[answers_order[1]],
    ).strip()
    extract_system = extract_system_prompt.template

    extraction = (
        await chat(
            llm,
            messages=[
                {"role": "system", "content": extract_system},
                {"role": "user", "content": extract_user},
            ],
            response_format=PairwiseExtractionLLMResponse,
            **(additional_call_args or {}),
        )
    ).formatted_response

    if extraction is None:
        msg = "LLM did not return a structured PairwiseExtractionLLMResponse."
        raise RuntimeError(msg)

    # --- Step 2: judge the unique content on relevance and diversity ---
    judge_user = judge_user_prompt.substitute(
        score_id=score_id_text,
        question=question,
        common=extraction.common,
        unique1=extraction.unique_answer_1,
        unique2=extraction.unique_answer_2,
    ).strip()
    judge_system = judge_system_prompt.template

    verdict = (
        await chat(
            llm,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": judge_user},
            ],
            response_format=UnbiasedPairwiseLLMResponse,
            **(additional_call_args or {}),
        )
    ).formatted_response

    if verdict is None:
        msg = "LLM did not return a structured UnbiasedPairwiseLLMResponse."
        raise RuntimeError(msg)

    criterion_verdicts = {
        "relevance": verdict.relevance,
        "diversity": verdict.diversity,
    }

    rows: list[dict[str, Any]] = []
    for criterion_name in UNBIASED_CRITERIA:
        criterion_verdict = criterion_verdicts[criterion_name]
        rows.append({
            "score_id": score_id,
            "question": question,
            "answer_1_name": answers_order[0],
            "answer_1": answers_text[answers_order[0]],
            "answer_2_name": answers_order[1],
            "answer_2": answers_text[answers_order[1]],
            "criteria": criterion_name,
            f"{answers_order[0]}_score": SCORE_MAPPING[criterion_verdict.winner],
            f"{answers_order[1]}_score": 1 - SCORE_MAPPING[criterion_verdict.winner],
            "reasoning": criterion_verdict.reasoning,
            "common": extraction.common,
            "unique_1": extraction.unique_answer_1,
            "unique_2": extraction.unique_answer_2,
            "trial": trial,
        })

    if complete_callback:
        complete_callback()

    return rows
