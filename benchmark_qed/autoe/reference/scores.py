# Copyright (c) 2025 Microsoft Corporation.
"""Reference scoring functions for evaluation tasks.

This module provides functions for scoring generated answers against reference
(ground truth) answers using LLM-based evaluation with configurable criteria.
"""

import asyncio
import functools
import itertools
import uuid
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import Progress, TaskID

from benchmark_qed.autoe.config import Criteria
from benchmark_qed.autoe.data_model import ConditionPair, ReferenceLLMResponse
from benchmark_qed.autoe.prompts import reference as reference_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel
from benchmark_qed.llm.utils import chat_typed_response

REFERENCE_PROMPTS_PATH = Path(reference_prompts.__file__).parent


def get_reference_scores(
    *,
    llm_client: ChatModel,
    llm_config: LLMConfig,
    generated_answers: pd.DataFrame,
    reference_answers: pd.DataFrame,
    criteria: list[Criteria],
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    trials: int,
    score_min: int = 1,
    score_max: int = 10,
    include_score_id_in_prompt: bool = True,
    question_id_key: str = "question_id",
    question_text_key: str = "question_text",
) -> pd.DataFrame:
    """Score generated answers against reference answers using specified criteria.

    Args:
        llm_client: The LLM client to use for scoring.
        llm_config: The LLM configuration to use for scoring.
        generated_answers: DataFrame with generated answers.
        reference_answers: DataFrame with reference/ground truth answers.
        criteria: The criteria to use for scoring.
        assessment_system_prompt: Optional custom system prompt template.
        assessment_user_prompt: Optional custom user prompt template.
        trials: The number of trials to run for each comparison.
        score_min: The minimum score for the criteria.
        score_max: The maximum score for the criteria.
        include_score_id_in_prompt: Whether to include score ID in the prompt.
        question_id_key: The column name for question ID in the DataFrames.
        question_text_key: The column name for question text in the DataFrames.

    Returns
    -------
        DataFrame containing the scores for each condition.
    """
    pairs = (
        reference_answers.merge(
            generated_answers,
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
    # Select only the columns needed for ConditionPair
    pairs = pairs[["question_id", "question_text", "answer_base", "answer_other"]]

    with Progress(transient=True) as progress:

        def on_complete_callback(progress_task: TaskID) -> None:
            progress.update(progress_task, advance=1, refresh=True)

        progress_tasks = {
            criterion.name: progress.add_task(
                f"Scoring {criterion.name}...", total=len(pairs) * trials
            )
            for criterion in criteria
        }

        tasks = [
            get_reference_score(
                llm_client,
                question=pair.question_text,
                reference_answer=pair.answer_base,
                generated_answer=pair.answer_other,
                criteria_name=criterion.name,
                criteria_description=criterion.description,
                assessment_system_prompt=assessment_system_prompt,
                assessment_user_prompt=assessment_user_prompt,
                complete_callback=functools.partial(
                    on_complete_callback, progress_tasks[criterion.name]
                ),
                score_min=score_min,
                score_max=score_max,
                trial=n,
                include_score_id_in_prompt=include_score_id_in_prompt,
                additional_call_args=llm_config.call_args,
            )
            for pair in itertools.starmap(ConditionPair, pairs.itertuples(index=False))
            for criterion in criteria
            for n in range(trials)
        ]

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))

        return pd.DataFrame(results)


async def get_reference_score(
    llm: ChatModel,
    *,
    question: str,
    reference_answer: str,
    generated_answer: str,
    criteria_name: str,
    criteria_description: str,
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    complete_callback: Callable | None = None,
    trial: int = 0,
    score_min: int = 1,
    score_max: int = 10,
    include_score_id_in_prompt: bool = True,
    additional_call_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get the score for a generated answer against a reference answer.

    Args:
        llm: The LLM client to use for scoring.
        question: The question being answered.
        reference_answer: The reference/ground truth answer.
        generated_answer: The generated answer to evaluate.
        criteria_name: The name of the evaluation criteria.
        criteria_description: The description of the evaluation criteria.
        assessment_system_prompt: Optional custom system prompt template.
        assessment_user_prompt: Optional custom user prompt template.
        complete_callback: Callback function to invoke when evaluation completes.
        trial: The trial number for this evaluation.
        score_min: The minimum score value.
        score_max: The maximum score value.
        include_score_id_in_prompt: Whether to include score ID in the prompt.
        additional_call_args: Additional arguments to pass to the LLM call.

    Returns
    -------
        Dictionary containing the score and evaluation details.
    """
    assessment_system_prompt = assessment_system_prompt or load_template_file(
        REFERENCE_PROMPTS_PATH / "reference_system_prompt.txt"
    )

    assessment_user_prompt = assessment_user_prompt or load_template_file(
        REFERENCE_PROMPTS_PATH / "reference_user_prompt.txt"
    )
    answer_1_name, answer_2_name = (
        ("Reference", "Generated") if trial % 2 == 0 else ("Generated", "Reference")
    )
    answer_1, answer_2 = (
        (reference_answer, generated_answer)
        if trial % 2 == 0
        else (generated_answer, reference_answer)
    )

    score_id = uuid.uuid4().hex
    score_id_text = f"Score ID: {score_id}\n" if include_score_id_in_prompt else ""

    system_prompt = assessment_system_prompt.substitute(
        criteria_name=criteria_name,
        criteria_description=criteria_description,
        score_min=score_min,
        score_max=score_max,
    )
    user_prompt = assessment_user_prompt.substitute(
        score_id=score_id_text,
        query=question,
        answer_1_name=answer_1_name,
        answer_2_name=answer_2_name,
        answer_1=answer_1,
        answer_2=answer_2,
        criteria_name=criteria_name,
        criteria_description=criteria_description,
        score_min=score_min,
        score_max=score_max,
    ).strip()
    assessment_response = await chat_typed_response(
        llm,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        data_model=ReferenceLLMResponse,
        response_format={"type": "json_object"},
        **(additional_call_args or {}),
    )

    response = {
        "score_id": score_id,
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
        "criteria": criteria_name,
        "score": assessment_response.score,
        "reasoning": assessment_response.reasoning,
        "trial": trial,
    }

    if complete_callback:
        complete_callback()

    return response


def summarize_reference_scores(raw_scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize reference scores by calculating mean and std for each criteria.

    Args:
        raw_scores: DataFrame containing scores for each criteria.

    Returns
    -------
        DataFrame with summarized scores including mean and standard deviation.
    """
    summary_df = (
        raw_scores.drop(
            columns=[
                "question",
                "reference_answer",
                "generated_answer",
                "reasoning",
                "trial",
            ]
        )
        .groupby("criteria")
        .agg(list)
        .reset_index()
    )

    summary_df["mean"] = summary_df["score"].apply(np.mean)
    summary_df["std"] = summary_df["score"].apply(np.std)
    return summary_df.drop(columns=["score"])
