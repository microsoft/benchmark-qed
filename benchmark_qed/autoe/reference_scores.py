# Copyright (c) 2025 Microsoft Corporation.
"""Reference scoring functions for evaluation tasks."""

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
) -> pd.DataFrame:
    """
    Score a generated answer against a ground truth answer using the specified criteria.

    Args:
        llm_client (ChatModel): The LLM client to use for scoring.
        llm_config (LLMConfig): The LLM configuration to use for scoring.
        generated (Condition): The generated answer to score.
        reference (Condition): The reference answer to score against.
        criteria (list[Criteria]): The criteria to use for scoring.
        trials (int): The number of trials to run for each comparison.
        score_min (int): The minimum score for the criteria.
        score_max (int): The maximum score for the criteria.
        include_score_id_in_prompt (bool): Whether to include the score ID in the user prompt. Including the score ID can be helpful to invalidate cached scores in the LLM, but it is not strictly necessary.
        question_id_key (str): The name for the question ID in the DataFrame.

    Returns
    -------
    pd.DataFrame: A DataFrame containing the scores for each condition.
    """
    pairs = (
        reference_answers.merge(
            generated_answers,
            how="inner",
            on=[question_id_key],
            suffixes=("_base", "_other"),
        )
        .drop(columns=["question_text_other"])
        .rename(columns={"question_text_base": "question_text"})
    )

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
    """Get the score for a generated answer to a question according to the specified criteria."""
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
    """
    Summarize reference scores by calculating the mean and standard deviation for each criteria.

    Parameters
    ----------
    raw_scores : pd.DataFrame
        Input DataFrame containing scores for each criteria.

    Returns
    -------
    pd.DataFrame
        A DataFrame with summarized scores, including mean and standard deviation.
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
