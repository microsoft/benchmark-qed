# Copyright (c) 2025 Microsoft Corporation.
"""Pairwise scoring functions for evaluation tasks."""

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
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

from benchmark_qed.autoe.config import Criteria
from benchmark_qed.autoe.data_model import ConditionPair, PairwiseLLMResponse
from benchmark_qed.autoe.prompts import pairwise as pairwise_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel
from benchmark_qed.llm.utils import chat_typed_response

PAIRWISE_PROMPTS_PATH = Path(pairwise_prompts.__file__).parent

SCORE_MAPPING = {
    0: 0.5,
    1: 1.0,
    2: 0.0,
}


def get_pairwise_scores(
    *,
    llm_client: ChatModel,
    llm_config: LLMConfig,
    base_name: str,
    other_name: str,
    base_answers: pd.DataFrame,
    other_answers: pd.DataFrame,
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    criteria: list[Criteria],
    trials: int,
    include_score_id_in_prompt: bool = True,
    question_id_key: str = "question_id",
) -> pd.DataFrame:
    """Score a pair of conditions using the specified criteria.

    Args:
        llm_client (ChatModel): The LLM client to use for scoring.
        llm_config (LLMConfig): The LLM configuration to use for scoring.
        condition_a (Condition): The first condition to score.
        condition_b (Condition): The second condition to score.
        criteria (list[Criteria]): The criteria to use for scoring.
        trials (int): The number of trials to run for each comparison.
        include_score_id_in_prompt (bool): Whether to include the score ID in the user prompt. Including the score ID can be helpful to invalidate cached scores in the LLM, but it is not strictly necessary.
        question_id_key (str): The name for the question ID in the DataFrame.

    Returns
    -------
        pd.DataFrame: A DataFrame containing the scores for each condition.
    """
    pairs = (
        base_answers.merge(
            other_answers,
            how="inner",
            on=[question_id_key],
            suffixes=("_base", "_other"),
        )
        .drop(columns=["question_text_other"])
        .rename(columns={"question_text_base": "question_text"})
    )

    with Progress() as progress:

        def on_complete_callback(progress_task: TaskID) -> None:
            progress.update(progress_task, advance=1, refresh=True)

        progress_tasks = {
            criterion.name: progress.add_task(
                f"Scoring {criterion.name}...", total=len(pairs) * trials
            )
            for criterion in criteria
        }
        tasks = [
            get_pairwise_score(
                llm=llm_client,
                question=pair.question_text,
                answer_1_name=base_name,
                answer_1=pair.answer_base,
                answer_2_name=other_name,
                answer_2=pair.answer_other,
                criteria_name=criterion.name,
                criteria_description=criterion.description,
                assessment_system_prompt=assessment_system_prompt,
                assessment_user_prompt=assessment_user_prompt,
                complete_callback=functools.partial(
                    on_complete_callback, progress_tasks[criterion.name]
                ),
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

        result = pd.DataFrame(results)
        result["base_name"] = base_name
        result["other_name"] = other_name
        return result


async def get_pairwise_score(
    llm: ChatModel,
    *,
    question: str,
    answer_1_name: str,
    answer_1: str,
    answer_2_name: str,
    answer_2: str,
    criteria_name: str,
    criteria_description: str,
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    complete_callback: Callable | None = None,
    trial: int = 0,
    include_score_id_in_prompt: bool = True,
    additional_call_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get the score for two answers to a question according to the specified criteria.

    Args:
        question (str): The question for which the answers are provided.
        answer_1 (str): The first answer to the question.
        answer_2 (str): The second answer to the question.
        criteria_name (str): The criteria for assessing the answers.
        criteria_description (str): The description of the criteria for assessing the answers.
        llm (OpenAIChatLLM): The OpenAI LLM model to use for generating the prompts
        assessment_system_prompt (str): The prompt for the system assessment task.
        assessment_user_prompt (str): The prompt for the user assessment task.
        trial (int): The trial index for the assessment.
        include_score_id_in_prompt (bool): Whether to include the score ID in the user prompt.
        additional_call_args (dict[str, Any] | None): Additional arguments to pass to the LLM call, e.g. temperature, seed, max_tokens, etc.

    Returns
    -------
        dict[str, Any]: A dictionary containing the scores and reasoning for each answer.
    """
    assessment_system_prompt = assessment_system_prompt or load_template_file(
        PAIRWISE_PROMPTS_PATH / "pairwise_system_prompt.txt"
    )

    assessment_user_prompt = assessment_user_prompt or load_template_file(
        PAIRWISE_PROMPTS_PATH / "pairwise_user_prompt.txt"
    )

    answers_text = {
        answer_1_name: answer_1,
        answer_2_name: answer_2,
    }

    answers_order = (
        list(answers_text.keys())
        if trial % 2 == 0
        else list(reversed(answers_text.keys()))
    )

    score_id = uuid.uuid4().hex
    score_id_text = f"Score ID: {score_id}\n" if include_score_id_in_prompt else ""

    user_prompt = assessment_user_prompt.substitute(
        score_id=score_id_text,
        question=question,
        answer1=answers_text[answers_order[0]],
        answer2=answers_text[answers_order[1]],
        criteria_name=criteria_name,
        criteria_description=criteria_description,
    ).strip()

    system_prompt = assessment_system_prompt.substitute(
        criteria_name=criteria_name, criteria_description=criteria_description
    )
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
        data_model=PairwiseLLMResponse,
        response_format={"type": "json_object"},
        **(additional_call_args or {}),
    )

    response = {
        "score_id": score_id,
        "question": question,
        "answer_1_name": answers_order[0],
        "answer_1": answers_text[answers_order[0]],
        "answer_2_name": answers_order[1],
        "answer_2": answers_text[answers_order[1]],
        "criteria": criteria_name,
        f"{answers_order[0]}_score": SCORE_MAPPING[assessment_response.winner],
        f"{answers_order[1]}_score": 1 - SCORE_MAPPING[assessment_response.winner],
        "reasoning": assessment_response.reasoning,
        "trial": trial,
    }

    if complete_callback:
        complete_callback()

    return response


def analyze_criteria(raw_scores: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform statistical significance tests (paired t-test or Wilcoxon) on criteria-based scores.

    Parameters
    ----------
    raw_scores : pd.DataFrame
        Input DataFrame containing criteria and condition scores
    alpha : float, optional
        Significance threshold (default is 0.05).

    Returns
    -------
    pd.DataFrame
        A DataFrame with test results, statistics, and Holm-corrected p-values.
    """
    # Drop unused columns
    others = raw_scores["other_name"].unique()
    base_names = raw_scores["base_name"].unique()[0]

    scores_group = raw_scores.groupby([
        "question",
        "criteria",
        "question_set",
        "base_name",
        "other_name",
    ])

    results = [
        scores_group.agg(
            base_mean=(f"{base_name}_score", "mean"),
            base_scores=(f"{base_name}_score", list),
            other_mean=(f"{other}_score", "mean"),
            other_scores=(f"{other}_score", list),
        )
        .dropna()
        .reset_index()
        for other in others
        for base_name in base_names
    ]

    all_results = pd.concat(results, ignore_index=True)

    def _get_p_value(base_scores: np.ndarray, other_scores: np.ndarray) -> pd.Series:
        shapiro_base = shapiro(base_scores)
        shapiro_other = shapiro(other_scores)
        if shapiro_base.pvalue > alpha and shapiro_other.pvalue > alpha:
            result = ttest_rel(base_scores, other_scores)
            p_value = result.pvalue
            statistic = result.statistic
            test = "paired t-test"
        else:
            result = wilcoxon(base_scores, other_scores, method="approx")
            p_value = result.pvalue  # type: ignore
            statistic = result.zstatistic  # type: ignore
            test = "wilcoxon"
        if np.isnan(p_value) and np.array_equal(base_scores, other_scores):
            # If the scores are identical, set p_value to 1.0
            p_value = 1.0
        return pd.Series(
            [
                base_scores.mean(),
                shapiro_base.pvalue,
                shapiro_base.pvalue > alpha,
                other_scores.mean(),
                shapiro_other.pvalue,
                shapiro_other.pvalue > alpha,
                p_value,
                test,
                statistic,
                p_value < alpha,
            ],
            index=[
                "base_mean",
                "shapiro_base",
                "base_normal",
                "other_mean",
                "shapiro_other",
                "other_normal",
                "p_value",
                "test",
                "statistic",
                "uncorrected_significance",
            ],
        )

    final_result = (
        all_results.groupby(["question_set", "criteria", "base_name", "other_name"])
        .apply(
            lambda x: _get_p_value(
                x["base_mean"].to_numpy(),
                x["other_mean"].to_numpy(),
            ),
            include_groups=False,
        )
        .reset_index()
    )

    final_result["formatted_p_value"] = final_result["p_value"].apply(
        lambda x: f"{x:.3f}" if x > 0.001 else "< 0.001"
    )

    corrected_significance = final_result.groupby(["question_set", "criteria"]).apply(
        lambda group: pd.Series(
            [
                multipletests(group["p_value"].to_numpy(), method="holm")[1],
                group["base_name"].tolist(),
                group["other_name"].tolist(),
            ],
            index=["corrected_p_values", "base_name", "other_name"],
        ),
        include_groups=False,
    )
    corrected_significance = corrected_significance.explode([
        "corrected_p_values",
        "base_name",
        "other_name",
    ]).reset_index()

    final_result = final_result.merge(corrected_significance)
    final_result = final_result.rename(
        columns={
            "corrected_p_values": "corrected_p_value",
        }
    )

    final_result["corrected_significance"] = final_result["corrected_p_value"] < alpha
    final_result["formatted_corrected_p_value"] = final_result[
        "corrected_p_value"
    ].apply(lambda x: f"{x:.3f}" if x > 0.001 else "< 0.001")

    return final_result
