# Copyright (c) 2025 Microsoft Corporation.
"""Reference scoring functions for evaluation tasks.

This module provides functions for scoring generated answers against reference
(ground truth) answers using LLM-based evaluation with configurable criteria.
"""

import asyncio
import contextlib
import functools
import itertools
import uuid
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import pandas as pd
from graphrag_cache import CacheConfig, CacheType
from graphrag_llm.completion import LLMCompletion
from rich.progress import Progress, TaskID

from benchmark_qed.autoe.config import Criteria
from benchmark_qed.autoe.data_model import ConditionPair, ReferenceLLMResponse
from benchmark_qed.autoe.prompts import reference as reference_prompts
from benchmark_qed.autoe.reference.cache import (
    ReferenceScoreCache,
    build_cache_metadata,
    compute_cache_key,
    compute_logical_key,
)
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm import chat

REFERENCE_PROMPTS_PATH = Path(reference_prompts.__file__).parent


def get_reference_scores(
    *,
    llm_client: LLMCompletion,
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
    cache_config: CacheConfig | None = None,
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
        cache_config: GraphRAG cache backend configuration. Caching is disabled
            when omitted.

    Returns
    -------
        DataFrame containing the scores for each condition.
    """
    pairs = (
        reference_answers
        .merge(
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
    assessment_system_prompt = assessment_system_prompt or load_template_file(
        REFERENCE_PROMPTS_PATH / "reference_system_prompt.txt"
    )
    assessment_user_prompt = assessment_user_prompt or load_template_file(
        REFERENCE_PROMPTS_PATH / "reference_user_prompt.txt"
    )
    cache = ReferenceScoreCache(
        cache_config or CacheConfig(type=CacheType.Noop, storage=None)
    )
    cache_metadata = build_cache_metadata(
        model=llm_config.model,
        llm_provider=str(llm_config.llm_provider),
        init_args=llm_config.init_args,
        call_args=llm_config.call_args,
        custom_providers=[
            provider.model_dump(mode="json") for provider in llm_config.custom_providers
        ],
        system_prompt=assessment_system_prompt.template,
        user_prompt=assessment_user_prompt.template,
        include_score_id_in_prompt=include_score_id_in_prompt,
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
            _get_reference_score_with_cache(
                llm_client,
                cache=cache,
                cache_metadata=cache_metadata,
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

        async def _run_tasks() -> list[dict[str, Any]]:
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run_tasks())

        return pd.DataFrame(results)


async def _get_reference_score_with_cache(
    llm: LLMCompletion,
    *,
    cache: ReferenceScoreCache,
    cache_metadata: dict[str, Any],
    question: str,
    reference_answer: str,
    generated_answer: str,
    criteria_name: str,
    criteria_description: str,
    assessment_system_prompt: Template,
    assessment_user_prompt: Template,
    complete_callback: Callable | None,
    trial: int,
    score_min: int,
    score_max: int,
    include_score_id_in_prompt: bool,
    additional_call_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reuse or compute one reference score with cross-process coalescing."""
    logical_key = compute_logical_key(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
        criteria_name=criteria_name,
        criteria_description=criteria_description,
        trial=trial,
        score_min=score_min,
        score_max=score_max,
    )
    cache_key = compute_cache_key(logical_key, cache_metadata)
    owner_id = uuid.uuid4().hex

    while True:
        cached = await cache.get(cache_key)
        if cached is not None:
            if complete_callback:
                complete_callback()
            return cached
        if await cache.claim(cache_key, owner_id):
            cached = await cache.get(cache_key)
            if cached is not None:
                cache.release(cache_key, owner_id)
                if complete_callback:
                    complete_callback()
                return cached
            break
        await asyncio.sleep(0.1)

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(max(0.1, cache.lease_ttl_seconds / 3))
            cache.renew(cache_key, owner_id)

    heartbeat = asyncio.create_task(_heartbeat())
    try:
        score = await get_reference_score(
            llm,
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            criteria_name=criteria_name,
            criteria_description=criteria_description,
            assessment_system_prompt=assessment_system_prompt,
            assessment_user_prompt=assessment_user_prompt,
            trial=trial,
            score_min=score_min,
            score_max=score_max,
            include_score_id_in_prompt=include_score_id_in_prompt,
            additional_call_args=additional_call_args,
        )
        publication = await cache.publish(
            cache_key,
            score,
            cache_metadata,
            owner_id=owner_id,
            logical_key=logical_key,
        )
        if publication.accepted and isinstance(publication.value, dict):
            score = publication.value
    except (Exception, asyncio.CancelledError):
        cache.release(cache_key, owner_id)
        raise
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat

    if complete_callback:
        complete_callback()
    return score


async def get_reference_score(
    llm: LLMCompletion,
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
    assessment_response = (
        await chat(
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
            response_format=ReferenceLLMResponse,
            **(additional_call_args or {}),
        )
    ).formatted_response
    if assessment_response is None:
        msg = "LLM did not return a structured ReferenceLLMResponse."
        raise RuntimeError(msg)

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
        raw_scores
        .drop(
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
