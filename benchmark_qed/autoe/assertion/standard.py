# Copyright (c) 2025 Microsoft Corporation.
"""Standard assertion scoring functions.

This module provides functions for evaluating individual assertions against
answers using a language model.
"""

import asyncio
import functools
import itertools
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import Any
from uuid import uuid4

import pandas as pd
from rich.progress import Progress, TaskID

from benchmark_qed.autoe.data_model.assertion import (
    Assertion,
    AssertionLLMResponse,
)
from benchmark_qed.autoe.prompts import assertion as assertion_prompts
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
    """Score assertions based on the provided answers using a language model.

    Args:
        llm_client: The LLM client to use for scoring.
        llm_config: The LLM configuration to use for scoring.
        answers: DataFrame containing answers with columns 'question', 'answer'.
        assertions: DataFrame containing assertions with column 'assertion'.
        trials: Number of trials to run for each assertion.
        top_k: If specified, only evaluate the top-k assertions per question
            (ranked by rank if available, where lower rank = higher importance,
            otherwise uses first k assertions).
        assessment_system_prompt: Optional system prompt template for the
            assessment.
        assessment_user_prompt: Optional user prompt template for the assessment.
        include_score_id_in_prompt: Whether to include the score ID in the user
            prompt.
        question_id_key: Column name for question ID.
        question_text_key: Column name for question text.
        answer_text_key: Column name for answer text.

    Returns
    -------
        DataFrame: Results with assertion scores and metadata.
    """
    pairs = answers.merge(
        assertions,
        how="inner",
        on=[question_id_key],
        suffixes=("_base", "_other"),
    )

    # Handle column renaming - the question text column may have different names
    # in answers vs assertions, or may only exist in one of them
    question_col_other = f"{question_text_key}_other"
    question_col_base = f"{question_text_key}_base"

    # Drop the _other question column if it exists
    if question_col_other in pairs.columns:
        pairs = pairs.drop(columns=[question_col_other])

    # Rename columns to standard names
    rename_map = {question_id_key: "question_id", answer_text_key: "answer_text"}
    if question_col_base in pairs.columns:
        rename_map[question_col_base] = "question_text"
    elif question_text_key in pairs.columns:
        # If we're renaming a column to question_text but question_text already
        # exists (from assertions), drop the existing one first to avoid duplicates
        if question_text_key != "question_text" and "question_text" in pairs.columns:
            pairs = pairs.drop(columns=["question_text"])
        rename_map[question_text_key] = "question_text"
    # Also check if assertions have question_text column that wasn't suffixed
    elif "question_text" not in pairs.columns and "question_text_base" in pairs.columns:
        rename_map["question_text_base"] = "question_text"

    pairs = pairs.rename(columns=rename_map)
    pairs = pairs[["question_id", "question_text", "answer_text", "assertion"]]

    # Apply top-k filtering if specified
    if top_k is not None and top_k > 0:
        # Check if assertions have a 'rank' column for ranking
        if "rank" in assertions.columns:
            # Rank by rank (ascending - lower rank = higher importance) and take
            # top-k per question
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

        async def _run_tasks() -> list[dict[str, Any]]:
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run_tasks())

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
    """Evaluate an assertion based on the provided criteria and conditions.

    Args:
        llm_client: The LLM client to use for evaluation.
        assertion: The assertion text to evaluate.
        question: The question being answered.
        answer: The answer to evaluate against the assertion.
        trial: Trial number for this evaluation (for repeated trials).
        assessment_system_prompt: Optional custom system prompt template.
        assessment_user_prompt: Optional custom user prompt template.
        include_score_id_in_prompt: Whether to include a unique score ID in
            prompt.
        additional_call_args: Additional arguments to pass to the LLM call.
        complete_callback: Callback function to invoke when evaluation completes.

    Returns
    -------
        Dictionary containing:
            - score_id: Unique identifier for this evaluation
            - reasoning: Explanation from the LLM
            - score: Binary score (0 or 1)
            - question, answer, assertion, trial: Input data echoed back
    """
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
