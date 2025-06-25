# Copyright (c) 2025 Microsoft Corporation.
"""Evaluate assertions using a language model."""

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

from benchmark_qed.autoe.data_model.assertion import Assertion, AssertionLLMResponse
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
        assessment_system_prompt (Template | None): Optional system prompt template for the assessment.
        assessment_user_prompt (Template | None): Optional user prompt template for the assessment.
        include_score_id_in_prompt (bool): Whether to include the score ID in the user prompt.
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
