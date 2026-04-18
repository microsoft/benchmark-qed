# Copyright (c) 2025 Microsoft Corporation.
"""Hierarchical assertion scoring functions.

This module provides functions for evaluating hierarchical assertions with
supporting (local) assertions using a language model.

Two evaluation modes are supported:
- JOINT: Global and supporting assertions evaluated together in one LLM call.
  Cheaper but may have anchoring bias where supporting assertions influence
  global evaluation.
- STAGED: Global assertions evaluated first (standard scoring), then supporting
  assertions evaluated only for passed globals. More expensive but ensures
  global pass rate matches standard scoring.
"""

import asyncio
import functools
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any
from uuid import uuid4

import pandas as pd
from rich import print as rich_print
from rich.progress import Progress, TaskID

from benchmark_qed.autoe.assertion.standard import get_assertion_scores
from benchmark_qed.autoe.data_model.assertion import (
    HierarchicalAssertionLLMResponse,
    SupportingDiscoveryLLMResponse,
)
from benchmark_qed.autoe.prompts import assertion as assertion_prompts
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel
from benchmark_qed.llm.utils import chat_typed_response

ASSERTION_PROMPTS = Path(assertion_prompts.__file__).parent


class HierarchicalMode(Enum):
    """Evaluation mode for hierarchical assertion scoring.

    Attributes
    ----------
        JOINT: Evaluate global and supporting assertions together in one LLM
            call. Cheaper but may have anchoring bias.
        STAGED: Evaluate global assertions first (standard scoring), then
            supporting assertions only for passed globals. Ensures global
            pass rate matches standard scoring.
    """

    JOINT = "joint"
    STAGED = "staged"


def _format_supporting_assertion(
    sa: str | dict[str, Any],
) -> str:
    """Extract statement text from a supporting assertion.

    Supporting assertions may be stored as plain strings or as
    dictionaries with a ``statement`` key.  This helper normalises
    both representations to a string suitable for prompt formatting.

    Args:
        sa: A supporting assertion, either as a string or a dict
            containing a ``statement`` key.

    Returns
    -------
        The statement text as a string.
    """
    if isinstance(sa, dict):
        return str(sa.get("statement", sa))
    return str(sa)


def _select_best_global_reasoning(row: pd.Series) -> str:
    """Pick the longest reasoning from passing trials.

    When multiple trials are run, only the trials that passed
    (score == 1) are considered.  Among those, the longest
    reasoning string is selected so that the step-2 prompt
    receives the most informative context.

    Args:
        row: A DataFrame row with ``reasoning_list`` (list of str)
            and ``trial_scores`` (list of int).

    Returns
    -------
        The selected reasoning string, or an empty string if no
        reasoning is available.
    """
    reasonings: list[str] = row["reasoning_list"]  # type: ignore[assignment]
    scores: list[int] = row["trial_scores"]  # type: ignore[assignment]
    passing = [r for r, s in zip(reasonings, scores, strict=True) if s == 1]
    if not passing:
        # Fallback: return the first reasoning if available
        return reasonings[0] if reasonings else ""
    return max(passing, key=len)


def get_hierarchical_assertion_scores(
    *,
    llm_client: ChatModel,
    llm_config: LLMConfig,
    answers: pd.DataFrame,
    assertions: pd.DataFrame,
    trials: int,
    mode: HierarchicalMode = HierarchicalMode.JOINT,
    pass_threshold: float = 0.5,
    top_k: int | None = None,
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    include_score_id_in_prompt: bool = True,
    question_id_key: str = "question_id",
    question_text_key: str = "question_text",
    answer_text_key: str = "answer",
    supporting_assertions_key: str = "supporting_assertions",
) -> pd.DataFrame:
    """Score hierarchical assertions with supporting assertions using an LLM.

    This function evaluates global assertions along with their supporting (local)
    assertions using the specified evaluation mode.

    Args:
        llm_client: The LLM client to use for scoring.
        llm_config: The LLM configuration to use for scoring.
        answers: DataFrame containing answers with columns 'question', 'answer'.
        assertions: DataFrame containing assertions with columns 'assertion' and
            'supporting_assertions' (list of strings).
        trials: Number of trials to run for each assertion.
        mode: Evaluation mode - JOINT (default) or STAGED. See HierarchicalMode
            for details.
        pass_threshold: Threshold for determining if an assertion passed.
            Default 0.5. Only used in STAGED mode.
        top_k: If specified, only evaluate the top-k assertions per question.
        assessment_system_prompt: Optional system prompt template for the
            assessment. Only used in JOINT mode.
        assessment_user_prompt: Optional user prompt template for the assessment.
            Only used in JOINT mode.
        include_score_id_in_prompt: Whether to include the score ID in the user
            prompt.
        question_id_key: Column name for question ID.
        question_text_key: Column name for question text.
        answer_text_key: Column name for answer text.
        supporting_assertions_key: Column name for supporting assertions list.

    Returns
    -------
        DataFrame with hierarchical assertion scores including:
            - global_score: Whether the global assertion passed (0/1)
            - global_passed: Boolean indicating if global assertion passed
            - supporting_passed: List of bools for each supporting assertion
            - supporting_scores: List of ints (0/1) for each supporting assertion
            - support_count: Number of supporting assertions that passed
            - support_total: Total number of supporting assertions
            - has_discovery: Whether discovery was detected
            - discovery_reasoning: Explanation of discovered information
            - reasoning: Overall reasoning from the LLM
            - trial: Trial number
    """
    if mode == HierarchicalMode.JOINT:
        return _get_hierarchical_scores_joint(
            llm_client=llm_client,
            llm_config=llm_config,
            answers=answers,
            assertions=assertions,
            trials=trials,
            top_k=top_k,
            assessment_system_prompt=assessment_system_prompt,
            assessment_user_prompt=assessment_user_prompt,
            include_score_id_in_prompt=include_score_id_in_prompt,
            question_id_key=question_id_key,
            question_text_key=question_text_key,
            answer_text_key=answer_text_key,
            supporting_assertions_key=supporting_assertions_key,
        )
    # STAGED
    return _get_hierarchical_scores_staged(
        llm_client=llm_client,
        llm_config=llm_config,
        answers=answers,
        assertions=assertions,
        trials=trials,
        pass_threshold=pass_threshold,
        top_k=top_k,
        include_score_id_in_prompt=include_score_id_in_prompt,
        question_id_key=question_id_key,
        question_text_key=question_text_key,
        answer_text_key=answer_text_key,
        supporting_assertions_key=supporting_assertions_key,
    )


def _get_hierarchical_scores_joint(
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
    supporting_assertions_key: str = "supporting_assertions",
) -> pd.DataFrame:
    """Score hierarchical assertions using joint evaluation (internal).

    Evaluates global and supporting assertions together in a single LLM call
    per assertion. Cheaper but may have anchoring bias.
    """
    # Merge answers with assertions
    pairs = (
        answers.merge(
            assertions,
            how="inner",
            on=[question_id_key],
            suffixes=("_base", "_other"),
        )
        .drop(columns=[f"{question_text_key}_other"], errors="ignore")
        .rename(
            columns={
                f"{question_id_key}": "question_id",
                f"{question_text_key}_base": "question_text",
                f"{answer_text_key}": "answer_text",
            }
        )
    )

    # Ensure required columns exist
    required_cols = [
        "question_id",
        "question_text",
        "answer_text",
        "assertion",
        supporting_assertions_key,
    ]
    for col in required_cols:
        if col not in pairs.columns:
            msg = f"Missing required column: {col}"
            raise ValueError(msg)

    pairs = pairs[required_cols]
    pairs = pairs.rename(columns={supporting_assertions_key: "supporting_assertions"})  # type: ignore[call-overload]

    # Apply top-k filtering if specified
    if top_k is not None and top_k > 0:
        if "rank" in assertions.columns:
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
            pairs = pairs.groupby("question_id").head(top_k).reset_index(drop=True)

    with Progress() as progress:

        def on_complete_callback(progress_task: TaskID) -> None:
            progress.update(progress_task, advance=1, refresh=True)

        progress_task = progress.add_task(
            "Scoring hierarchical...", total=len(pairs) * trials
        )

        tasks = [
            evaluate_hierarchical_assertion(
                llm_client=llm_client,
                assertion=row["assertion"],  # type: ignore[arg-type]
                question=row["question_text"],  # type: ignore[arg-type]
                answer=row["answer_text"],  # type: ignore[arg-type]
                supporting_assertions=row["supporting_assertions"],  # type: ignore[arg-type]
                assessment_system_prompt=assessment_system_prompt,
                assessment_user_prompt=assessment_user_prompt,
                complete_callback=functools.partial(
                    on_complete_callback, progress_task
                ),
                trial=n,
                include_score_id_in_prompt=include_score_id_in_prompt,
                additional_call_args=llm_config.call_args,
            )
            for _, row in pairs.iterrows()
            for n in range(trials)
        ]

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))

        # Post-process results to add support metrics
        for result in results:
            result["support_count"] = sum(result["supporting_scores"])
            result["support_total"] = len(result["supporting_scores"])

        return pd.DataFrame(results)


def _get_hierarchical_scores_staged(
    *,
    llm_client: ChatModel,
    llm_config: LLMConfig,
    answers: pd.DataFrame,
    assertions: pd.DataFrame,
    trials: int,
    pass_threshold: float = 0.5,
    top_k: int | None = None,
    include_score_id_in_prompt: bool = True,
    question_id_key: str = "question_id",
    question_text_key: str = "question_text",
    answer_text_key: str = "answer",
    supporting_assertions_key: str = "supporting_assertions",
) -> pd.DataFrame:
    """Score hierarchical assertions using staged evaluation (internal).

    Evaluates global assertions first (standard scoring), then supporting
    assertions only for passed globals. Ensures global pass rate matches
    standard scoring but is more expensive.
    """
    rich_print("[bold]Step 1: Standard global assertion scoring...[/bold]")

    # Determine the actual question text column in assertions
    # It may be different from question_text_key (which is for answers)
    if question_text_key in assertions.columns:
        assertion_question_col = question_text_key
    elif "question_text" in assertions.columns:
        assertion_question_col = "question_text"
    else:
        msg = (
            f"Assertions must have either '{question_text_key}' or 'question_text' "
            f"column. Found: {assertions.columns.tolist()}"
        )
        raise ValueError(msg)

    # Prepare global assertions for standard scoring (without supporting
    # assertions)
    global_assertions_df = assertions[
        [question_id_key, assertion_question_col, "assertion"]
    ].copy()
    # Rename to standard name for get_assertion_scores
    if assertion_question_col != "question_text":
        global_assertions_df = global_assertions_df.rename(  # type: ignore[call-overload]
            columns={assertion_question_col: "question_text"}
        )

    # Run standard assertion scoring
    global_scores_raw = get_assertion_scores(
        llm_client=llm_client,
        llm_config=llm_config,
        answers=answers,
        assertions=global_assertions_df,  # type: ignore[arg-type]
        trials=trials,
        top_k=top_k,
        include_score_id_in_prompt=include_score_id_in_prompt,
        question_id_key=question_id_key,
        question_text_key=question_text_key,
        answer_text_key=answer_text_key,
    )

    # Note: get_assertion_scores returns 'question' as question_text, not
    # question_id. We need to map back to question_id for merging with assertions
    question_text_to_id = assertions.set_index(assertion_question_col)[
        question_id_key
    ].to_dict()
    global_scores_raw["question_id"] = global_scores_raw["question"].map(
        question_text_to_id  # type: ignore[arg-type]
    )

    # Aggregate global scores to determine which assertions passed
    global_aggregated = (
        global_scores_raw.groupby(["question_id", "question", "assertion"])
        .agg(
            global_score_mean=("score", "mean"),
            global_score=("score", lambda x: int(x.mean() > pass_threshold)),
            reasoning_list=("reasoning", list),
            trial_scores=("score", list),
        )
        .reset_index()
    )
    global_aggregated["global_passed"] = global_aggregated["global_score"] == 1

    passed_count = global_aggregated["global_passed"].sum()
    total_count = len(global_aggregated)
    per_q_pass_rate = (
        global_aggregated.groupby("question")["global_score"].mean().mean()
    )
    rich_print(
        f"  Global assertions: {passed_count}/{total_count} passed "
        f"({passed_count / total_count * 100:.1f}%)"
    )
    rich_print(f"  Global pass rate (per-question avg): {per_q_pass_rate * 100:.1f}%")

    # Get the passed assertions for step 2
    passed_assertions = global_aggregated[global_aggregated["global_passed"]][
        [
            "question_id",
            "question",
            "assertion",
            "reasoning_list",
            "trial_scores",
        ]
    ].copy()

    # Select the best global reasoning for each passed assertion:
    # pick the longest reasoning from passing trials only, so the
    # step-2 prompt gets the most informative context.
    passed_assertions["global_reasoning"] = passed_assertions.apply(  # type: ignore[union-attr]
        _select_best_global_reasoning, axis=1
    )

    if len(passed_assertions) == 0:
        rich_print("[yellow]No assertions passed. Skipping step 2.[/yellow]")
        # Return results with empty supporting info
        results = []
        for _, row in global_aggregated.iterrows():
            n_supporting = len(
                assertions[assertions["assertion"] == row["assertion"]][
                    supporting_assertions_key
                ].iloc[0]  # type: ignore[union-attr]
            )
            for trial_idx, (_score, reasoning) in enumerate(
                zip(row["trial_scores"], row["reasoning_list"], strict=True)
            ):
                results.append({
                    "question": row["question"],
                    "assertion": row["assertion"],
                    "global_passed": False,
                    "global_score": 0,
                    "reasoning": reasoning,
                    "supporting_passed": [False] * n_supporting,
                    "supporting_scores": [0] * n_supporting,
                    "supporting_results": [],
                    "support_count": 0,
                    "support_total": n_supporting,
                    "has_discovery": False,
                    "discovery_reasoning": "",
                    "trial": trial_idx,
                })
        return pd.DataFrame(results)

    rich_print(
        f"\n[bold]Step 2: Supporting + discovery for {len(passed_assertions)} "
        f"passed assertions...[/bold]"
    )

    # Merge passed assertions with their supporting assertions using question_id
    assertions_for_merge = assertions[
        [question_id_key, "assertion", supporting_assertions_key]
    ].copy()
    assertions_for_merge = assertions_for_merge.rename(  # type: ignore[call-overload]
        columns={
            question_id_key: "question_id",
            supporting_assertions_key: "supporting_assertions",
        }
    )
    passed_with_supporting = passed_assertions.merge(  # type: ignore[union-attr]
        assertions_for_merge,
        on=["question_id", "assertion"],
        how="left",
    )

    # Drop rows where supporting_assertions is NaN (shouldn't happen but be safe)
    before_drop = len(passed_with_supporting)
    passed_with_supporting = passed_with_supporting.dropna(
        subset=["supporting_assertions"]
    )
    if len(passed_with_supporting) < before_drop:
        rich_print(
            f"[yellow]Warning: Dropped {before_drop - len(passed_with_supporting)} "
            f"rows with missing supporting assertions[/yellow]"
        )

    # Also need the answer text - lookup by question_id
    answer_lookup = answers.set_index(question_id_key)[answer_text_key].to_dict()
    passed_with_supporting["answer_text"] = passed_with_supporting["question_id"].map(
        answer_lookup  # type: ignore[arg-type]
    )

    # Run supporting + discovery evaluation for passed assertions
    # across all trials for proper variance estimation
    total_evals = len(passed_with_supporting) * trials
    with Progress() as progress:

        def on_complete_callback(progress_task: TaskID) -> None:
            progress.update(progress_task, advance=1, refresh=True)

        progress_task = progress.add_task("Evaluating supporting...", total=total_evals)

        tasks = [
            evaluate_supporting_discovery(
                llm_client=llm_client,
                assertion=row["assertion"],  # type: ignore[arg-type]
                question=row["question"],  # type: ignore[arg-type]
                answer=row["answer_text"],  # type: ignore[arg-type]
                supporting_assertions=row["supporting_assertions"],  # type: ignore[arg-type]
                question_id=row["question_id"],  # type: ignore[arg-type]
                trial=trial_idx,
                global_reasoning=row["global_reasoning"],  # type: ignore[arg-type]
                complete_callback=functools.partial(
                    on_complete_callback, progress_task
                ),
                include_score_id_in_prompt=include_score_id_in_prompt,
                additional_call_args=llm_config.call_args,
            )
            for trial_idx in range(trials)
            for _, row in passed_with_supporting.iterrows()
        ]

        loop = asyncio.get_event_loop()
        supporting_results = loop.run_until_complete(asyncio.gather(*tasks))

    # Build lookup for supporting results keyed by
    # (question_id, assertion, trial) for trial-indexed retrieval
    supporting_lookup: dict[tuple[str, str, int], dict] = {
        (r["question_id"], r["assertion"], r["trial"]): r for r in supporting_results
    }

    # Combine all results
    results = []
    for _, row in global_aggregated.iterrows():
        question = row["question"]
        question_id = row["question_id"]
        assertion_text = row["assertion"]
        global_passed = row["global_passed"]

        # Get supporting assertions count
        matching_assertion = assertions[assertions["assertion"] == assertion_text]
        if len(matching_assertion) > 0:
            n_supporting = len(matching_assertion[supporting_assertions_key].iloc[0])  # type: ignore[union-attr]
            supporting_assertions_list = matching_assertion[
                supporting_assertions_key
            ].iloc[0]  # type: ignore[union-attr]
        else:
            n_supporting = 0
            supporting_assertions_list = []

        for trial_idx, (_score, reasoning) in enumerate(
            zip(row["trial_scores"], row["reasoning_list"], strict=True)
        ):
            # Use (question_id, assertion, trial) to look up
            # the trial-specific supporting result
            supp_key = (question_id, assertion_text, trial_idx)
            if global_passed and supp_key in supporting_lookup:  # type: ignore[operator]
                supp_result = supporting_lookup[supp_key]  # type: ignore[arg-type]
                results.append({
                    "question": question,
                    "assertion": assertion_text,
                    "global_passed": True,
                    "global_score": 1,
                    "reasoning": reasoning,
                    "supporting_passed": supp_result["supporting_passed"],
                    "supporting_scores": supp_result["supporting_scores"],
                    "supporting_results": supp_result["supporting_results"],
                    "support_count": sum(supp_result["supporting_scores"]),
                    "support_total": n_supporting,
                    "has_discovery": supp_result["has_discovery"],
                    "discovery_reasoning": supp_result["discovery_reasoning"],
                    "supporting_assertions": supporting_assertions_list,
                    "trial": trial_idx,
                })
            else:
                # Failed assertion - no supporting data
                results.append({
                    "question": question,
                    "assertion": assertion_text,
                    "global_passed": False,
                    "global_score": 0,
                    "reasoning": reasoning,
                    "supporting_passed": [False] * n_supporting,
                    "supporting_scores": [0] * n_supporting,
                    "supporting_results": [
                        {
                            "id": f"SA{i + 1}",
                            "passed": False,
                            "reasoning": "Global assertion failed",
                        }
                        for i in range(n_supporting)
                    ],
                    "support_count": 0,
                    "support_total": n_supporting,
                    "has_discovery": False,
                    "discovery_reasoning": "",
                    "supporting_assertions": supporting_assertions_list,
                    "trial": trial_idx,
                })

    return pd.DataFrame(results)


async def evaluate_hierarchical_assertion(
    llm_client: ChatModel,
    assertion: str,
    question: str,
    answer: str,
    supporting_assertions: list[str | dict[str, Any]],
    trial: int = 0,
    *,
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    include_score_id_in_prompt: bool = True,
    additional_call_args: dict[str, Any] | None = None,
    complete_callback: Callable | None = None,
) -> dict[str, Any]:
    """Evaluate a hierarchical assertion with its supporting assertions.

    This function evaluates a global assertion along with its supporting (local)
    assertions in a single LLM call. It also detects if the answer contains
    information beyond what is covered by the supporting assertions (discovery).

    Args:
        llm_client: The LLM client to use for evaluation.
        assertion: The global assertion to evaluate.
        question: The question being answered.
        answer: The answer to evaluate.
        supporting_assertions: List of supporting/local assertions that underpin
            the global assertion. Each item may be a string or a dict with a
            ``statement`` key.
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
            - reasoning: Overall reasoning from the LLM
            - global_passed: Whether the global assertion passed (bool)
            - global_score: Integer score (1 if passed, 0 otherwise)
            - supporting_passed: List of bools for each supporting assertion
            - supporting_scores: List of ints (1/0) for each supporting assertion
            - has_discovery: Whether discovery was detected
            - discovery_reasoning: Explanation of any discovered information
            - question, answer, assertion, trial: Input data echoed back
            - supporting_assertions: The supporting assertions evaluated
    """
    assessment_system_prompt = assessment_system_prompt or load_template_file(
        ASSERTION_PROMPTS / "hierarchical_assertion_system_prompt.txt"
    )

    assessment_user_prompt = assessment_user_prompt or load_template_file(
        ASSERTION_PROMPTS / "hierarchical_assertion_user_prompt.txt"
    )
    score_id = uuid4().hex

    # Format supporting assertions with IDs (SA1, SA2, etc.) for the prompt
    supporting_assertions_formatted = "\n".join(
        f"SA{i + 1}: {_format_supporting_assertion(sa)}"
        for i, sa in enumerate(supporting_assertions)
    )
    # Create ID mapping for validation
    expected_ids = [f"SA{i + 1}" for i in range(len(supporting_assertions))]

    messages = [
        {
            "role": "system",
            "content": assessment_system_prompt.substitute(),
        },
        {
            "role": "user",
            "content": assessment_user_prompt.substitute(
                score_id=score_id if include_score_id_in_prompt else "",
                assertion=assertion,
                question=question,
                answer=answer,
                supporting_assertions=supporting_assertions_formatted,
            ),
        },
    ]

    response = await chat_typed_response(
        llm=llm_client,
        messages=messages,
        data_model=HierarchicalAssertionLLMResponse,
        **(additional_call_args or {}),
    )

    if complete_callback:
        complete_callback()

    # Extract supporting results and validate
    # Build a dict from response results by ID for easy lookup
    results_by_id = {r.id: r for r in response.supporting_results}

    # Construct ordered lists matching input order
    supporting_passed = []
    supporting_scores = []
    supporting_reasoning = []
    for expected_id in expected_ids:
        if expected_id in results_by_id:
            result = results_by_id[expected_id]
            supporting_passed.append(result.passed)
            supporting_scores.append(1 if result.passed else 0)
            supporting_reasoning.append(result.reasoning)
        else:
            # Missing result - default to False
            supporting_passed.append(False)
            supporting_scores.append(0)
            supporting_reasoning.append("No evaluation provided")

    return {
        "score_id": score_id,
        "reasoning": response.reasoning,
        "global_passed": response.global_passed,
        "global_score": 1 if response.global_passed else 0,
        "supporting_passed": supporting_passed,
        "supporting_scores": supporting_scores,
        "supporting_reasoning": supporting_reasoning,
        "supporting_results": [
            {"id": eid, "passed": sp, "reasoning": sr}
            for eid, sp, sr in zip(
                expected_ids, supporting_passed, supporting_reasoning, strict=True
            )
        ],
        "has_discovery": response.has_discovery,
        "discovery_reasoning": response.discovery_reasoning,
        "question": question,
        "answer": answer,
        "assertion": assertion,
        "supporting_assertions": supporting_assertions,
        "trial": trial,
    }


async def evaluate_supporting_discovery(
    llm_client: ChatModel,
    assertion: str,
    question: str,
    answer: str,
    supporting_assertions: list[str | dict[str, Any]],
    question_id: str | None = None,
    trial: int = 0,
    global_reasoning: str = "",
    *,
    assessment_system_prompt: Template | None = None,
    assessment_user_prompt: Template | None = None,
    include_score_id_in_prompt: bool = True,
    additional_call_args: dict[str, Any] | None = None,
    complete_callback: Callable | None = None,
) -> dict[str, Any]:
    """Evaluate supporting assertions and discovery for a passed assertion.

    This function is used in the two-call approach where the global assertion has
    already been evaluated separately (and passed). This call only evaluates
    supporting assertions and discovery detection.

    Args:
        llm_client: The LLM client to use for evaluation.
        assertion: The global assertion (already determined to be satisfied).
        question: The question being answered.
        answer: The answer to evaluate.
        supporting_assertions: List of supporting/local assertions. Each
            item may be a string or a dict with a ``statement`` key.
        question_id: Optional question ID for tracking in results.
        trial: Trial number for this evaluation (for repeated trials).
        global_reasoning: Reasoning from the step-1 global evaluation that
            explains why the main assertion was determined to be satisfied.
            Included in the prompt so the LLM has context for evaluating
            supporting assertions and discovery.
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
            - supporting_passed: List of bools for each supporting assertion
            - supporting_scores: List of ints (1/0) for each supporting assertion
            - supporting_reasoning: List of reasoning strings
            - supporting_results: List of dicts with id, passed, reasoning
            - has_discovery: Whether discovery was detected
            - discovery_reasoning: Explanation of any discovered information
            - question, question_id, answer, assertion, trial: Input data echoed
                back
            - supporting_assertions: The supporting assertions evaluated
    """
    assessment_system_prompt = assessment_system_prompt or load_template_file(
        ASSERTION_PROMPTS / "supporting_discovery_system_prompt.txt"
    )

    assessment_user_prompt = assessment_user_prompt or load_template_file(
        ASSERTION_PROMPTS / "supporting_discovery_user_prompt.txt"
    )
    score_id = uuid4().hex

    # Format supporting assertions with IDs (SA1, SA2, etc.) for the prompt
    supporting_assertions_formatted = "\n".join(
        f"SA{i + 1}: {_format_supporting_assertion(sa)}"
        for i, sa in enumerate(supporting_assertions)
    )
    # Create ID mapping for validation
    expected_ids = [f"SA{i + 1}" for i in range(len(supporting_assertions))]

    messages = [
        {
            "role": "system",
            "content": assessment_system_prompt.substitute(),
        },
        {
            "role": "user",
            "content": assessment_user_prompt.substitute(
                score_id=score_id if include_score_id_in_prompt else "",
                assertion=assertion,
                question=question,
                answer=answer,
                supporting_assertions=supporting_assertions_formatted,
                global_reasoning=global_reasoning,
            ),
        },
    ]

    response = await chat_typed_response(
        llm=llm_client,
        messages=messages,
        data_model=SupportingDiscoveryLLMResponse,
        **(additional_call_args or {}),
    )

    if complete_callback:
        complete_callback()

    # Extract supporting results and validate
    results_by_id = {r.id: r for r in response.supporting_results}

    # Construct ordered lists matching input order
    supporting_passed = []
    supporting_scores = []
    supporting_reasoning = []
    for expected_id in expected_ids:
        if expected_id in results_by_id:
            result = results_by_id[expected_id]
            supporting_passed.append(result.passed)
            supporting_scores.append(1 if result.passed else 0)
            supporting_reasoning.append(result.reasoning)
        else:
            # Missing result - default to False
            supporting_passed.append(False)
            supporting_scores.append(0)
            supporting_reasoning.append("No evaluation provided")

    return {
        "score_id": score_id,
        "supporting_passed": supporting_passed,
        "supporting_scores": supporting_scores,
        "supporting_reasoning": supporting_reasoning,
        "supporting_results": [
            {"id": eid, "passed": sp, "reasoning": sr}
            for eid, sp, sr in zip(
                expected_ids, supporting_passed, supporting_reasoning, strict=True
            )
        ],
        "has_discovery": response.has_discovery,
        "discovery_reasoning": response.discovery_reasoning,
        "question": question,
        "question_id": question_id,
        "answer": answer,
        "assertion": assertion,
        "supporting_assertions": supporting_assertions,
        "trial": trial,
    }
