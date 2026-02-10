# Copyright (c) 2025 Microsoft Corporation.
"""Orchestration and pipeline functions for assertion evaluation.

This module provides functions for loading data, running evaluations for
multiple RAG methods, and generating summary reports.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from rich import print as rich_print

from benchmark_qed.autoe.assertion.aggregation import aggregate_hierarchical_scores
from benchmark_qed.autoe.assertion.hierarchical import (
    HierarchicalMode,
    get_hierarchical_assertion_scores,
)
from benchmark_qed.autoe.assertion.significance import (
    compare_assertion_scores_significance,
    compare_hierarchical_assertion_scores_significance,
    summarize_significance_results,
)
from benchmark_qed.autoe.assertion.standard import get_assertion_scores
from benchmark_qed.cli.utils import print_df
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.llm.type.base import ChatModel


def load_and_normalize_assertions(
    input_dir: str,
    question_set: str,
    assertions_filename_template: str = "{question_set}_assertions.json",
) -> pd.DataFrame:
    """Load assertions from JSON file and normalize nested dictionaries.

    Args:
        input_dir: Directory containing assertion files.
        question_set: Name of the question set.
        assertions_filename_template: Template for assertion filename
            (default: "{question_set}_assertions.json").

    Returns:
        DataFrame with normalized assertion data containing question_id,
        question_text, assertion, rank.
    """
    assertions_file = assertions_filename_template.format(question_set=question_set)
    assertions_raw = pd.read_json(f"{input_dir}/{assertions_file}")

    # Explode assertions and normalize the nested dictionaries
    assertions = assertions_raw.explode("assertions").reset_index(drop=True)

    # Normalize the assertion dictionaries into separate columns
    assertion_normalized = pd.json_normalize(
        cast(list[dict[str, Any]], assertions["assertions"].tolist())
    )
    assertions = pd.concat(
        [
            assertions.drop("assertions", axis=1),
            assertion_normalized[["statement", "rank"]],  # Keep only statement and rank
        ],
        axis=1,
    )

    # Rename the statement column to assertion for compatibility
    return assertions.rename(columns={"statement": "assertion"})


def load_and_normalize_hierarchical_assertions(
    assertions_path: str | Path,
    *,
    assertions_key: str = "assertions",
    supporting_assertions_key: str = "supporting_assertions",
) -> pd.DataFrame:
    """Load hierarchical assertions from JSON and normalize for evaluation.

    Loads assertions with supporting sub-assertions, explodes the nested
    structure, normalizes dict columns, renames 'statement' to 'assertion'
    for consistency, and filters to only include assertions that have
    non-empty supporting assertions.

    Parameters
    ----------
    assertions_path : str | Path
        Path to the JSON file containing assertions.
    assertions_key : str
        Column name containing the assertions list
        (default: "assertions").
    supporting_assertions_key : str
        Column name containing supporting assertions within each
        assertion (default: "supporting_assertions").

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized hierarchical assertions, each row
        containing a single assertion with its supporting assertions.

    Raises
    ------
    ValueError
        If the assertions file is missing required columns or contains
        no valid hierarchical assertions.
    """
    assertions_raw = pd.read_json(assertions_path, encoding="utf-8")

    # Validate the assertions key exists
    if assertions_key not in assertions_raw.columns:
        msg = f"Assertions file missing required '{assertions_key}' column."
        raise ValueError(msg)

    # Filter out rows with missing or empty assertions
    if assertions_raw[assertions_key].isna().any():
        rich_print(
            "[bold yellow]Some questions do not have assertions. "
            "These will be skipped.[/bold yellow]"
        )
        assertions_raw = assertions_raw[
            ~assertions_raw[assertions_key].isna()
        ]

    # Explode the assertions list into individual rows
    assertions = assertions_raw.explode(assertions_key).reset_index(
        drop=True
    )

    # Normalize nested assertion dicts into separate columns
    if assertions[assertions_key].apply(
        lambda x: isinstance(x, dict)
    ).any():
        assertion_details = pd.json_normalize(
            assertions[assertions_key].tolist()
        )
        assertions = pd.concat(
            [
                assertions.drop(columns=[assertions_key]),
                assertion_details,
            ],
            axis=1,
        )
        # Rename 'statement' to 'assertion' for consistency
        if "statement" in assertions.columns:
            assertions = assertions.rename(
                columns={"statement": "assertion"}
            )
    else:
        assertions = assertions.rename(
            columns={assertions_key: "assertion"}
        )

    # Validate supporting assertions column exists
    if supporting_assertions_key not in assertions.columns:
        msg = (
            f"Assertions missing '{supporting_assertions_key}' column. "
            "Use load_and_normalize_assertions for standard assertions."
        )
        raise ValueError(msg)

    # Filter to only include assertions with non-empty supporting assertions
    has_supporting = assertions[supporting_assertions_key].apply(
        lambda x: isinstance(x, list) and len(x) > 0
    )
    if not has_supporting.all():
        n_missing = (~has_supporting).sum()
        rich_print(
            f"[bold yellow]{n_missing} assertions without supporting "
            f"assertions will be skipped.[/bold yellow]"
        )
    assertions = assertions[has_supporting].reset_index(drop=True)

    if len(assertions) == 0:
        msg = "No valid hierarchical assertions found after filtering."
        raise ValueError(msg)

    return assertions


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
    question_text_key: str = "question_text",
    answer_text_key: str = "answer",
) -> dict[str, Any] | None:
    """Evaluate a single RAG method against assertions for a question set.

    Args:
        llm_client: LLM client for evaluation.
        llm_config: LLM configuration.
        generated_rag: Name of the RAG method.
        question_set: Name of the question set.
        assertions: DataFrame with assertions.
        input_dir: Input directory path.
        output_dir: Output directory path.
        trials: Number of evaluation trials.
        top_k_assertions: Number of top assertions to evaluate (None for all).
        pass_threshold: Threshold for assertion pass/fail.
        answers_path_template: Template for answers file path
            (default: "{input_dir}/{generated_rag}/{question_set}.json").
        question_text_key: Column name for question text (default: "question_text").
        answer_text_key: Column name for answer text (default: "answer").

    Returns:
        Dictionary with evaluation results or None if evaluation failed.
    """
    question_set_output_dir = output_dir / question_set
    if not question_set_output_dir.exists():
        question_set_output_dir.mkdir(parents=True)

    # Define answers path before try block so it's available in except block
    answers_path = answers_path_template.format(
        input_dir=input_dir, generated_rag=generated_rag, question_set=question_set
    )

    try:
        # Load answers for this RAG method and question set
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
            question_text_key=question_text_key,
            answer_text_key=answer_text_key,
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
                f"    [bold red]{generated_rag} ({question_set}): "
                f"{len(failed_assertions)} assertions failed[/bold red]"
            )
        else:
            rich_print(
                f"    [bold green]{generated_rag} ({question_set}): "
                f"All assertions passed[/bold green]"
            )

        rich_print(
            f"    {generated_rag} ({question_set}) - Overall accuracy: "
            f"{overall_accuracy:.3f} ({total_success}/{total_assertions})"
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
            f"    [bold yellow]Warning: Could not find answers file at "
            f"{answers_path}: {e}[/bold yellow]"
        )
        return None
    except (OSError, ValueError, KeyError) as e:
        rich_print(
            f"    [bold red]Error processing {generated_rag}/{question_set}: "
            f"{e}[/bold red]"
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
    run_significance_test: bool = True,
    significance_alpha: float = 0.05,
    significance_correction: str = "holm",
    question_text_key: str = "question_text",
    answer_text_key: str = "answer",
) -> pd.DataFrame:
    """Run assertion-based evaluation for multiple question sets and RAG methods.

    Args:
        llm_client: LLM client for evaluation.
        llm_config: LLM configuration.
        question_sets: List of question set names.
        generated_rags: List of RAG method names.
        input_dir: Input directory path.
        output_dir: Output directory path.
        trials: Number of evaluation trials.
        top_k_assertions: Number of top assertions to evaluate (None for all).
        pass_threshold: Threshold for assertion pass/fail.
        assertions_filename_template: Template for assertion filename
            (default: "{question_set}_assertions.json").
        answers_path_template: Template for answers file path
            (default: "{input_dir}/{generated_rag}/{question_set}.json").
        run_significance_test: Whether to run statistical significance tests
            (default: True).
        significance_alpha: Alpha level for significance tests (default: 0.05).
        significance_correction: P-value correction method (default: "holm").
        question_text_key: Column name for question text (default: "question_text").
        answer_text_key: Column name for answer text (default: "answer").

    Returns:
        DataFrame with overall results summary.
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
                question_text_key=question_text_key,
                answer_text_key=answer_text_key,
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

    # Run statistical significance tests if requested
    if run_significance_test and len(generated_rags) >= 2:
        compare_assertion_scores_significance(
            output_dir=output_dir,
            generated_rags=generated_rags,
            question_sets=question_sets,
            alpha=significance_alpha,
            correction_method=significance_correction,
        )

    return overall_summary_df


def run_hierarchical_assertion_evaluation(
    llm_client: ChatModel,
    llm_config: LLMConfig,
    generated_rags: list[str],
    assertions: pd.DataFrame,
    input_dir: str,
    output_dir: Path,
    trials: int,
    pass_threshold: float,
    mode: HierarchicalMode = HierarchicalMode.JOINT,
    answers_path_template: str = "{input_dir}/{generated_rag}/data_global.json",
    run_significance_test: bool = True,
    significance_alpha: float = 0.05,
    significance_correction: str = "holm",
    run_clustered_permutation: bool = False,
    n_permutations: int = 10_000,
    permutation_seed: int | None = None,
    question_id_key: str = "question_id",
    question_text_key: str = "question_text",
    answer_text_key: str = "answer",
    supporting_assertions_key: str = "supporting_assertions",
) -> pd.DataFrame:
    """Run hierarchical assertion evaluation for multiple RAG methods.

    This is a pipeline function that evaluates hierarchical assertions
    (with supporting assertions) across multiple RAG methods and optionally
    runs statistical significance tests.

    Args:
        llm_client: LLM client for evaluation.
        llm_config: LLM configuration.
        generated_rags: List of RAG method names.
        assertions: DataFrame with hierarchical assertions (must have
            supporting_assertions column).
        input_dir: Input directory containing RAG answer files.
        output_dir: Output directory for results.
        trials: Number of evaluation trials.
        pass_threshold: Threshold for assertion pass/fail.
        mode: Hierarchical evaluation mode (JOINT or STAGED).
        answers_path_template: Template for answers file path.
        run_significance_test: Whether to run statistical significance tests.
        significance_alpha: Alpha level for significance tests.
        significance_correction: P-value correction method.
        run_clustered_permutation: Whether to run assertion-level clustered
            permutation tests as secondary analysis.
        n_permutations: Number of permutations for clustered permutation tests.
        permutation_seed: Random seed for reproducibility of permutation tests.
        question_id_key: Column name for question ID.
        question_text_key: Column name for question text.
        answer_text_key: Column name for answer text.
        supporting_assertions_key: Column name for supporting assertions.

    Returns:
        DataFrame with comparison summary across all RAG methods.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_aggregated: list[pd.DataFrame] = []
    aggregated_scores_dict: dict[str, pd.DataFrame] = {}

    # Evaluate each RAG method
    for generated_rag in generated_rags:
        rich_print(f"\n[bold]Processing {generated_rag}[/bold]")

        # Build answers path
        answers_path = answers_path_template.format(
            input_dir=input_dir, generated_rag=generated_rag
        )

        try:
            answers = pd.read_json(answers_path)
        except FileNotFoundError:
            rich_print(
                f"  [yellow]Warning: {answers_path} not found, skipping[/yellow]"
            )
            continue

        # Run hierarchical scoring
        scores = get_hierarchical_assertion_scores(
            llm_client=llm_client,
            llm_config=llm_config,
            answers=answers,
            assertions=assertions,
            trials=trials,
            mode=mode,
            pass_threshold=pass_threshold,
            question_id_key=question_id_key,
            question_text_key=question_text_key,
            answer_text_key=answer_text_key,
            supporting_assertions_key=supporting_assertions_key,
        )

        # Save raw per-trial scores
        rag_output_dir = output_dir / generated_rag
        rag_output_dir.mkdir(parents=True, exist_ok=True)
        scores.to_csv(rag_output_dir / "hierarchical_scores_raw.csv", index=False)

        # Aggregate scores across trials
        aggregated = aggregate_hierarchical_scores(scores, pass_threshold=pass_threshold)
        aggregated["rag_method"] = generated_rag
        aggregated.to_csv(
            rag_output_dir / "hierarchical_scores_aggregated.csv", index=False
        )
        all_aggregated.append(aggregated)
        aggregated_scores_dict[generated_rag] = aggregated

        # Print summary stats (per-question averages, consistent with significance tests)
        per_q_support = aggregated.groupby("question")["support_level"].mean()
        if "support_level" in aggregated.columns:
            rich_print(
                f"  Average support level (per-question avg): "
                f"{per_q_support.mean() * 100:.1f}%"
            )

    if not all_aggregated:
        rich_print("[red]Error: No RAG methods were successfully processed[/red]")
        return pd.DataFrame()

    # Combine all results
    all_aggregated_df = pd.concat(all_aggregated, ignore_index=True)
    all_aggregated_df.to_csv(output_dir / "all_hierarchical_scores.csv", index=False)

    # Create comparison summary with per-question metrics
    # All metrics are computed per-question first, then averaged across questions
    # (consistent with how significance tests compare RAG methods)
    comparison_rows = []
    for rag_method, df in aggregated_scores_dict.items():
        # Compute per-question supporting pass rate (all assertions)
        # Count all supporting assertion evaluations (no deduplication)
        # because the same supporting assertion can have different results
        # under different global assertions
        per_question_supporting_rates = []
        for _, group in df.groupby("question"):
            total_supporting = 0
            total_passed = 0
            for support_results in group["support_results"]:
                if support_results:
                    total_supporting += len(support_results)
                    total_passed += sum(1 for sr in support_results if sr["passed"])
            if total_supporting > 0:
                pass_rate = total_passed / total_supporting
                per_question_supporting_rates.append(pass_rate)

        # Compute conditional metrics (only for passed global assertions)
        # These give a clearer picture of support quality by excluding
        # structural zeros from failed assertions.
        passed_df = df[df["global_score"] == 1]
        per_q_supp_rates_passed = []
        for _, group in passed_df.groupby("question"):
            total_supp = 0
            total_supp_passed = 0
            for support_results in group["support_results"]:
                if support_results:
                    total_supp += len(support_results)
                    total_supp_passed += sum(
                        1 for sr in support_results if sr["passed"]
                    )
            if total_supp > 0:
                per_q_supp_rates_passed.append(
                    total_supp_passed / total_supp
                )

        # All metrics: per-question average first, then mean across questions
        row: dict[str, object] = {
            "rag_method": rag_method,
            "global_pass_rate": (
                df.groupby("question")["global_score"].mean().mean()
            ),
            "avg_support_level": (
                df.groupby("question")["support_level"].mean().mean()
            ),
            "supporting_pass_rate": (
                np.mean(per_question_supporting_rates)
                if per_question_supporting_rates else 0.0
            ),
            "discovery_rate": (
                df.groupby("question")["has_discovery"].mean().mean()
            ),
        }

        # Conditional metrics (passed globals only)
        if not passed_df.empty:
            per_q = passed_df.groupby("question")
            row["support_level_passed"] = (
                per_q["support_level"].mean().mean()
            )
            row["supporting_pass_rate_passed"] = (
                np.mean(per_q_supp_rates_passed)
                if per_q_supp_rates_passed else 0.0
            )
            row["discovery_rate_passed"] = (
                per_q["has_discovery"].mean().mean()
            )
        else:
            row["support_level_passed"] = None
            row["supporting_pass_rate_passed"] = None
            row["discovery_rate_passed"] = None

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "hierarchical_comparison_summary.csv", index=False)
    print_df(comparison_df, "Hierarchical Assertion Scores Comparison")

    # Run statistical significance tests if requested
    if run_significance_test and len(aggregated_scores_dict) >= 2:
        rich_print("\n[bold]Running significance tests...[/bold]")
        sig_results = compare_hierarchical_assertion_scores_significance(
            aggregated_scores=aggregated_scores_dict,
            alpha=significance_alpha,
            correction_method=significance_correction,
            output_dir=output_dir,
            run_clustered_permutation=run_clustered_permutation,
            n_permutations=n_permutations,
            permutation_seed=permutation_seed,
        )

        # Build and display significance summary table
        sig_summary = summarize_significance_results(sig_results)
        if not sig_summary.empty:
            print_df(sig_summary, "Significance Test Summary")
            sig_summary.to_csv(
                output_dir / "significance_summary.csv", index=False
            )

    return comparison_df
