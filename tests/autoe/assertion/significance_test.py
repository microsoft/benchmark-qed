"""Tests for standard assertion significance analysis."""

from pathlib import Path

import pandas as pd

from benchmark_qed.autoe.assertion.significance import (
    compare_assertion_scores_significance,
)


def _write_standard_score_inputs(base_dir: Path) -> None:
    """Create minimal standard assertion summary files for significance tests."""
    question_set_dir = base_dir / "qset_a"
    question_set_dir.mkdir(parents=True, exist_ok=True)

    # Per-question summaries used for the default significance path.
    pd.DataFrame(
        {
            "question": ["q1", "q2", "q3"],
            "success": [3, 3, 3],
            "fail": [0, 0, 0],
            "pass_rate": [1.0, 1.0, 1.0],
        }
    ).to_csv(
        question_set_dir / "rag_a_summary_by_question.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "question": ["q1", "q2", "q3"],
            "success": [0, 0, 0],
            "fail": [3, 3, 3],
            "pass_rate": [0.0, 0.0, 0.0],
        }
    ).to_csv(
        question_set_dir / "rag_b_summary_by_question.csv",
        index=False,
    )

    # Per-assertion summaries used for clustered permutation path.
    pd.DataFrame(
        {
            "question": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "assertion": ["a1", "a2", "a1", "a2", "a1", "a2"],
            "score": [1, 1, 1, 1, 1, 1],
            "score_mean": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "score_std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(
        question_set_dir / "rag_a_summary_by_assertion.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "question": ["q1", "q1", "q2", "q2", "q3", "q3"],
            "assertion": ["a1", "a2", "a1", "a2", "a1", "a2"],
            "score": [0, 0, 0, 0, 0, 0],
            "score_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "score_std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(
        question_set_dir / "rag_b_summary_by_assertion.csv",
        index=False,
    )


def test_compare_assertion_scores_significance_default(
    tmp_path: Path,
) -> None:
    """Run standard significance tests and verify default outputs exist."""
    _write_standard_score_inputs(tmp_path)

    results = compare_assertion_scores_significance(
        output_dir=tmp_path,
        generated_rags=["rag_a", "rag_b"],
        question_sets=["qset_a"],
        alpha=0.05,
        correction_method="holm",
    )

    assert "qset_a" in results

    question_set_dir = tmp_path / "qset_a"
    assert (question_set_dir / "significance_group_stats.csv").exists()
    assert (question_set_dir / "significance_omnibus.csv").exists()


def test_compare_assertion_scores_significance_with_clustered_permutation(
    tmp_path: Path,
) -> None:
    """Run optional clustered permutation tests for standard assertions."""
    _write_standard_score_inputs(tmp_path)

    results = compare_assertion_scores_significance(
        output_dir=tmp_path,
        generated_rags=["rag_a", "rag_b"],
        question_sets=["qset_a"],
        alpha=0.05,
        correction_method="holm",
        run_clustered_permutation=True,
        n_permutations=200,
        permutation_seed=7,
    )

    assert "qset_a" in results
    assert "qset_a_clustered" in results
    assert (
        "Clustered permutation test"
        in results["qset_a_clustered"].omnibus.test_name
    )

    question_set_dir = tmp_path / "qset_a"
    assert (
        question_set_dir / "significance_group_stats_clustered.csv"
    ).exists()
    assert (question_set_dir / "significance_omnibus_clustered.csv").exists()
