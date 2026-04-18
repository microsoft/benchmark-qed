# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for load_and_normalize_hierarchical_assertions."""

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmark_qed.autoe.assertion.pipeline import (
    load_and_normalize_hierarchical_assertions,
)


def _make_assertions_file(
    tmp_path: Path,
    data: list[dict],
    filename: str = "assertions.json",
) -> Path:
    """Write assertion data to a JSON file and return its path."""
    path = tmp_path / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _sample_hierarchical_data() -> list[dict]:
    """Return minimal hierarchical assertion data for testing."""
    return [
        {
            "question_id": "q1",
            "question_text": "What is X?",
            "assertions": [
                {
                    "statement": "X is a thing.",
                    "rank": 1,
                    "supporting_assertions": ["SA1", "SA2"],
                },
                {
                    "statement": "X is also Y.",
                    "rank": 2,
                    "supporting_assertions": ["SA3"],
                },
            ],
        },
        {
            "question_id": "q2",
            "question_text": "What is Z?",
            "assertions": [
                {
                    "statement": "Z is everything.",
                    "rank": 1,
                    "supporting_assertions": ["SA4"],
                },
            ],
        },
    ]


class TestLoadAndNormalizeHierarchicalAssertions:
    """Tests for load_and_normalize_hierarchical_assertions."""

    def test_basic_loading(self, tmp_path: Path) -> None:
        """Load valid hierarchical assertions and verify structure."""
        path = _make_assertions_file(tmp_path, _sample_hierarchical_data())
        result = load_and_normalize_hierarchical_assertions(path)

        # Should have 3 rows total (2 from q1 + 1 from q2)
        assert len(result) == 3
        assert "assertion" in result.columns
        assert "supporting_assertions" in result.columns
        assert "question_id" in result.columns

    def test_statement_renamed_to_assertion(self, tmp_path: Path) -> None:
        """Verify 'statement' column is renamed to 'assertion'."""
        path = _make_assertions_file(tmp_path, _sample_hierarchical_data())
        result = load_and_normalize_hierarchical_assertions(path)

        assert "statement" not in result.columns
        assert "assertion" in result.columns
        assert result["assertion"].iloc[0] == "X is a thing."

    def test_filters_empty_supporting_assertions(self, tmp_path: Path) -> None:
        """Assertions without supporting assertions are filtered out."""
        data = [
            {
                "question_id": "q1",
                "question_text": "What?",
                "assertions": [
                    {
                        "statement": "Has support.",
                        "rank": 1,
                        "supporting_assertions": ["SA1"],
                    },
                    {
                        "statement": "No support.",
                        "rank": 2,
                        "supporting_assertions": [],
                    },
                ],
            }
        ]
        path = _make_assertions_file(tmp_path, data)
        result = load_and_normalize_hierarchical_assertions(path)

        assert len(result) == 1
        assert result["assertion"].iloc[0] == "Has support."

    def test_missing_assertions_key_raises(self, tmp_path: Path) -> None:
        """Raise ValueError when assertions key column is missing."""
        data = [{"question_id": "q1", "other_col": "value"}]
        path = _make_assertions_file(tmp_path, data)

        with pytest.raises(ValueError, match="missing required"):
            load_and_normalize_hierarchical_assertions(path)

    def test_missing_supporting_assertions_raises(self, tmp_path: Path) -> None:
        """Raise ValueError when supporting_assertions column is missing."""
        data = [
            {
                "question_id": "q1",
                "question_text": "What?",
                "assertions": [
                    {"statement": "X.", "rank": 1},
                ],
            }
        ]
        path = _make_assertions_file(tmp_path, data)

        with pytest.raises(ValueError, match="missing"):
            load_and_normalize_hierarchical_assertions(path)

    def test_all_filtered_raises(self, tmp_path: Path) -> None:
        """Raise ValueError when all assertions are filtered out."""
        data = [
            {
                "question_id": "q1",
                "question_text": "What?",
                "assertions": [
                    {
                        "statement": "X.",
                        "rank": 1,
                        "supporting_assertions": [],
                    },
                ],
            }
        ]
        path = _make_assertions_file(tmp_path, data)

        with pytest.raises(ValueError, match="No valid"):
            load_and_normalize_hierarchical_assertions(path)

    def test_custom_assertions_key(self, tmp_path: Path) -> None:
        """Support custom assertion column names."""
        data = [
            {
                "question_id": "q1",
                "question_text": "What?",
                "claims": [
                    {
                        "statement": "A claim.",
                        "rank": 1,
                        "supporting_assertions": ["SA1"],
                    },
                ],
            }
        ]
        path = _make_assertions_file(tmp_path, data)
        result = load_and_normalize_hierarchical_assertions(
            path, assertions_key="claims"
        )

        assert len(result) == 1
        assert result["assertion"].iloc[0] == "A claim."

    def test_non_dict_assertions_renamed(self, tmp_path: Path) -> None:
        """Non-dict assertions use the assertions_key rename path."""
        # Build a DataFrame where assertions are plain strings but
        # supporting_assertions is a separate column
        assertions_df = pd.DataFrame({
            "question_id": ["q1"],
            "question_text": ["What?"],
            "assertions": ["A plain assertion."],
            "supporting_assertions": [["SA1"]],
        })
        path = tmp_path / "assertions.json"
        assertions_df.to_json(path, orient="records")

        result = load_and_normalize_hierarchical_assertions(path)

        assert "assertion" in result.columns
        assert result["assertion"].iloc[0] == "A plain assertion."

    def test_preserves_rank_column(self, tmp_path: Path) -> None:
        """Rank column is preserved after normalization."""
        path = _make_assertions_file(tmp_path, _sample_hierarchical_data())
        result = load_and_normalize_hierarchical_assertions(path)

        assert "rank" in result.columns
        assert result["rank"].iloc[0] == 1

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        """Accept both str and Path for assertions_path."""
        path = _make_assertions_file(tmp_path, _sample_hierarchical_data())
        # Pass as string
        result_str = load_and_normalize_hierarchical_assertions(str(path))
        # Pass as Path
        result_path = load_and_normalize_hierarchical_assertions(path)

        assert len(result_str) == len(result_path)
