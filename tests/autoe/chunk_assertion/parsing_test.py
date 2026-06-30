# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for chunk-assertion response parsing and chunk collection."""

import pytest

from benchmark_qed.autoe.chunk_assertion.scoring import (
    _collect_chunks,
    _extract_grade,
)
from benchmark_qed.autoe.data_model.chunk_assertion import ChunkAssertionGrade
from benchmark_qed.autoe.data_model.retrieval_result import RetrievalResult


class TestExtractGrade:
    """Tests for _extract_grade response parsing."""

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("full_support", ChunkAssertionGrade.FULL_SUPPORT),
            ("partial_support", ChunkAssertionGrade.PARTIAL_SUPPORT),
            ("no_support", ChunkAssertionGrade.NO_SUPPORT),
            ("Label: full_support", ChunkAssertionGrade.FULL_SUPPORT),
            ("no_support: nothing relevant here", ChunkAssertionGrade.NO_SUPPORT),
            ("Full Support - clearly substantiated", ChunkAssertionGrade.FULL_SUPPORT),
            ("partial support, somewhat relevant", ChunkAssertionGrade.PARTIAL_SUPPORT),
        ],
    )
    def test_explicit_and_spaced_labels(self, response: str, expected: str) -> None:
        """Explicit and space-separated labels parse to the right grade."""
        assert _extract_grade(response) == expected

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("1", ChunkAssertionGrade.FULL_SUPPORT),
            ("0.5", ChunkAssertionGrade.PARTIAL_SUPPORT),
            ("0", ChunkAssertionGrade.NO_SUPPORT),
            ("1.0", ChunkAssertionGrade.FULL_SUPPORT),
            ("0.0", ChunkAssertionGrade.NO_SUPPORT),
            ("Score: 0.5\nReasoning: partly relevant", ChunkAssertionGrade.PARTIAL_SUPPORT),
            ("1 - the chunk states the claim directly", ChunkAssertionGrade.FULL_SUPPORT),
            ("0 - unrelated", ChunkAssertionGrade.NO_SUPPORT),
        ],
    )
    def test_numeric_scores(self, response: str, expected: str) -> None:
        """Numeric scores (1 / 0.5 / 0) are the primary parsed signal."""
        assert _extract_grade(response) == expected

    @pytest.mark.parametrize(
        "response",
        [
            "The chunk does not support the assertion.",
            "unsupported",
            "There is no support for this claim.",
            "",
            "   ",
            "completely irrelevant text",
        ],
    )
    def test_negatives_default_to_no_support(self, response: str) -> None:
        """Negatives and ambiguous text never misclassify as full_support."""
        assert _extract_grade(response) == ChunkAssertionGrade.NO_SUPPORT


class TestCollectChunks:
    """Tests for _collect_chunks flattening and ordering."""

    def _result(
        self, context: list[dict[str, object]], question_id: str = "q1"
    ) -> RetrievalResult:
        return RetrievalResult(
            question_id=question_id,
            question_text="question",
            context=context,
            context_id_key="chunk_id",
            context_text_key="text",
        )

    def test_sorts_by_rank(self) -> None:
        """Context items are ordered by their explicit rank field."""
        eval_result = self._result([
            {"chunk_id": "c3", "text": "third", "rank": 3},
            {"chunk_id": "c1", "text": "first", "rank": 1},
            {"chunk_id": "c2", "text": "second", "rank": 2},
        ])
        chunks = _collect_chunks(eval_result)
        assert [c["chunk_text"] for c in chunks] == ["first", "second", "third"]

    def test_preserves_order_without_rank(self) -> None:
        """Without rank, context list order is preserved."""
        eval_result = self._result([
            {"chunk_id": "c1", "text": "alpha"},
            {"chunk_id": "c2", "text": "beta"},
        ])
        chunks = _collect_chunks(eval_result)
        assert [c["chunk_text"] for c in chunks] == ["alpha", "beta"]

    def test_deduplicates(self) -> None:
        """Identical (chunk_id, text) pairs are collapsed to one entry."""
        eval_result = self._result([
            {"chunk_id": "c1", "text": "dup", "rank": 1},
            {"chunk_id": "c1", "text": "dup", "rank": 2},
            {"chunk_id": "c2", "text": "unique", "rank": 3},
        ])
        chunks = _collect_chunks(eval_result)
        assert [c["chunk_text"] for c in chunks] == ["dup", "unique"]

    def test_skips_empty_text(self) -> None:
        """Context items with blank text are ignored."""
        eval_result = self._result([
            {"chunk_id": "c1", "text": "   ", "rank": 1},
            {"chunk_id": "c2", "text": "kept", "rank": 2},
        ])
        chunks = _collect_chunks(eval_result)
        assert [c["chunk_text"] for c in chunks] == ["kept"]

    def test_empty_context_returns_empty(self) -> None:
        """An empty context yields no chunks."""
        assert _collect_chunks(self._result([])) == []
