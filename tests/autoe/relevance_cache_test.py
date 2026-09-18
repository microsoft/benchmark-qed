# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for retrieval relevance caching."""

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.relevance import (
    RelevanceAssessmentItem,
    RelevanceAssessmentResponse,
)
from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.base import (
    RelevanceRater,
)
from benchmark_qed.cache import SQLiteCache


class StubRelevanceRater(RelevanceRater):
    def __init__(
        self,
        cache_dir: Path | None,
        *,
        rater_params: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.rater_params = rater_params or {"model": "test-model"}
        super().__init__(cache_dir=cache_dir)

    def _get_cache_relevant_params(self) -> dict[str, Any]:
        return self.rater_params

    async def _rate_relevance_impl(
        self, query: str, text_units: list[TextUnit]
    ) -> RelevanceAssessmentResponse:
        self.calls.append([unit.id for unit in text_units])
        return RelevanceAssessmentResponse(
            assessment=[
                RelevanceAssessmentItem(
                    text_unit=unit,
                    reasoning=f"{query}:{unit.text}",
                    score=len(unit.text) % 4,
                )
                for unit in text_units
            ]
        )


def _text_unit(unit_id: str, text: str) -> TextUnit:
    return TextUnit(id=unit_id, short_id=None, text=text)


async def test_second_rater_uses_sqlite_cache(tmp_path: Path) -> None:
    units = [_text_unit("one", "first"), _text_unit("two", "second")]
    first = StubRelevanceRater(tmp_path)
    expected = await first.rate_relevance("query", units)

    second = StubRelevanceRater(tmp_path)
    actual = await second.rate_relevance("query", units)

    assert len(first.calls) == 1
    assert second.calls == []
    assert actual == expected
    assert second.cache_hits == 2
    assert second.get_cache_stats()["cache_files"] == 2


async def test_duplicate_pairs_share_one_assessment(tmp_path: Path) -> None:
    rater = StubRelevanceRater(tmp_path)

    result = await rater.rate_relevance(
        "query",
        [_text_unit("one", "same"), _text_unit("two", " SAME ")],
    )

    assert rater.calls == [["one"]]
    assert len(result.assessment) == 2
    assert result.assessment[0] == result.assessment[1]
    assert rater.cache_misses == 2
    assert rater.get_cache_stats()["cache_files"] == 1


async def test_configuration_metadata_is_redacted(tmp_path: Path) -> None:
    params = {
        "model": "test-model",
        "call_args": {"temperature": 0, "api_key": "secret"},
    }
    rater = StubRelevanceRater(tmp_path, rater_params=params)
    unit = _text_unit("one", "text")
    await rater.rate_relevance("query", [unit])

    key = rater._generate_cache_key("query", unit, params)
    entry = SQLiteCache(tmp_path / "relevance_cache.sqlite3", "StubRelevanceRater").get(
        key
    )

    assert entry is not None
    assert entry[1]["rater_params"]["call_args"] == {
        "api_key": "<redacted>",
        "temperature": 0,
    }
    assert "secret" not in json.dumps(entry)


async def test_imports_legacy_per_key_json(tmp_path: Path) -> None:
    unit = _text_unit("one", "legacy")
    key_builder = StubRelevanceRater(None)
    key = key_builder._generate_cache_key(
        "query", unit, key_builder._get_cache_relevant_params()
    )
    assessment = RelevanceAssessmentItem(
        text_unit=unit,
        reasoning="from legacy cache",
        score=3,
    )
    (tmp_path / f"{key}.json").write_text(
        json.dumps({
            "timestamp": "2026-09-11T00:00:00+00:00",
            "query": "query",
            "assessment_item": assessment.model_dump(),
        }),
        encoding="utf-8",
    )

    rater = StubRelevanceRater(tmp_path)
    result = await rater.rate_relevance("query", [unit])

    assert rater.calls == []
    assert result.assessment == [assessment]


async def test_clear_cache_removes_sqlite_and_legacy_entries(tmp_path: Path) -> None:
    rater = StubRelevanceRater(tmp_path)
    await rater.rate_relevance("query", [_text_unit("one", "text")])
    legacy_file = tmp_path / "legacy.json"
    legacy_file.write_text("{}", encoding="utf-8")

    rater.clear_cache()

    assert rater.get_cache_stats()["cache_files"] == 0
    assert not legacy_file.exists()
    assert rater.cache_hits == 0
    assert rater.cache_misses == 0


async def test_invalid_result_count_is_reported(tmp_path: Path) -> None:
    class InvalidRater(StubRelevanceRater):
        async def _rate_relevance_impl(
            self, query: str, text_units: list[TextUnit]
        ) -> RelevanceAssessmentResponse:
            return RelevanceAssessmentResponse(assessment=[])

    rater = InvalidRater(tmp_path)

    with pytest.raises(RuntimeError, match="returned 0 results for 1"):
        await rater.rate_relevance("query", [_text_unit("one", "text")])


async def test_warns_when_relevance_configuration_differs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    unit = _text_unit("one", "text")
    first = StubRelevanceRater(tmp_path, rater_params={"model": "first"})
    await first.rate_relevance("query", [unit])

    second = StubRelevanceRater(tmp_path, rater_params={"model": "second"})
    with caplog.at_level("WARNING"):
        await second.rate_relevance("query", [unit])

    assert "same inputs with different rater_params" in caplog.text
