# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for calibrated absolute scoring."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

import benchmark_qed.autoe.calibrated.evaluation as calibrated_evaluation
import benchmark_qed.autoe.calibrated.scoring as calibrated
from benchmark_qed.autoe.calibrated import (
    AbsoluteScoreResponse,
    AnswerItem,
    BatchRankingResponse,
    DirectAbsoluteScoreResponse,
    aggregate_passes,
    answer_items_from_frame,
    calibrate_answers,
    compare_direct_and_calibrated,
    map_ratings_to_levels,
    schedule_batches,
    score_answers,
    score_direct_baseline,
    select_percentile_exemplars,
)
from benchmark_qed.autoe.calibrated.scoring import RankingItem
from benchmark_qed.autoe.cli import _summarize_calibrated_absolute_scores
from benchmark_qed.cache import create_default_cache_config
from benchmark_qed.config.llm_config import LLMConfig
from benchmark_qed.config.model.score import Criteria


def test_schedule_batches_is_balanced_and_deterministic() -> None:
    item_ids = [f"item-{index}" for index in range(8)]

    first = schedule_batches(item_ids, appearances=3, seed=42)
    second = schedule_batches(item_ids, appearances=3, seed=42)

    assert first == second
    assert all(2 <= len(batch) <= 5 for batch in first)
    assert {
        item_id: sum(item_id in batch for batch in first) for item_id in item_ids
    } == dict.fromkeys(item_ids, 3)


def test_rating_levels_and_exemplars_cover_scale() -> None:
    ratings = {f"item-{index}": float(index) for index in range(15)}

    levels = map_ratings_to_levels(ratings)
    exemplars = select_percentile_exemplars(ratings, levels)

    assert set(levels.values()) == {1, 2, 3, 4, 5}
    assert set(exemplars) == {"25", "50", "75"}
    assert all(
        set(exemplar_set) == {"1", "2", "3", "4", "5"}
        for exemplar_set in exemplars.values()
    )


def test_answer_items_reject_duplicate_question_ids() -> None:
    frame = pd.DataFrame([
        {"question_id": "q1", "question_text": "one", "answer": "first"},
        {"question_id": "q1", "question_text": "two", "answer": "second"},
    ])

    with pytest.raises(ValueError, match="duplicate question IDs"):
        answer_items_from_frame(frame, condition="test")


def test_absolute_summary_reports_insufficient_evidence() -> None:
    scores = pd.DataFrame([
        {
            "target_id": "test:q1",
            "condition": "test",
            "criteria": "quality",
            "status": "ok",
            "level": 4,
            "confidence_score": 0.9,
        },
        {
            "target_id": "test:q2",
            "condition": "test",
            "criteria": "quality",
            "status": "insufficient_evidence",
            "level": None,
            "confidence_score": 0.0,
        },
    ])

    summary = _summarize_calibrated_absolute_scores(scores)

    assert summary.loc[0, "answers"] == 2
    assert summary.loc[0, "scored_answers"] == 1
    assert summary.loc[0, "insufficient_evidence"] == 1
    assert summary.loc[0, "mean_level"] == 4


def test_absolute_response_rejects_nonadjacent_levels() -> None:
    with pytest.raises(ValidationError, match="closest levels must be"):
        AbsoluteScoreResponse(
            status="ok",
            level=4,
            confidence="medium",
            closest_level=4,
            second_closest_level=2,
            midpoint_relation="better",
            is_borderline=False,
            evidence=["The target is stronger."],
            rationale="Compared with all anchors.",
        )


def test_aggregate_passes_uses_middle_vote_when_all_differ() -> None:
    records = [
        {
            "percentile": percentile,
            "result": {
                "status": "ok",
                "level": level,
                "confidence": "medium",
                "closest_level": level,
                "second_closest_level": level - 1,
                "midpoint_relation": "better",
                "is_borderline": False,
                "evidence": ["evidence"],
                "rationale": "rationale",
            },
        }
        for percentile, level in [(50, 2), (75, 4), (25, 3)]
    ]

    result = aggregate_passes("target", "quality", records, 3)

    assert result["level"] == 3
    assert result["votes"] == [2, 4, 3]
    assert result["is_borderline"] is True


def test_compare_direct_and_calibrated_reports_reliability_separately() -> None:
    direct = pd.DataFrame([
        {
            "target_id": "target:q1",
            "criteria": "quality",
            "trial": 1,
            "level": 2,
            "elapsed_seconds": 1.0,
        },
        {
            "target_id": "target:q1",
            "criteria": "quality",
            "trial": 2,
            "level": 2,
            "elapsed_seconds": 2.0,
        },
        {
            "target_id": "target:q1",
            "criteria": "quality",
            "trial": 3,
            "level": 3,
            "elapsed_seconds": 3.0,
        },
        {
            "target_id": "target:q2",
            "criteria": "quality",
            "trial": 1,
            "level": 5,
            "elapsed_seconds": 2.0,
        },
        {
            "target_id": "target:q2",
            "criteria": "quality",
            "trial": 2,
            "level": 5,
            "elapsed_seconds": 2.0,
        },
        {
            "target_id": "target:q2",
            "criteria": "quality",
            "trial": 3,
            "level": 5,
            "elapsed_seconds": 2.0,
        },
    ])
    calibrated_scores = pd.DataFrame([
        {
            "target_id": "target:q1",
            "criteria": "quality",
            "level": 3,
            "vote_agreement": 2 / 3,
        },
        {
            "target_id": "target:q2",
            "criteria": "quality",
            "level": 5,
            "vote_agreement": 1.0,
        },
    ])

    items, summary = compare_direct_and_calibrated(direct, calibrated_scores)

    first = items.loc[items["target_id"] == "target:q1"].iloc[0]
    assert first["direct_level"] == 2
    assert first["direct_pairwise_agreement"] == pytest.approx(1 / 3)
    assert first["absolute_level_difference"] == 1
    assert not first["method_exact_agreement"]
    assert first["method_within_one"]
    assert summary.loc[0, "direct_unanimous_rate"] == 0.5
    assert summary.loc[0, "method_exact_agreement"] == 0.5
    assert summary.loc[0, "method_within_one"] == 1.0


def test_compare_direct_and_calibrated_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match=r"direct\.elapsed_seconds"):
        compare_direct_and_calibrated(
            pd.DataFrame([
                {"target_id": "q1", "criteria": "quality", "trial": 1, "level": 3}
            ]),
            pd.DataFrame([
                {
                    "target_id": "q1",
                    "criteria": "quality",
                    "level": 3,
                    "vote_agreement": 1.0,
                }
            ]),
        )


async def test_direct_baseline_makes_one_call_per_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = AnswerItem(
        id="target:q1",
        condition="target",
        question_id="q1",
        question="question",
        answer="answer",
    )
    criterion = Criteria(name="quality", description="Overall answer quality.")
    calls: list[list[dict[str, str]]] = []
    completed: list[str] = []

    async def fake_chat(
        *args: object, messages: list[dict[str, str]], **kwargs: object
    ) -> object:
        await asyncio.sleep(0)
        calls.append(messages)
        return SimpleNamespace(
            formatted_response=DirectAbsoluteScoreResponse(
                level=3,
                confidence="medium",
                evidence=["The answer addresses the question."],
                rationale="The answer is adequate but limited.",
            )
        )

    monkeypatch.setattr(calibrated_evaluation, "chat", fake_chat)

    records = await score_direct_baseline(
        llm=object(),  # type: ignore[arg-type]
        llm_config=LLMConfig.model_validate({"auth_type": "azure_managed_identity"}),
        targets=[target],
        criteria=[criterion],
        trials=3,
        complete_callback=completed.append,
    )

    assert len(records) == 3
    assert [record["trial"] for record in records] == [1, 2, 3]
    assert completed == ["quality", "quality", "quality"]
    assert len(calls) == 3
    assert all('"scale"' in call[1]["content"] for call in calls)
    assert all("exemplar" not in call[1]["content"].lower() for call in calls)


async def test_rank_batch_retries_semantically_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = [
        AnswerItem(
            id=f"reference:q{index}",
            condition="reference",
            question_id=f"q{index}",
            question=f"question {index}",
            answer=f"answer {index}",
        )
        for index in range(2)
    ]
    responses = [
        BatchRankingResponse(
            ranking=[
                RankingItem(item_id=batch[0].id, rank=1),
                RankingItem(item_id=batch[0].id, rank=2),
            ],
            rationale="invalid duplicate",
        ),
        BatchRankingResponse(
            ranking=[
                RankingItem(item_id=batch[0].id, rank=1),
                RankingItem(item_id=batch[1].id, rank=2),
            ],
            rationale="corrected",
        ),
    ]
    calls: list[list[dict[str, str]]] = []

    async def fake_chat(
        *args: object, messages: list[dict[str, str]], **kwargs: object
    ) -> object:
        await asyncio.sleep(0)
        calls.append(messages.copy())
        return SimpleNamespace(formatted_response=responses[len(calls) - 1])

    monkeypatch.setattr(calibrated, "chat", fake_chat)

    response = await calibrated._rank_batch(
        object(),  # type: ignore[arg-type]
        LLMConfig.model_validate({"auth_type": "azure_managed_identity"}),
        batch,
        Criteria(name="quality", description="Overall answer quality."),
    )

    assert response.rationale == "corrected"
    assert len(calls) == 2
    assert "duplicate item IDs" in calls[1][-1]["content"]
    assert "missing item IDs" in calls[1][-1]["content"]


async def test_rank_batch_reports_invalid_response_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = [
        AnswerItem(
            id=f"reference:q{index}",
            condition="reference",
            question_id=f"q{index}",
            question=f"question {index}",
            answer=f"answer {index}",
        )
        for index in range(2)
    ]
    invalid = BatchRankingResponse(
        ranking=[RankingItem(item_id=batch[0].id, rank=1)],
        rationale="still invalid",
    )

    async def fake_chat(*args: object, **kwargs: object) -> object:
        await asyncio.sleep(0)
        return SimpleNamespace(formatted_response=invalid)

    monkeypatch.setattr(calibrated, "chat", fake_chat)

    with pytest.raises(
        RuntimeError,
        match=r"invalid ranking after 3 attempts.*missing item IDs",
    ):
        await calibrated._rank_batch(
            object(),  # type: ignore[arg-type]
            LLMConfig.model_validate({"auth_type": "azure_managed_identity"}),
            batch,
            Criteria(name="quality", description="Overall answer quality."),
        )


async def test_score_pass_retries_semantically_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = AnswerItem(
        id="target:q0",
        condition="target",
        question_id="q0",
        question="question",
        answer="answer",
    )
    exemplars = {
        level: target.model_copy(
            update={"id": f"reference:q{level}", "condition": "reference"}
        )
        for level in range(1, 6)
    }
    responses = [
        calibrated.AbsoluteScoreLLMResponse(
            status="insufficient_evidence",
            level=3,
            confidence="medium",
            closest_level=3,
            second_closest_level=2,
            midpoint_relation="similar",
            is_borderline=False,
            evidence=["The answer lacks evidence."],
            rationale="Invalid mixed response.",
        ),
        calibrated.AbsoluteScoreLLMResponse(
            status="insufficient_evidence",
            level=None,
            confidence="n/a",
            closest_level=None,
            second_closest_level=None,
            midpoint_relation="insufficient_evidence",
            is_borderline=False,
            evidence=["The answer lacks evidence."],
            rationale="Corrected response.",
        ),
    ]
    calls: list[list[dict[str, str]]] = []

    async def fake_chat(
        *args: object, messages: list[dict[str, str]], **kwargs: object
    ) -> object:
        await asyncio.sleep(0)
        calls.append(messages.copy())
        return SimpleNamespace(formatted_response=responses[len(calls) - 1])

    monkeypatch.setattr(calibrated, "chat", fake_chat)

    response = await calibrated._score_one_pass_impl(
        object(),  # type: ignore[arg-type]
        LLMConfig.model_validate({"auth_type": "azure_managed_identity"}),
        target,
        exemplars,
        Criteria(name="quality", description="Overall answer quality."),
    )

    assert response.status == "insufficient_evidence"
    assert response.level is None
    assert len(calls) == 2
    assert "insufficient evidence cannot assign" in calls[1][-1]["content"]


async def test_score_pass_reports_invalid_response_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = AnswerItem(
        id="target:q0",
        condition="target",
        question_id="q0",
        question="question",
        answer="answer",
    )
    exemplars = {
        level: target.model_copy(
            update={"id": f"reference:q{level}", "condition": "reference"}
        )
        for level in range(1, 6)
    }
    invalid = calibrated.AbsoluteScoreLLMResponse(
        status="insufficient_evidence",
        level=3,
        confidence="medium",
        closest_level=3,
        second_closest_level=2,
        midpoint_relation="similar",
        is_borderline=False,
        evidence=["The answer lacks evidence."],
        rationale="Invalid mixed response.",
    )

    async def fake_chat(*args: object, **kwargs: object) -> object:
        await asyncio.sleep(0)
        return SimpleNamespace(formatted_response=invalid)

    monkeypatch.setattr(calibrated, "chat", fake_chat)

    with pytest.raises(
        RuntimeError,
        match=r"invalid absolute score after 3 attempts.*target:q0",
    ):
        await calibrated._score_one_pass_impl(
            object(),  # type: ignore[arg-type]
            LLMConfig.model_validate({"auth_type": "azure_managed_identity"}),
            target,
            exemplars,
            Criteria(name="quality", description="Overall answer quality."),
        )


async def test_calibrate_and_score_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        AnswerItem(
            id=f"reference:q{index}",
            condition="reference",
            question_id=f"q{index}",
            question=f"question {index}",
            answer=f"answer {index}",
        )
        for index in range(5)
    ]
    criterion = Criteria(name="quality", description="Overall answer quality.")
    llm_config = LLMConfig.model_validate({"auth_type": "azure_managed_identity"})

    async def fake_rank_batch(*args: object) -> BatchRankingResponse:
        await asyncio.sleep(0)
        batch = args[2]
        assert isinstance(batch, list)
        return BatchRankingResponse(
            ranking=[
                RankingItem(item_id=item.id, rank=index)
                for index, item in enumerate(reversed(batch), start=1)
            ],
            rationale="ordered by fixture",
        )

    monkeypatch.setattr(calibrated, "_rank_batch", fake_rank_batch)
    calibration = await calibrate_answers(
        llm=object(),  # type: ignore[arg-type]
        llm_config=llm_config,
        items=items,
        criteria=[criterion],
        appearances=1,
    )

    assert set(calibration["scales"]["quality"]["levels"].values()) == {
        1,
        2,
        3,
        4,
        5,
    }

    async def fake_score_one_pass(*args: object) -> AbsoluteScoreResponse:
        await asyncio.sleep(0)
        return AbsoluteScoreResponse(
            status="ok",
            level=3,
            confidence="high",
            closest_level=3,
            second_closest_level=2,
            midpoint_relation="similar",
            is_borderline=False,
            evidence=["fixture evidence"],
            rationale="fixture rationale",
        )

    monkeypatch.setattr(calibrated, "_score_one_pass", fake_score_one_pass)
    target = items[0].model_copy(update={"id": "target:q0", "condition": "target"})
    scores = await score_answers(
        llm=object(),  # type: ignore[arg-type]
        llm_config=llm_config,
        targets=[target],
        calibration=calibration,
        passes=3,
    )

    assert len(scores) == 1
    assert scores[0]["level"] == 3
    assert scores[0]["confidence"] == "high"
    assert scores[0]["votes"] == [3, 3, 3]


async def test_calibration_and_scoring_reruns_use_sqlite_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        AnswerItem(
            id=f"reference:q{index}",
            condition="reference",
            question_id=f"q{index}",
            question=f"question {index}",
            answer=f"answer {index}",
        )
        for index in range(5)
    ]
    criterion = Criteria(name="quality", description="Overall answer quality.")
    llm_config = LLMConfig.model_validate({"auth_type": "azure_managed_identity"})
    calibration_calls = 0

    async def fake_rank_impl(*args: object) -> BatchRankingResponse:
        nonlocal calibration_calls
        await asyncio.sleep(0)
        calibration_calls += 1
        batch = args[2]
        assert isinstance(batch, list)
        return BatchRankingResponse(
            ranking=[
                RankingItem(item_id=item.id, rank=index)
                for index, item in enumerate(batch, start=1)
            ],
            rationale="cached ranking",
        )

    monkeypatch.setattr(calibrated, "_rank_batch_impl", fake_rank_impl)
    calibration_cache = create_default_cache_config(
        tmp_path / "calibration",
        database_name="cache.sqlite3",
    )
    kwargs = {
        "llm": object(),
        "llm_config": llm_config,
        "items": items,
        "criteria": [criterion],
        "appearances": 1,
        "cache_config": calibration_cache,
    }
    first_calibration = await calibrate_answers(**kwargs)  # type: ignore[arg-type]
    second_calibration = await calibrate_answers(**kwargs)  # type: ignore[arg-type]

    assert calibration_calls == 1
    assert second_calibration == first_calibration

    scoring_calls = 0

    async def fake_score_impl(*args: object) -> AbsoluteScoreResponse:
        nonlocal scoring_calls
        await asyncio.sleep(0)
        scoring_calls += 1
        return AbsoluteScoreResponse(
            status="ok",
            level=3,
            confidence="high",
            closest_level=3,
            second_closest_level=2,
            midpoint_relation="similar",
            is_borderline=False,
            evidence=["cached evidence"],
            rationale="cached rationale",
        )

    monkeypatch.setattr(calibrated, "_score_one_pass_impl", fake_score_impl)
    target = items[0].model_copy(update={"id": "target:q0", "condition": "target"})
    scoring_cache = create_default_cache_config(
        tmp_path / "scoring",
        database_name="cache.sqlite3",
    )
    completed: list[str] = []
    score_kwargs = {
        "llm": object(),
        "llm_config": llm_config,
        "targets": [target],
        "calibration": first_calibration,
        "passes": 3,
        "cache_config": scoring_cache,
        "complete_callback": completed.append,
    }
    first_scores = await score_answers(**score_kwargs)  # type: ignore[arg-type]
    second_scores = await score_answers(**score_kwargs)  # type: ignore[arg-type]

    assert scoring_calls == 1
    assert second_scores == first_scores
    assert completed == ["quality"] * 6
