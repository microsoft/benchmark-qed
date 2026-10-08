# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for pairwise score caching."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
from graphrag_cache import CacheConfig, CacheType

from benchmark_qed.autoe.config import Criteria
from benchmark_qed.autoe.data_model import PairwiseLLMResponse
from benchmark_qed.autoe.pairwise import scores
from benchmark_qed.autoe.pairwise.cache import (
    PairwiseScoreCache,
    build_cache_metadata,
    compute_cache_key,
    compute_logical_key,
)
from benchmark_qed.cache import create_default_cache_config
from benchmark_qed.config.llm_config import AuthType, LLMConfig

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion


def _metadata(*, model: str = "gpt-4.1") -> dict[str, Any]:
    return build_cache_metadata(
        model=model,
        llm_provider="openai.chat",
        init_args={},
        call_args={"temperature": 0},
        custom_providers=[],
        system_prompt="system",
        user_prompt="user",
        include_score_id_in_prompt=True,
    )


def _logical_key(*, trial: int = 0) -> str:
    return compute_logical_key(
        question="question",
        answer_1_name="base",
        answer_1="base answer",
        answer_2_name="other",
        answer_2="other answer",
        criteria_name="relevance",
        criteria_description="description",
        trial=trial,
    )


def test_cache_key_distinguishes_trials_and_model() -> None:
    assert compute_cache_key(_logical_key(trial=0), _metadata()) != compute_cache_key(
        _logical_key(trial=1), _metadata()
    )
    assert compute_cache_key(_logical_key(), _metadata()) != compute_cache_key(
        _logical_key(), _metadata(model="gpt-4o")
    )


def test_cache_metadata_redacts_credentials() -> None:
    metadata = build_cache_metadata(
        model="model",
        llm_provider="custom.chat",
        init_args={"api_key": "secret"},
        call_args={"headers": {"Authorization": "secret"}},
        custom_providers=[],
        system_prompt="system",
        user_prompt="user",
        include_score_id_in_prompt=False,
    )

    assert metadata["init_args"]["api_key"] == "<redacted>"
    assert metadata["call_args"]["headers"]["Authorization"] == "<redacted>"


async def test_noop_cache_does_not_reuse_scores(monkeypatch: Any) -> None:
    call_count = 0

    async def fake_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        await asyncio.sleep(0)
        call_count += 1
        return type(
            "Response",
            (),
            {"formatted_response": PairwiseLLMResponse(winner=1, reasoning="reason")},
        )()

    monkeypatch.setattr(scores, "chat", fake_chat)
    cache_config = CacheConfig(type=CacheType.Noop, storage=None)

    await scores._get_pairwise_score_with_cache(
        cast("LLMCompletion", object()),
        cache=scores.PairwiseScoreCache(cache_config),
        cache_metadata=_metadata(),
        question="question",
        answer_1_name="base",
        answer_1="base answer",
        answer_2_name="other",
        answer_2="other answer",
        criteria_name="relevance",
        criteria_description="description",
        assessment_system_prompt=scores.Template(
            "$criteria_name $criteria_description"
        ),
        assessment_user_prompt=scores.Template(
            "$score_id $question $answer1 $answer2 $criteria_name $criteria_description"
        ),
        complete_callback=None,
        trial=0,
        include_score_id_in_prompt=True,
        additional_call_args={},
    )

    assert call_count == 1


async def test_publish_in_claim_window_returns_cached_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = PairwiseScoreCache(CacheConfig(type=CacheType.Memory, storage=None))
    canonical = {"reasoning": "cached winner"}
    get_calls = 0
    judge_called = False
    completed = 0

    async def racing_get(
        _cache_key: str,
    ) -> tuple[Any, dict[str, Any]] | None:
        nonlocal get_calls
        await asyncio.sleep(0)
        get_calls += 1
        return None if get_calls == 1 else (canonical, {})

    async def fail_if_judged(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal judge_called
        await asyncio.sleep(0)
        judge_called = True
        return {"reasoning": "wrong"}

    def on_complete() -> None:
        nonlocal completed
        completed += 1

    monkeypatch.setattr(cache._store, "get", racing_get)
    monkeypatch.setattr(scores, "get_pairwise_score", fail_if_judged)

    result = await scores._get_pairwise_score_with_cache(
        cast("LLMCompletion", object()),
        cache=cache,
        cache_metadata=_metadata(),
        question="question",
        answer_1_name="base",
        answer_1="base answer",
        answer_2_name="other",
        answer_2="other answer",
        criteria_name="relevance",
        criteria_description="description",
        assessment_system_prompt=scores.Template(
            "$criteria_name $criteria_description"
        ),
        assessment_user_prompt=scores.Template(
            "$score_id $question $answer1 $answer2 $criteria_name $criteria_description"
        ),
        complete_callback=on_complete,
        trial=0,
        include_score_id_in_prompt=True,
        additional_call_args={},
    )

    cache_key = compute_cache_key(_logical_key(), _metadata())
    assert result == canonical
    assert not judge_called
    assert completed == 1
    assert cache._store.try_acquire(cache_key, "next-owner", ttl_seconds=30)
    cache._store.release_leases([cache_key], "next-owner")


@pytest.mark.parametrize("include_score_id_in_prompt", [True, False])
def test_four_trials_are_independent_and_rerun_uses_cache(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    include_score_id_in_prompt: bool,
) -> None:
    call_count = 0

    async def fake_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        await asyncio.sleep(0)
        call_count += 1
        return type(
            "Response",
            (),
            {"formatted_response": PairwiseLLMResponse(winner=1, reasoning="reason")},
        )()

    monkeypatch.setattr(scores, "chat", fake_chat)
    llm_config = LLMConfig(
        auth_type=AuthType.AzureManagedIdentity,
        call_args={"temperature": 0},
    )
    cache_config = create_default_cache_config(
        tmp_path,
        database_name="pairwise.sqlite3",
    )
    base_answers = pd.DataFrame([
        {
            "question_id": "q1",
            "question_text": "question",
            "answer": "base answer",
        }
    ])
    other_answers = pd.DataFrame([
        {
            "question_id": "q1",
            "question_text": "question",
            "answer": "other answer",
        }
    ])
    kwargs = {
        "llm_client": object(),
        "llm_config": llm_config,
        "base_name": "base",
        "other_name": "other",
        "base_answers": base_answers,
        "other_answers": other_answers,
        "criteria": [Criteria(name="relevance", description="description")],
        "trials": 4,
        "include_score_id_in_prompt": include_score_id_in_prompt,
        "cache_config": cache_config,
    }

    first = scores.get_pairwise_scores(**kwargs)
    second = scores.get_pairwise_scores(**kwargs)

    assert call_count == 4
    assert first["trial"].tolist() == [0, 1, 2, 3]
    assert first["answer_1_name"].tolist() == ["base", "other", "base", "other"]
    pd.testing.assert_frame_equal(first, second)
