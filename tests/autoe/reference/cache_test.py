# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for reference score caching."""

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark_qed.autoe.config import Criteria
from benchmark_qed.autoe.data_model import ReferenceLLMResponse
from benchmark_qed.autoe.reference import scores
from benchmark_qed.cache import create_default_cache_config
from benchmark_qed.config.llm_config import AuthType, LLMConfig


def _answers(answer: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "question_id": "q1",
            "question_text": "question",
            "answer": answer,
        }
    ])


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "llm_client": object(),
        "llm_config": LLMConfig(
            auth_type=AuthType.AzureManagedIdentity,
            call_args={"temperature": 0},
        ),
        "generated_answers": _answers("generated answer"),
        "reference_answers": _answers("reference answer"),
        "criteria": [Criteria(name="correctness", description="description")],
        "trials": 2,
        "include_score_id_in_prompt": False,
        "cache_config": create_default_cache_config(
            tmp_path,
            database_name="reference.sqlite3",
        ),
    }


def test_second_reference_run_reuses_all_trials(
    tmp_path: Path, monkeypatch: Any
) -> None:
    call_count = 0

    async def fake_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        await asyncio.sleep(0)
        call_count += 1
        return type(
            "Response",
            (),
            {
                "formatted_response": ReferenceLLMResponse(
                    score=8,
                    reasoning="reason",
                )
            },
        )()

    monkeypatch.setattr(scores, "chat", fake_chat)
    kwargs = _config(tmp_path)

    first = scores.get_reference_scores(**kwargs)
    second = scores.get_reference_scores(**kwargs)

    assert call_count == 2
    pd.testing.assert_frame_equal(first, second)


def test_score_range_change_invalidates_reference_cache(
    tmp_path: Path, monkeypatch: Any
) -> None:
    call_count = 0

    async def fake_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        await asyncio.sleep(0)
        call_count += 1
        return type(
            "Response",
            (),
            {
                "formatted_response": ReferenceLLMResponse(
                    score=4,
                    reasoning="reason",
                )
            },
        )()

    monkeypatch.setattr(scores, "chat", fake_chat)
    kwargs = _config(tmp_path)
    kwargs["trials"] = 1

    scores.get_reference_scores(**kwargs)
    kwargs["score_max"] = 5
    scores.get_reference_scores(**kwargs)

    assert call_count == 2
