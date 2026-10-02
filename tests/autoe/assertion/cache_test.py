# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for standard assertion score caching."""

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from benchmark_qed.autoe.assertion import standard
from benchmark_qed.autoe.data_model import AssertionLLMResponse
from benchmark_qed.cache import create_default_cache_config
from benchmark_qed.config.llm_config import AuthType, LLMConfig


def _answers(answer: str = "answer") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "question_id": "q1",
            "question_text": "question",
            "answer": answer,
        }
    ])


def _assertions(assertion: str = "assertion") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "question_id": "q1",
            "question_text": "question",
            "assertion": assertion,
        }
    ])


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "llm_client": object(),
        "llm_config": LLMConfig(
            auth_type=AuthType.AzureManagedIdentity,
            call_args={"temperature": 0},
        ),
        "answers": _answers(),
        "assertions": _assertions(),
        "trials": 2,
        "include_score_id_in_prompt": False,
        "cache_config": create_default_cache_config(
            tmp_path,
            database_name="assertion.sqlite3",
        ),
    }


def test_second_assertion_run_reuses_all_trials(
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
                "formatted_response": AssertionLLMResponse(
                    score=1,
                    reasoning="reason",
                )
            },
        )()

    monkeypatch.setattr(standard, "chat", fake_chat)
    kwargs = _config(tmp_path)

    first = standard.get_assertion_scores(**kwargs)
    second = standard.get_assertion_scores(**kwargs)

    assert call_count == 2
    pd.testing.assert_frame_equal(first, second)


def test_answer_and_assertion_changes_invalidate_cache(
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
                "formatted_response": AssertionLLMResponse(
                    score=1,
                    reasoning="reason",
                )
            },
        )()

    monkeypatch.setattr(standard, "chat", fake_chat)
    kwargs = _config(tmp_path)
    kwargs["trials"] = 1

    standard.get_assertion_scores(**kwargs)
    kwargs["answers"] = _answers("changed answer")
    standard.get_assertion_scores(**kwargs)
    kwargs["assertions"] = _assertions("changed assertion")
    standard.get_assertion_scores(**kwargs)

    assert call_count == 3


def test_failed_assertion_call_is_not_cached(tmp_path: Path, monkeypatch: Any) -> None:
    call_count = 0

    async def fake_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        await asyncio.sleep(0)
        call_count += 1
        if call_count == 1:
            message = "transient failure"
            raise RuntimeError(message)
        return type(
            "Response",
            (),
            {
                "formatted_response": AssertionLLMResponse(
                    score=1,
                    reasoning="reason",
                )
            },
        )()

    monkeypatch.setattr(standard, "chat", fake_chat)
    kwargs = _config(tmp_path)
    kwargs["trials"] = 1

    with pytest.raises(RuntimeError, match="transient failure"):
        standard.get_assertion_scores(**kwargs)
    result = standard.get_assertion_scores(**kwargs)

    assert call_count == 2
    assert len(result) == 1
