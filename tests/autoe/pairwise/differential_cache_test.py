# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for differential pairwise stage caching."""

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark_qed.autoe.config import Criteria
from benchmark_qed.autoe.data_model import (
    DifferentialCriterionVerdict,
    DifferentialPairwiseLLMResponse,
    PairwiseExtractionLLMResponse,
)
from benchmark_qed.autoe.pairwise import differential
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
        "base_name": "base",
        "other_name": "other",
        "base_answers": _answers("base answer"),
        "other_answers": _answers("other answer"),
        "criteria": [Criteria(name="relevance", description="description")],
        "trials": 2,
        "include_score_id_in_prompt": False,
        "cache_config": create_default_cache_config(
            tmp_path,
            database_name="differential_pairwise.sqlite3",
        ),
    }


def _response(value: Any) -> Any:
    return type("Response", (), {"formatted_response": value})()


def test_second_differential_run_reuses_both_stages(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls = {"extraction": 0, "verdict": 0}

    async def fake_chat(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        if kwargs["response_format"] is PairwiseExtractionLLMResponse:
            calls["extraction"] += 1
            return _response(
                PairwiseExtractionLLMResponse(
                    common="common",
                    unique_answer_1="first",
                    unique_answer_2="second",
                )
            )
        calls["verdict"] += 1
        return _response(
            DifferentialPairwiseLLMResponse(
                verdicts=[
                    DifferentialCriterionVerdict(
                        criteria="relevance",
                        winner=1,
                        reasoning="reason",
                    )
                ]
            )
        )

    monkeypatch.setattr(differential, "chat", fake_chat)
    kwargs = _config(tmp_path)

    first = differential.get_differential_pairwise_scores(**kwargs)
    second = differential.get_differential_pairwise_scores(**kwargs)

    assert calls == {"extraction": 2, "verdict": 2}
    pd.testing.assert_frame_equal(first, second)


def test_judge_failure_resumes_from_cached_extraction(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls = {"extraction": 0, "verdict": 0}
    fail_judge = True

    async def fake_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal fail_judge
        await asyncio.sleep(0)
        if kwargs["response_format"] is PairwiseExtractionLLMResponse:
            calls["extraction"] += 1
            return _response(
                PairwiseExtractionLLMResponse(
                    common="common",
                    unique_answer_1="first",
                    unique_answer_2="second",
                )
            )
        calls["verdict"] += 1
        if fail_judge:
            fail_judge = False
            msg = "transient"
            raise ConnectionError(msg)
        return _response(
            DifferentialPairwiseLLMResponse(
                verdicts=[
                    DifferentialCriterionVerdict(
                        criteria="relevance",
                        winner=1,
                        reasoning="reason",
                    )
                ]
            )
        )

    monkeypatch.setattr(differential, "chat", fake_chat)
    kwargs = _config(tmp_path)
    kwargs["trials"] = 1

    try:
        differential.get_differential_pairwise_scores(**kwargs)
    except ConnectionError:
        pass
    else:
        msg = "Expected the first judge call to fail"
        raise AssertionError(msg)

    result = differential.get_differential_pairwise_scores(**kwargs)

    assert calls == {"extraction": 1, "verdict": 2}
    assert len(result) == 1


def test_criteria_change_reuses_extraction_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls = {"extraction": 0, "verdict": 0}

    async def fake_chat(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        if kwargs["response_format"] is PairwiseExtractionLLMResponse:
            calls["extraction"] += 1
            return _response(
                PairwiseExtractionLLMResponse(
                    common="common",
                    unique_answer_1="first",
                    unique_answer_2="second",
                )
            )
        calls["verdict"] += 1
        user_prompt = kwargs["messages"][1]["content"]
        criterion = "accuracy" if "accuracy" in user_prompt else "relevance"
        return _response(
            DifferentialPairwiseLLMResponse(
                verdicts=[
                    DifferentialCriterionVerdict(
                        criteria=criterion,
                        winner=1,
                        reasoning="reason",
                    )
                ]
            )
        )

    monkeypatch.setattr(differential, "chat", fake_chat)
    kwargs = _config(tmp_path)
    kwargs["trials"] = 1

    differential.get_differential_pairwise_scores(**kwargs)
    kwargs["criteria"] = [Criteria(name="accuracy", description="description")]
    differential.get_differential_pairwise_scores(**kwargs)

    assert calls == {"extraction": 1, "verdict": 2}
