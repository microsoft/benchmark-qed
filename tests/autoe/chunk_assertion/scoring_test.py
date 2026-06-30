# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Integration tests for run_assertion_eval_chunk_mode.

Covers question_id alignment, @k truncation with rank ordering, and cache reuse
across runs, with the LLM call mocked out.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from graphrag_storage.file_storage import FileStorage

from benchmark_qed.autoe.chunk_assertion.scoring import run_assertion_eval_chunk_mode
from benchmark_qed.autoe.data_model.retrieval_result import (
    load_retrieval_results_from_dicts,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

    from benchmark_qed.config.llm_config import LLMConfig


@pytest.fixture
def grade_by_chunk() -> dict[str, str]:
    """Mutable mapping from chunk text to the grade the fake LLM should return."""
    return {}


@pytest.fixture
def call_counter() -> list[int]:
    """Single-element list tracking how many LLM calls were made."""
    return [0]


@pytest.fixture
def patched_chat(
    monkeypatch: pytest.MonkeyPatch,
    grade_by_chunk: dict[str, str],
    call_counter: list[int],
) -> None:
    """Patch benchmark_qed.llm.chat to return grades from grade_by_chunk."""

    async def _fake_chat(_llm: Any, messages: list[dict[str, str]], **_: Any) -> Any:  # noqa: RUF029
        call_counter[0] += 1
        chunk_text = messages[-1]["content"]
        return SimpleNamespace(content=grade_by_chunk.get(chunk_text, "no_support"))

    monkeypatch.setattr("benchmark_qed.llm.chat", _fake_chat)


async def _run(
    eval_results: list[dict[str, Any]],
    question_set: dict[str, Any],
    output_dir: Path,
    cache_path: Path,
    *,
    k_list: list[int] | None = None,
) -> dict[str, Any]:
    """Invoke run_assertion_eval_chunk_mode with test-friendly defaults."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return await run_assertion_eval_chunk_mode(
        load_retrieval_results_from_dicts(
            eval_results,
            context_id_key="chunk_id",
            context_text_key="text",
            question_text_key="text",
        ),
        question_set,
        llm_client=cast("LLMCompletion", object()),
        llm_config=cast(
            "LLMConfig", SimpleNamespace(concurrent_requests=2, call_args={})
        ),
        output_storage=FileStorage(base_dir=str(output_dir)),
        pass_threshold=0.5,
        cache_path=cache_path,
        k_list=k_list or [1],
        system_prompt="judge",
        user_prompt="{chunk}",
    )


@pytest.mark.usefixtures("patched_chat")
async def test_at_k_truncation_respects_rank_order(
    tmp_path: Path, grade_by_chunk: dict[str, str]
) -> None:
    """@k uses the top-ranked chunks even when the JSON list is unordered."""
    grade_by_chunk.update({
        "rank1": "no_support",
        "rank2": "no_support",
        "rank3": "full_support",
    })
    eval_results = [
        {
            "question_id": "q1",
            "text": "Q",
            "context": [
                {"text": "rank3", "chunk_id": "c3", "rank": 3},
                {"text": "rank1", "chunk_id": "c1", "rank": 1},
                {"text": "rank2", "chunk_id": "c2", "rank": 2},
            ],
        }
    ]
    question_set = {
        "assertions": [
            {
                "question_id": "q1",
                "question_text": "Q",
                "assertions": [{"statement": "S"}],
            }
        ]
    }

    summaries = await _run(
        eval_results,
        question_set,
        tmp_path / "out",
        tmp_path / "cache.jsonl",
        k_list=[1],
    )

    # Across all chunks the best is the rank-3 full_support chunk.
    assert summaries["all"].coverage == 1.0
    assert summaries["all"].mean_retrieved_chunks == 3.0
    # At k=1 only the top-ranked (rank 1) chunk is considered: no_support.
    assert summaries["k1"].coverage == 0.0
    assert summaries["k1"].mean_retrieved_chunks == 1.0


@pytest.mark.usefixtures("patched_chat")
async def test_alignment_by_question_id(
    tmp_path: Path, grade_by_chunk: dict[str, str]
) -> None:
    """Chunks are matched to assertions by question_id, not list position."""
    grade_by_chunk.update({"good": "full_support", "bad": "no_support"})
    # eval_results are in the reverse order of the assertions list.
    eval_results = [
        {
            "question_id": "q2",
            "text": "Q2",
            "context": [{"text": "bad", "chunk_id": "c2", "rank": 1}],
        },
        {
            "question_id": "q1",
            "text": "Q1",
            "context": [{"text": "good", "chunk_id": "c1", "rank": 1}],
        },
    ]
    question_set = {
        "assertions": [
            {
                "question_id": "q1",
                "question_text": "Q1",
                "assertions": [{"statement": "S1"}],
            },
            {
                "question_id": "q2",
                "question_text": "Q2",
                "assertions": [{"statement": "S2"}],
            },
        ]
    }

    summaries = await _run(
        eval_results, question_set, tmp_path / "out", tmp_path / "cache.jsonl"
    )

    per_query = summaries["all"].per_query_metrics
    # q1 (index 0) got the "good" chunk -> covered; q2 (index 1) got "bad".
    assert per_query["0000"]["coverage"] == 1.0
    assert per_query["0001"]["coverage"] == 0.0


@pytest.mark.usefixtures("patched_chat")
async def test_second_run_uses_cache(
    tmp_path: Path, grade_by_chunk: dict[str, str], call_counter: list[int]
) -> None:
    """A second run with the same cache makes no new LLM calls."""
    grade_by_chunk.update({"a": "full_support", "b": "no_support"})
    eval_results = [
        {
            "question_id": "q1",
            "text": "Q",
            "context": [
                {"text": "a", "chunk_id": "c1", "rank": 1},
                {"text": "b", "chunk_id": "c2", "rank": 2},
            ],
        }
    ]
    question_set = {
        "assertions": [
            {
                "question_id": "q1",
                "question_text": "Q",
                "assertions": [{"statement": "S"}],
            }
        ]
    }
    cache_path = tmp_path / "cache.jsonl"

    await _run(eval_results, question_set, tmp_path / "out1", cache_path)
    calls_after_first = call_counter[0]
    assert calls_after_first == 2  # one call per (assertion, chunk) pair

    await _run(eval_results, question_set, tmp_path / "out2", cache_path)
    assert call_counter[0] == calls_after_first  # fully served from cache
