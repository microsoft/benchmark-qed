# Copyright (c) 2025 Microsoft Corporation.
"""Main chunk-level assertion evaluation module."""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import print as rich_print

from benchmark_qed.autoe.chunk_assertion.aggregation import summarize_at_k
from benchmark_qed.autoe.chunk_assertion.cache import (
    ContentAddressedCache,
    compute_cache_key,
)
from benchmark_qed.autoe.data_model.chunk_assertion import (
    ChunkAssertionGrade,
    EvalSummary,
    grade_to_score,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_storage import Storage

    from benchmark_qed.autoe.data_model.retrieval_result import RetrievalResult
    from benchmark_qed.config.llm_config import LLMConfig

log: logging.Logger = logging.getLogger(__name__)


def _collect_chunks(eval_result: RetrievalResult) -> list[dict[str, Any]]:
    """Flatten one question's retrieval into an ordered chunk list.

    Consumes a :class:`RetrievalResult`. When every context item carries an
    explicit ``rank`` the items are sorted by it (top-k semantics) rather than
    relying on list order. Duplicate (id, text) pairs and blank text are dropped.

    Args:
        eval_result: Retrieval result for a single question.

    Returns
    -------
        List of dicts with chunk_text and chunk_id
    """
    chunks: list[dict[str, Any]] = []
    seen: set[tuple[str | int | None, str]] = set()

    items = list(eval_result.context)
    ranks = [eval_result.get_context_item_rank(item) for item in items]
    if items and all(rank is not None for rank in ranks):
        items = [
            item
            for _rank, item in sorted(
                zip(ranks, items, strict=True),
                key=operator.itemgetter(0),  # type: ignore[arg-type, return-value]
            )
        ]

    for item in items:
        text = eval_result.get_context_item_text(item).strip()
        if not text:
            continue
        chunk_id = eval_result.get_context_item_id(item)
        dedupe_key = (chunk_id, text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        chunks.append({
            "chunk_text": text,
            "chunk_id": chunk_id,
        })

    return chunks


async def _label_chunk_assertion(
    llm_client: LLMCompletion,
    assertion_text: str,
    chunk_text: str,
    system_prompt: str,
    user_prompt: str,
    additional_call_args: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Judge a single (assertion, chunk) pair via LLM.

    Args:
        llm_client: LLM client instance
        assertion_text: Assertion statement
        chunk_text: Chunk/passage text
        system_prompt: System prompt template
        user_prompt: User prompt template
        additional_call_args: Extra arguments forwarded to the LLM call.

    Returns
    -------
        (grade, reasoning) tuple where grade is one of full_support/partial_support/no_support
    """
    from benchmark_qed.llm import chat

    try:
        # Format prompts with assertion and chunk
        formatted_system = system_prompt.format(
            assertion=assertion_text, chunk=chunk_text
        )
        formatted_user = user_prompt.format(assertion=assertion_text, chunk=chunk_text)

        response = await chat(
            llm_client,
            messages=[
                {"role": "system", "content": formatted_system},
                {"role": "user", "content": formatted_user},
            ],
            **(additional_call_args or {}),
        )

        # Parse response
        response_text = (response.content or "").strip()
    except Exception as e:  # noqa: BLE001 - any provider error maps to an error sentinel
        log.warning("LLM judging failed for an (assertion, chunk) pair: %s", e)
        return "__error__", str(e)

    # Extract grade from response (look for grade keywords)
    return _extract_grade(response_text), response_text


def _extract_grade(response_text: str) -> str:
    """Extract a grade from an LLM response.

    The prompt asks for a numeric score (1 / 0.5 / 0), which is the primary
    signal and is mapped to a grade. Free-form label parsing is kept as a
    fallback for robustness when a model ignores the format.

    Args:
        response_text: LLM response text

    Returns
    -------
        One of full_support, partial_support, no_support
    """
    response_lower = response_text.strip().lower()
    if not response_lower:
        return ChunkAssertionGrade.NO_SUPPORT

    # Primary signal: a numeric score of 1, 0.5, or 0. Check 0.5 first so the
    # "0" in "0.5" is not matched as a standalone zero.
    if re.search(r"(?<![\d.])0\.5(?![\d])", response_lower):
        return ChunkAssertionGrade.PARTIAL_SUPPORT
    if re.search(r"(?<![\d.])1(?:\.0+)?(?![\d.])", response_lower):
        return ChunkAssertionGrade.FULL_SUPPORT
    if re.search(r"(?<![\d.])0(?:\.0+)?(?![\d.])", response_lower):
        return ChunkAssertionGrade.NO_SUPPORT

    labels = (
        ChunkAssertionGrade.FULL_SUPPORT,
        ChunkAssertionGrade.PARTIAL_SUPPORT,
        ChunkAssertionGrade.NO_SUPPORT,
    )

    # Fallback: prefer an explicit label token.
    first_token = response_lower.split(None, 1)[0].strip(":-*. ")
    if first_token in labels:
        return first_token

    # Search for an explicit underscore label anywhere in the response.
    for label in labels:
        if label in response_lower:
            return label

    # Final fallback: handle space-separated phrasing. Check negatives and
    # partial before full so "does not support" / "partial support" aren't
    # misread.
    if (
        "no support" in response_lower
        or "not support" in response_lower
        or "unsupported" in response_lower
    ):
        return ChunkAssertionGrade.NO_SUPPORT
    if "partial" in response_lower and "support" in response_lower:
        return ChunkAssertionGrade.PARTIAL_SUPPORT
    if "full" in response_lower and "support" in response_lower:
        return ChunkAssertionGrade.FULL_SUPPORT

    return ChunkAssertionGrade.NO_SUPPORT


async def run_assertion_eval_chunk_mode(
    eval_results: list[RetrievalResult],
    question_set: dict[str, Any],
    *,
    llm_client: LLMCompletion,
    llm_config: LLMConfig,
    output_storage: Storage,
    pass_threshold: float = 0.5,
    cache_path: Path | None = None,
    k_list: list[int] | None = None,
    system_prompt: str = "",
    user_prompt: str = "",
    max_chunks_per_question: int | None = None,
) -> dict[str, EvalSummary]:
    """Judge retrieved chunks against per-question assertions and report @k.

    Evaluates each (assertion, chunk) pair via LLM, caches results by SHA256
    of (assertion_text + chunk_content), and aggregates metrics at each k.

    Args:
        eval_results: List of evaluation results (one per question)
        question_set: Question set dict with 'assertions' field
        llm_client: LLM client for judging
        llm_config: LLM configuration
        output_storage: Storage backend for writing debug records (local or blob)
        pass_threshold: Score threshold for pass (0.5 = partial+ is passing)
        cache_path: Path to persistent cache (created if not specified)
        k_list: List of k values to report (e.g., [5, 10, 20, 50])
        system_prompt: System prompt template
        user_prompt: User prompt template
        max_chunks_per_question: Cap on chunks evaluated per question (keeps the
            highest-ranked chunks). None means evaluate all retrieved chunks.

    Returns
    -------
        Dict of {label: EvalSummary} keyed by f"k{k}" plus "all" entry
    """
    if cache_path is None:
        cache_path = Path.cwd() / ".benchmark_qed_cache" / "chunk_assertions.jsonl"
    cache = ContentAddressedCache(cache_path)

    # Track per-(q, a) chunk grades: (question_idx, assertion_idx) -> [(rank, score, grade), ...]
    per_chunk_grades: dict[tuple[int, int], list[tuple[int, float, str]]] = {}
    assertion_call_stats: dict[tuple[int, int], dict[str, int]] = {}
    uncached_work: list[tuple[str, str, str, int, int, int]] = []
    retrieved_chunk_counts: list[int] = []

    total_checks = 0
    cache_hits = 0
    call_errors = 0

    # First pass: collect chunks and check cache
    assertions_list = question_set.get("assertions", [])

    # Prefer aligning eval_results by question_id to avoid silent mismatches when
    # the assertions file and chunks/answers file have different ordering. Fall
    # back to positional alignment (with a warning) when ids are missing or do
    # not line up, so datasets that rely on ordering still work.
    eval_by_question_id: dict[str, Any] = {}
    for row in eval_results:
        eval_by_question_id[str(row.question_id)] = row

    positional_fallbacks = 0

    for q_idx, q_row in enumerate(assertions_list):
        assertions = q_row.get("assertions", [])
        if not assertions:
            continue

        qid = q_row.get("question_id")
        eval_row = eval_by_question_id.get(str(qid)) if qid is not None else None
        if eval_row is None:
            # No id match: fall back to positional alignment.
            if qid is not None and eval_by_question_id:
                positional_fallbacks += 1
            eval_row = eval_results[q_idx] if q_idx < len(eval_results) else None
        chunks = _collect_chunks(eval_row) if eval_row is not None else []
        if max_chunks_per_question is not None and max_chunks_per_question > 0:
            chunks = chunks[:max_chunks_per_question]
        retrieved_chunk_counts.append(len(chunks))

        for a_idx, assertion in enumerate(assertions):
            assertion_text = (assertion.get("statement") or "").strip()
            if not assertion_text:
                continue
            q_key = (q_idx, a_idx)
            per_chunk_grades[q_key] = []
            assertion_call_stats[q_key] = {"successful": 0, "failed": 0}

            for rank, chunk in enumerate(chunks):
                chunk_content = str(chunk["chunk_text"])
                total_checks += 1
                cache_key = compute_cache_key(
                    assertion_text,
                    chunk_content,
                    model=llm_config.model,
                    call_args=llm_config.call_args,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                cached_grade = cache.get(cache_key)
                if cached_grade is not None:
                    cache_hits += 1
                    per_chunk_grades[q_key].append((
                        rank,
                        grade_to_score(cached_grade),
                        cached_grade,
                    ))
                    assertion_call_stats[q_key]["successful"] += 1
                else:
                    uncached_work.append((
                        assertion_text,
                        chunk_content,
                        cache_key,
                        q_idx,
                        a_idx,
                        rank,
                    ))

    if positional_fallbacks:
        rich_print(
            f"  [yellow]WARNING: {positional_fallbacks} question(s) had a "
            "question_id with no match in the chunks/answers file; falling back "
            "to positional alignment. Verify the two files refer to the same "
            "questions.[/yellow]"
        )

    if total_checks:
        hit_pct = 100 * cache_hits / total_checks
        rich_print(
            f"  Assertion-chunk cache: {total_checks} total checks, {cache_hits} hits ({hit_pct:.0f}%)"
        )
    else:
        rich_print("  No assertions or chunks to evaluate.")

    # Second pass: evaluate uncached pairs
    if uncached_work:
        rich_print(
            f"  Running {len(uncached_work)} LLM judgements for uncached (assertion, chunk) pairs...",
        )
        concurrent_requests = llm_config.concurrent_requests

        n_total = len(uncached_work)
        next_progress_threshold = max(1, n_total // 10)

        # Process work concurrently using a semaphore to limit parallelism
        semaphore = asyncio.Semaphore(max(1, concurrent_requests))

        async def _judge_one(
            work_item: tuple[str, str, str, int, int, int],
        ) -> tuple[tuple[str, str, str, int, int, int], str, str]:
            assertion_text, chunk_content, _cache_key, _q_idx, _a_idx, _rank = work_item
            async with semaphore:
                grade, reasoning = await _label_chunk_assertion(
                    llm_client,
                    assertion_text,
                    chunk_content,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    additional_call_args=llm_config.call_args,
                )
            return work_item, grade, reasoning

        tasks = [asyncio.ensure_future(_judge_one(w)) for w in uncached_work]
        for n_done, coro in enumerate(asyncio.as_completed(tasks), start=1):
            work_item, grade, _reasoning = await coro
            _assertion_text, _chunk_content, cache_key, q_idx, a_idx, rank = work_item

            q_key = (q_idx, a_idx)
            if grade == "__error__":
                call_errors += 1
                assertion_call_stats[q_key]["failed"] += 1
            else:
                cache.put(cache_key, grade)
                mapped_grade = (
                    grade
                    if grade
                    in {
                        ChunkAssertionGrade.FULL_SUPPORT,
                        ChunkAssertionGrade.PARTIAL_SUPPORT,
                        ChunkAssertionGrade.NO_SUPPORT,
                    }
                    else ChunkAssertionGrade.NO_SUPPORT
                )
                per_chunk_grades[q_key].append((
                    rank,
                    grade_to_score(grade),
                    mapped_grade,
                ))
                assertion_call_stats[q_key]["successful"] += 1

            if n_done >= next_progress_threshold:
                rich_print(
                    f"    LLM judging progress: {n_done}/{n_total} pairs ({(100.0 * n_done / n_total):.1f}%)",
                )
                next_progress_threshold += max(1, n_total // 10)

        # Flush cache
        cache.flush()
        rich_print(f"  Cached {len(uncached_work)} new entries to {cache_path.name}")
        if call_errors:
            rich_print(f"  WARNING: {call_errors} LLM calls failed")

    # Sort chunks by rank for @k truncation
    for q_key in per_chunk_grades:
        per_chunk_grades[q_key].sort(key=operator.itemgetter(0))

    total_assertions = len(per_chunk_grades)
    successful_calls = total_checks - call_errors

    # Write debug records (child storage creates the debug/ subdir for local FS
    # and prefixes keys for blob storage).
    debug_storage = output_storage.child("debug")
    for q_idx, q_row in enumerate(assertions_list):
        q_text = q_row.get("question_text", "")
        assertions = q_row.get("assertions", [])
        if not assertions:
            continue
        debug_record: dict[str, Any] = {
            "question_id": q_row.get("question_id", q_idx),
            "question_text": q_text,
            "assertions": [],
        }
        for a_idx, assertion in enumerate(assertions):
            q_key = (q_idx, a_idx)
            rows = per_chunk_grades.get(q_key, [])
            best_score = max((s for (_r, s, _g) in rows), default=None)
            best_grade = max(rows, key=operator.itemgetter(1))[2] if rows else None
            debug_record["assertions"].append({
                "statement": assertion.get("statement", ""),
                "score": float(best_score) if best_score is not None else None,
                "grade": best_grade if best_grade is not None else "unscored",
                "passed": (best_score is not None and best_score >= pass_threshold),
                "scored": best_score is not None,
            })
        await debug_storage.set(
            f"q_{q_idx:04d}.json", json.dumps(debug_record, indent=2)
        )

    if k_list is None:
        k_list = []
    k_list_sorted = sorted({int(k) for k in k_list if int(k) > 0})

    summaries: dict[str, EvalSummary] = {}
    for k in k_list_sorted:
        summaries[f"k{k}"] = summarize_at_k(
            k=k,
            question_set=question_set,
            per_chunk_grades=per_chunk_grades,
            retrieved_chunk_counts=retrieved_chunk_counts,
            n_assertions_total=total_assertions,
            pass_threshold=pass_threshold,
            total_calls=total_checks,
            successful_calls=successful_calls,
            failed_calls=call_errors,
        )
    summaries["all"] = summarize_at_k(
        k=None,
        question_set=question_set,
        per_chunk_grades=per_chunk_grades,
        retrieved_chunk_counts=retrieved_chunk_counts,
        n_assertions_total=total_assertions,
        pass_threshold=pass_threshold,
        total_calls=total_checks,
        successful_calls=successful_calls,
        failed_calls=call_errors,
    )
    return summaries
