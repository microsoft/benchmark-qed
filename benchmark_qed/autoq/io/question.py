# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to save/load questions."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from graphrag_storage import Storage

from benchmark_qed.autoq.data_model.question import Question

log: logging.Logger = logging.getLogger(__name__)


def _normalize_assertion(assertion: dict[str, Any] | str) -> dict[str, Any]:
    """
    Normalize an assertion to a dictionary with consistent structure.

    Assertions can be:
    - dict: Standard format with statement, sources, score, etc.
    - str: Legacy format (plain text assertion statement)

    Args:
        assertion: An assertion dictionary or string

    Returns
    -------
        A normalized dictionary with statement, sources, score, reasoning, attributes keys

    """
    if isinstance(assertion, str):
        # Legacy format: plain string assertion
        return {
            "statement": assertion,
            "sources": [],
            "score": 0,
            "reasoning": "",
            "attributes": {},
        }
    # Standard dict format
    return {
        "statement": assertion.get("statement", ""),
        "sources": assertion.get("sources") or [],
        "score": assertion.get("score", 0),
        "reasoning": assertion.get("reasoning", ""),
        "attributes": assertion.get("attributes") or {},
    }


async def _save_assertions(questions: list[Question], storage: Storage) -> None:
    """
    Extract and save assertions from questions to separate JSON files with ranks.

    Creates the following files:
    - assertions.json: Final assertions with ranks and supporting_assertions metadata
    - assertion_sources.json: Source text chunks for each assertion (kept separate to reduce file size)
    - map_assertions.json: Intermediate assertions from map step (for global assertions)
    - map_assertion_sources.json: Source text chunks for map assertions

    For global assertions, each assertion includes:
    - supporting_assertions: The child/local assertions that were consolidated into it
    - This enables comprehensive coverage scoring beyond binary pass/fail

    Args:
        questions: List of Question objects with assertions in attributes
        storage: Storage backend to write the JSON files to
    """
    questions_with_assertions = []
    questions_with_map_assertions = []
    assertion_sources_data = []
    map_assertion_sources_data = []

    for question in questions:
        # Check if question has assertions in its attributes
        if not question.attributes:
            continue

        assertions = question.attributes.get("assertions", [])
        if not assertions:
            continue

        # Normalize all assertions to dictionaries
        assertion_dicts = [_normalize_assertion(a) for a in assertions]

        # Sort assertions by score (descending) then by source count (descending) to determine ranks
        sorted_assertions = sorted(
            assertion_dicts,
            key=lambda a: (-a.get("score", 0), -len(a.get("sources", []))),
        )

        # Add rank information (rank 1 = highest importance)
        ranked_assertions = []
        question_assertion_sources = []
        for rank, assertion in enumerate(sorted_assertions, 1):
            sources = assertion.get("sources") or []
            attributes = assertion.get("attributes") or {}

            ranked_assertion = {
                "statement": assertion["statement"],
                "source_count": len(sources),
                "score": assertion.get("score", 0),
                "reasoning": assertion.get("reasoning", ""),
                "rank": rank,
            }

            # Include validation scores if available
            if attributes and "validation" in attributes:
                ranked_assertion["validation"] = attributes["validation"]

            # Include supporting_assertions if available (for global assertions)
            # These are the child/local assertions that were consolidated
            if attributes and "supporting_assertions" in attributes:
                ranked_assertion["supporting_assertions"] = attributes[
                    "supporting_assertions"
                ]

            ranked_assertions.append(ranked_assertion)

            # Collect sources for separate file
            if sources:
                question_assertion_sources.append({
                    "rank": rank,
                    "statement": assertion["statement"],
                    "sources": sources,
                })

        questions_with_assertions.append({
            "question_id": question.id,
            "question_text": question.text,
            "assertions": ranked_assertions,
            "claims": question.attributes.get("claims", []),  # Include claims for stats
        })

        # Add sources data if any assertions have sources
        if question_assertion_sources:
            assertion_sources_data.append({
                "question_id": question.id,
                "question_text": question.text,
                "assertion_sources": question_assertion_sources,
            })

        # Process map_assertions if available (for global assertions)
        map_assertions = question.attributes.get("map_assertions", [])
        if map_assertions:
            map_assertion_dicts = [_normalize_assertion(a) for a in map_assertions]

            # Sort and rank map assertions
            sorted_map_assertions = sorted(
                map_assertion_dicts,
                key=lambda a: (-a.get("score", 0), -len(a.get("sources", []))),
            )

            ranked_map_assertions = []
            question_map_assertion_sources = []
            for rank, assertion in enumerate(sorted_map_assertions, 1):
                sources = assertion.get("sources") or []
                ranked_map_assertions.append({
                    "statement": assertion["statement"],
                    "source_count": len(sources),
                    "score": assertion.get("score", 0),
                    "reasoning": assertion.get("reasoning", ""),
                    "rank": rank,
                })

                # Collect sources for separate file
                if sources:
                    question_map_assertion_sources.append({
                        "rank": rank,
                        "statement": assertion["statement"],
                        "sources": sources,
                    })

            questions_with_map_assertions.append({
                "question_id": question.id,
                "question_text": question.text,
                "map_assertions": ranked_map_assertions,
            })

            # Add map assertion sources if any
            if question_map_assertion_sources:
                map_assertion_sources_data.append({
                    "question_id": question.id,
                    "question_text": question.text,
                    "map_assertion_sources": question_map_assertion_sources,
                })

    # Save assertions to file as a direct list
    if questions_with_assertions:
        await storage.set(
            "assertions.json",
            json.dumps(questions_with_assertions, indent=4),
        )

    # Save assertion sources to separate file
    if assertion_sources_data:
        await storage.set(
            "assertion_sources.json",
            json.dumps(assertion_sources_data, indent=4),
        )
        log.info(
            "Saved assertion sources for %d questions",
            len(assertion_sources_data),
        )

    # Save map_assertions to separate file (for global assertions)
    if questions_with_map_assertions:
        await storage.set(
            "map_assertions.json",
            json.dumps(questions_with_map_assertions, indent=4),
        )
        log.info(
            "Saved map (source) assertions for %d questions",
            len(questions_with_map_assertions),
        )

    # Save map assertion sources to separate file
    if map_assertion_sources_data:
        await storage.set(
            "map_assertion_sources.json",
            json.dumps(map_assertion_sources_data, indent=4),
        )
        log.info(
            "Saved map assertion sources for %d questions",
            len(map_assertion_sources_data),
        )

    # Generate and save assertion statistics
    import tempfile

    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.stats import (
        compute_assertion_stats,
        save_stats_to_file,
    )

    if questions_with_assertions:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        stats = compute_assertion_stats(
            assertions_data=questions_with_assertions,
            assertion_type="global",
            file_path="assertions.json",
            sources_data=assertion_sources_data or None,
        )
        save_stats_to_file(stats, tmp_path)
        await storage.set("assertions_stats.json", tmp_path.read_text(encoding="utf-8"))
        tmp_path.unlink(missing_ok=True)
        log.info(
            "Generated assertion statistics: %d questions, %d assertions",
            stats.total_questions,
            stats.total_assertions,
        )

    if questions_with_map_assertions:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        map_stats = compute_assertion_stats(
            assertions_data=questions_with_map_assertions,
            assertion_type="map",
            file_path="map_assertions.json",
            sources_data=map_assertion_sources_data or None,
        )
        save_stats_to_file(map_stats, tmp_path)
        await storage.set(
            "map_assertions_stats.json", tmp_path.read_text(encoding="utf-8")
        )
        tmp_path.unlink(missing_ok=True)
        log.info(
            "Generated map assertion statistics: %d questions, %d assertions",
            map_stats.total_questions,
            map_stats.total_assertions,
        )


async def load_questions(
    storage: Storage, file_name: str, question_text_only: bool = False
) -> list[Question]:
    """Read question list from a json file via storage backend."""
    data = await storage.get(file_name)
    question_list = json.loads(data)
    if question_text_only:
        return [Question(id=str(uuid4()), text=question) for question in question_list]
    questions = []
    for question in question_list:
        if isinstance(question, str):
            questions.append(Question(id=str(uuid4()), text=question))
        else:
            questions.append(Question(**question))
    return questions


async def save_questions(
    questions: list[Question],
    storage: Storage,
    output_name: str,
    question_text_only: bool = False,
    include_embedding: bool = False,
    save_assertions: bool = True,
) -> None:
    """Save question list to a json file via storage backend."""
    if question_text_only:
        question_list = [question.text for question in questions]
    else:
        question_list = [asdict(question) for question in questions]
        if not include_embedding:
            for question in question_list:
                question.pop("embedding", None)

    await storage.set(f"{output_name}.json", json.dumps(question_list, indent=4))

    # Save assertions separately if requested
    if save_assertions:
        await _save_assertions(questions, storage)
