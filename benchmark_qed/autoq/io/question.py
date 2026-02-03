# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to save/load questions."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

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


def _save_assertions(questions: list[Question], output_path: Path) -> None:
    """
    Extract and save assertions from questions to separate JSON files with ranks.

    Creates the following files:
    - assertions.json: Final assertions with ranks and supporting_assertions metadata
    - assertion_sources.json: Source text chunks for each assertion (kept separate to reduce file size)
    - map_assertions.json: Intermediate assertions from map step (for global questions)
    - map_assertion_sources.json: Source text chunks for map assertions

    For global assertions, each assertion includes:
    - supporting_assertions: The child/local assertions that were consolidated into it
    - This enables comprehensive coverage scoring beyond binary pass/fail

    Args:
        questions: List of Question objects with assertions in attributes
        output_path: Directory path to save the JSON files
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
                ranked_assertion["supporting_assertions"] = attributes["supporting_assertions"]

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
        assertions_file = output_path / "assertions.json"
        Path(assertions_file).write_text(
            json.dumps(questions_with_assertions, indent=4)
        )

    # Save assertion sources to separate file
    if assertion_sources_data:
        assertion_sources_file = output_path / "assertion_sources.json"
        Path(assertion_sources_file).write_text(
            json.dumps(assertion_sources_data, indent=4)
        )
        log.info(
            "Saved assertion sources for %d questions to %s",
            len(assertion_sources_data),
            assertion_sources_file,
        )

    # Save map_assertions to separate file (for global assertions)
    if questions_with_map_assertions:
        map_assertions_file = output_path / "map_assertions.json"
        Path(map_assertions_file).write_text(
            json.dumps(questions_with_map_assertions, indent=4)
        )
        log.info(
            "Saved map (source) assertions for %d questions to %s",
            len(questions_with_map_assertions),
            map_assertions_file,
        )

    # Save map assertion sources to separate file
    if map_assertion_sources_data:
        map_assertion_sources_file = output_path / "map_assertion_sources.json"
        Path(map_assertion_sources_file).write_text(
            json.dumps(map_assertion_sources_data, indent=4)
        )
        log.info(
            "Saved map assertion sources for %d questions to %s",
            len(map_assertion_sources_data),
            map_assertion_sources_file,
        )

    # Generate and save assertion statistics
    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.stats import (
        compute_assertion_stats,
        save_stats_to_file,
    )

    if questions_with_assertions:
        stats = compute_assertion_stats(
            assertions_data=questions_with_assertions,
            assertion_type="global",
            file_path=str(output_path / "assertions.json"),
            sources_data=assertion_sources_data if assertion_sources_data else None,
        )
        save_stats_to_file(stats, output_path / "assertions_stats.json")
        log.info("Generated assertion statistics: %d questions, %d assertions",
                 stats.total_questions, stats.total_assertions)

    if questions_with_map_assertions:
        map_stats = compute_assertion_stats(
            assertions_data=questions_with_map_assertions,
            assertion_type="map",
            file_path=str(output_path / "map_assertions.json"),
            sources_data=map_assertion_sources_data if map_assertion_sources_data else None,
        )
        save_stats_to_file(map_stats, output_path / "map_assertions_stats.json")
        log.info("Generated map assertion statistics: %d questions, %d assertions",
                 map_stats.total_questions, map_stats.total_assertions)


def load_questions(file_path: str, question_text_only: bool = False) -> list[Question]:
    """Read question list from a json file."""
    question_list = json.loads(Path(file_path).read_text())
    if question_text_only:
        return [Question(id=str(uuid4()), text=question) for question in question_list]
    questions = []
    for question in question_list:
        if isinstance(question, str):
            questions.append(Question(id=str(uuid4()), text=question))
        else:
            questions.append(Question(**question))
    return questions


def save_questions(
    questions: list[Question],
    output_path: str,
    output_name: str,
    question_text_only: bool = False,
    include_embedding: bool = False,
    save_assertions: bool = True,
) -> None:
    """Save question list to a json file."""
    if question_text_only:
        question_list = [question.text for question in questions]
    else:
        question_list = [asdict(question) for question in questions]
        if not include_embedding:
            for question in question_list:
                question.pop("embedding", None)

    output_path_obj = Path(output_path)
    if not output_path_obj.exists():
        output_path_obj.mkdir(parents=True, exist_ok=True)
    output_file = output_path_obj / f"{output_name}.json"

    Path(output_file).write_text(json.dumps(question_list, indent=4))

    # Save assertions separately if requested
    if save_assertions:
        _save_assertions(questions, output_path_obj)
