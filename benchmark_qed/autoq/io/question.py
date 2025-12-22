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
    """Extract and save assertions from questions to a separate JSON file with ranks."""
    questions_with_assertions = []

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
        for rank, assertion in enumerate(sorted_assertions, 1):
            sources = assertion.get("sources") or []
            ranked_assertion = {
                "statement": assertion["statement"],
                "source_count": len(sources),
                "score": assertion.get("score", 0),
                "reasoning": assertion.get("reasoning", ""),
                "rank": rank,
            }

            # Include validation scores if available
            attributes = assertion.get("attributes") or {}
            if attributes and "validation" in attributes:
                ranked_assertion["validation"] = attributes["validation"]

            ranked_assertions.append(ranked_assertion)

        questions_with_assertions.append({
            "question_id": question.id,
            "question_text": question.text,
            "assertions": ranked_assertions,
        })

    # Save assertions to file as a direct list
    if questions_with_assertions:
        assertions_file = output_path / "assertions.json"
        Path(assertions_file).write_text(
            json.dumps(questions_with_assertions, indent=4)
        )


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
