# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to save/load questions."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from benchmark_qed.autoq.data_model.question import Question

log = logging.getLogger(__name__)


def _save_assertions(questions: list[Question], output_path: Path) -> None:
    """Extract and save assertions from questions to a separate JSON file with ranks."""
    questions_with_assertions = []

    for question in questions:
        # Check if question has assertions in its attributes
        if hasattr(question, "attributes") and question.attributes:
            assertions = question.attributes.get("assertions", [])
            if assertions:
                # Convert assertions to a list of dictionaries if needed
                assertion_dicts = []
                for i, assertion in enumerate(assertions):
                    if isinstance(assertion, dict):
                        # Debug logging for dict assertions
                        sources = assertion.get("sources", [])
                        if sources is None:
                            log.warning(
                                "Question %s, assertion %s: Dict assertion has None sources",
                                question.id,
                                i,
                            )
                        elif not isinstance(sources, list | tuple):
                            log.warning(
                                "Question %s, assertion %s: Dict assertion has non-list sources: %s",
                                question.id,
                                i,
                                type(sources),
                            )
                        assertion_dicts.append(assertion)
                    elif hasattr(assertion, "__dict__"):  # Handle dataclass/object
                        sources = (
                            assertion.sources if hasattr(assertion, "sources") else []
                        )
                        if sources is None:
                            log.warning(
                                "Question %s, assertion %s: Object assertion has None sources",
                                question.id,
                                i,
                            )
                            sources = []
                        elif not isinstance(sources, list | tuple):
                            log.warning(
                                "Question %s, assertion %s: Object assertion has non-list sources: %s",
                                question.id,
                                i,
                                type(sources),
                            )
                            sources = [] if sources is None else [sources]

                        assertion_dict = {
                            "statement": assertion.statement
                            if hasattr(assertion, "statement")
                            else str(assertion),
                            "sources": sources,
                            "score": assertion.score
                            if hasattr(assertion, "score")
                            else 0,
                            "reasoning": assertion.reasoning
                            if hasattr(assertion, "reasoning")
                            else "",
                        }
                        log.debug(
                            "Question %s, assertion %s: Created dict with %s sources",
                            question.id,
                            i,
                            len(sources),
                        )
                        assertion_dicts.append(assertion_dict)
                    elif isinstance(assertion, str):
                        log.debug(
                            "Question %s, assertion %s: String assertion, 0 sources",
                            question.id,
                            i,
                        )
                        assertion_dicts.append({
                            "statement": assertion,
                            "sources": [],
                            "score": 0,
                            "reasoning": "",
                        })

                if assertion_dicts:
                    # Sort assertions by score (descending) then by source count (descending) to determine ranks
                    sorted_assertions = sorted(
                        assertion_dicts,
                        key=lambda a: (-a.get("score", 0), -len(a.get("sources", []))),
                    )

                    # Add rank information (rank 1 = highest importance)
                    ranked_assertions = []
                    for rank, assertion in enumerate(sorted_assertions, 1):
                        # Debug logging for source_count calculation
                        sources = assertion.get("sources", [])
                        if sources is None:
                            log.warning(
                                "Question %s: Assertion has None sources, setting to empty list",
                                question.id,
                            )
                            sources = []
                        elif not isinstance(sources, list | tuple):
                            log.warning(
                                "Question %s: Assertion has non-list sources: %s, sources: %s",
                                question.id,
                                type(sources),
                                sources,
                            )
                            sources = [] if sources is None else [sources]

                        source_count = len(sources)
                        if source_count == 0:
                            log.debug(
                                "Question %s: Assertion with 0 sources - Statement: '%s...'",
                                question.id,
                                assertion["statement"][:100],
                            )

                        ranked_assertion = {
                            "statement": assertion["statement"],
                            "source_count": source_count,
                            "score": assertion.get("score", 0),
                            "reasoning": assertion.get("reasoning", ""),
                            "rank": rank,
                        }

                        # Include validation scores if available
                        attributes = assertion.get("attributes", {})
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
