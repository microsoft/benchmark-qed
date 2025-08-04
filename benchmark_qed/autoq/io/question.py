# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to save/load questions."""

import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from benchmark_qed.autoq.data_model.question import Question


def _save_assertions(questions: list[Question], output_path: Path) -> None:
    """Extract and save assertions from questions to a separate JSON file."""
    questions_with_assertions = []
    
    for question in questions:
        # Check if question has assertions in its attributes
        if hasattr(question, 'attributes') and question.attributes:
            assertions = question.attributes.get('assertions', [])
            if assertions:
                # Extract just the assertion statements as per the plan
                assertion_statements = []
                for assertion in assertions:
                    if isinstance(assertion, dict) and 'statement' in assertion:
                        assertion_statements.append(assertion['statement'])
                    elif isinstance(assertion, str):
                        assertion_statements.append(assertion)
                
                if assertion_statements:
                    questions_with_assertions.append({
                        "question_id": question.id,
                        "question_text": question.text,
                        "assertions": assertion_statements
                    })
    
    # Save assertions to file as a direct list
    if questions_with_assertions:
        assertions_file = output_path / "assertions.json"
        Path(assertions_file).write_text(json.dumps(questions_with_assertions, indent=4))


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
