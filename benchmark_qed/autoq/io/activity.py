# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to save/load activities."""

import json
from pathlib import Path

from benchmark_qed.autoq.data_model.activity import (
    EXCLUDE_ENTITIES,
    ActivityContext,
    Entity,
    TaskContext,
)


def save_activity_context(
    activity_context: ActivityContext,
    output_path: str,
    output_name: str = "activity_context",
) -> None:
    """Save the activity context to a JSON file."""
    output_path_obj = Path(output_path)
    if not output_path_obj.exists():
        output_path_obj.mkdir(parents=True, exist_ok=True)
    json_output_file = output_path_obj / f"{output_name}.json"
    Path(json_output_file).write_text(
        json.dumps(activity_context.model_dump(exclude=EXCLUDE_ENTITIES), indent=2)
    )
    json_embeddings_file = output_path_obj / f"{output_name}_full.json"
    Path(json_embeddings_file).write_text(json.dumps(activity_context.model_dump()))


def load_activity_context(file_path: str) -> ActivityContext:
    """Load the activity context from a JSON file."""
    data = json.loads(Path(file_path).read_text())
    dataset_description = data.get("dataset_description", "")
    task_contexts_json = data.get("task_contexts", [])
    task_contexts = [
        TaskContext(
            persona=task_context.get("persona", ""),
            task=task_context.get("task", ""),
            entities=[
                Entity(
                    name=entity.get("name", ""),
                    description=entity.get("description", ""),
                    relevance_score=entity.get("relevance_score", 50),
                    embedding=entity.get("embedding", None),
                )
                for entity in task_context.get("entities", [])
            ],
        )
        for task_context in task_contexts_json
    ]
    return ActivityContext(
        dataset_description=dataset_description, task_contexts=task_contexts
    )
