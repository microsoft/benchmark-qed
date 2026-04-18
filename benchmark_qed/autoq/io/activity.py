# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to save/load activities."""

import json

from graphrag_storage import Storage

from benchmark_qed.autoq.data_model.activity import (
    EXCLUDE_ENTITIES,
    ActivityContext,
    Entity,
    TaskContext,
)


async def save_activity_context(
    activity_context: ActivityContext,
    storage: Storage,
    output_name: str = "activity_context",
) -> None:
    """Save the activity context to a JSON file via storage backend."""
    await storage.set(
        f"{output_name}.json",
        json.dumps(activity_context.model_dump(exclude=EXCLUDE_ENTITIES), indent=2),
    )
    await storage.set(
        f"{output_name}_full.json",
        json.dumps(activity_context.model_dump()),
    )


async def load_activity_context(storage: Storage, file_name: str) -> ActivityContext:
    """Load the activity context from a JSON file via storage backend."""
    data = json.loads(await storage.get(file_name))
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
