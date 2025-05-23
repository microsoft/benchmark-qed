"""Util functions to save/load activities."""

import json
import pickle
from pathlib import Path

from benchmark_qed.autoq.data_model.activity import ActivityContext, Entity, TaskContext


def save_activity_context(
    activity_context: ActivityContext,
    output_path: str,
    output_name: str = "activity_context",
) -> None:
    """Save the activity context to a JSON file."""
    output_path_obj = Path(output_path)
    if not output_path_obj.exists():
        output_path_obj.mkdir(parents=True, exist_ok=True)
    output_file = output_path_obj / f"{output_name}.json"
    with open(output_file, "w") as f:
        json.dump(activity_context.to_dict(include_entity_embedding=False), f)
    pickle_file = output_path_obj / f"{output_name}.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(activity_context, f)


def load_activity_context(file_path: str) -> ActivityContext:
    """Load the activity context from a JSON file."""
    with open(file_path) as f:
        data = json.load(f)
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
