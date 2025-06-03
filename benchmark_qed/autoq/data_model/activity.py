# Copyright (c) 2025 Microsoft Corporation.
"""Define activity model, including personas, associated tasks, and relevant entities."""

from pydantic import BaseModel

"""
A package containing the ActivityContext model.
Each Activity is associated with a set of personas.
Each persona has a set of tasks. Each task has a set of entities.
The ActivityContext is used to store the context for all activity questions.
"""

EXCLUDE_ENTITIES = {
    "task_contexts": {"__all__": {"entities": {"__all__": {"embedding"}}}}
}


class Entity(BaseModel):
    """Data class for storing the context for a single entity associated with a task."""

    name: str
    description: str = ""
    relevance_score: int = 50
    embedding: list[float] | None = None

    def to_str(self) -> str:
        """Return a string representation of the entity."""
        return f"{self.name}: ({self.description})"


class TaskContext(BaseModel):
    """Data class for storing the context for a single task associated with a persona."""

    persona: str
    task: str
    entities: list[Entity]


class ActivityContext(BaseModel):
    """Data class for storing the context for all activity questions."""

    dataset_description: str
    task_contexts: list[TaskContext]

    def get_all_entities(self) -> list[Entity]:
        """Return a list of all entities across all task contexts, removing duplicates."""
        seen = set()
        entities = []
        for ctx in self.task_contexts:
            for entity in ctx.entities or []:
                if entity.name not in seen:
                    seen.add(entity.name)
                    entities.append(entity)
        return entities
