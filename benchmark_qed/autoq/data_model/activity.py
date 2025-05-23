from dataclasses import dataclass
from typing import Any

"""
A package containing the ActivityContext model. 
Each Activity is associated with a set of personas. 
Each persona has a set of tasks. Each task has a set of entities. 
The ActivityContext is used to store the context for all activity questions.
"""


@dataclass
class Entity:
    """Data class for storing the context for a single entity associated with a task"""

    name: str
    description: str = ""
    relevance_score: int = 50
    embedding: list[float] | None = None

    def to_str(self) -> str:
        return f"{self.name}: ({self.description})"

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        entity_dict = {
            "name": self.name,
            "description": self.description,
            "relevance_score": self.relevance_score,
        }
        if include_embedding and self.embedding is not None:
            entity_dict["embedding"] = self.embedding
        return entity_dict


@dataclass
class TaskContext:
    """Data class for storing the context for a single task associated with a persona"""

    persona: str
    task: str
    entities: list[Entity]

    def to_dict(self, include_entity_embedding: bool = True) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "task": self.task,
            "entities": [
                entity.to_dict(include_entity_embedding) for entity in self.entities
            ],
        }


@dataclass
class ActivityContext:
    """Data class for storing the context for all activity questions"""

    dataset_description: str
    task_contexts: list[TaskContext]

    def to_dict(self, include_entity_embedding: bool = True) -> dict[str, Any]:
        return {
            "dataset_description": self.dataset_description,
            "task_contexts": [
                task_context.to_dict(include_entity_embedding)
                for task_context in self.task_contexts
            ],
        }

    def get_all_entities(self) -> list[Entity]:
        all_entities = [
            entity
            for task_context in self.task_contexts
            if task_context.entities
            for entity in task_context.entities
        ]

        # remove duplicates
        cleaned_entity_names: set[str] = set()
        final_entities: list[Entity] = []
        for entity in all_entities:
            if entity.name not in cleaned_entity_names:
                cleaned_entity_names.add(entity.name)
                final_entities.append(entity)
        return final_entities
