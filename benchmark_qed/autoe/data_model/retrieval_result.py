# Copyright (c) 2025 Microsoft Corporation.
"""Data model for retrieval results in retrieval scoring."""

from typing import Any, Sequence

from pydantic import BaseModel, Field, model_validator
from benchmark_qed.autod.data_model.text_unit import TextUnit


class RetrievalResult(BaseModel):
    """Container for query and context data for relevance assessment."""

    question_id: str = Field(description="Unique identifier for the question.")
    question_text: str = Field(description="The text of the question to assess relevance against.")
    context: Sequence[TextUnit | dict[str, str]] = Field(description="List of text units or dictionaries representing the context.")
    context_id_key: str = Field(default="source_id", description="Key name for the ID field in dictionary context items.")
    context_text_key: str = Field(default="source_text", description="Key name for the text field in dictionary context items.")

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_retrieval_result(self) -> "RetrievalResult":
        """Validate the RetrievalResult after initialization."""
        if not self.question_id:
            raise ValueError("question_id cannot be empty")
        if not self.question_text:
            raise ValueError("question_text cannot be empty")
        if not isinstance(self.context, list):
            raise ValueError("context must be a list")
        
        # Validate context items
        for i, item in enumerate(self.context):
            if not isinstance(item, TextUnit) and not (
                isinstance(item, dict) and self.context_id_key in item and self.context_text_key in item
            ):
                raise ValueError(
                    f"Context item {i} must be TextUnit or dict with '{self.context_id_key}' and '{self.context_text_key}' keys, "
                    f"got: {type(item)}"
                )
        return self

    @property
    def context_size(self) -> int:
        """Get the number of items in the context."""
        return len(self.context)

    def get_context_item_id(self, item: TextUnit | dict[str, str]) -> str:
        """Extract the ID from a context item."""
        if isinstance(item, TextUnit):
            return item.id
        return item[self.context_id_key]

    def get_context_item_text(self, item: TextUnit | dict[str, str]) -> str:
        """Extract the text from a context item."""
        if isinstance(item, TextUnit):
            return item.text
        return item[self.context_text_key]

    @classmethod
    def from_question(
        cls,
        question_id: str,
        question_text: str,
        context: list[TextUnit | dict[str, str]],
        context_id_key: str = "id",
        context_text_key: str = "text",
    ) -> "RetrievalResult":
        """Create RetrievalResult from question parameters."""
        return cls(
            question_id=question_id,
            question_text=question_text,
            context=context,
            context_id_key=context_id_key,
            context_text_key=context_text_key,
        )


def load_retrieval_results_from_dicts(
    data: list[dict[str, Any]], 
    context_id_key: str = "source_id", 
    context_text_key: str = "source_text",
    question_id_key: str = "question_id",
    question_text_key: str = "question_text", 
    context_key: str = "context",
    auto_transform_context: bool = False
) -> list[RetrievalResult]:
    """
    Load a list of RetrievalResult objects from a list of dictionaries.
    
    Args:
        data: List of dictionaries containing query context data.
        context_id_key: Key name for the ID field in dictionary context items.
        context_text_key: Key name for the text field in dictionary context items.
        question_id_key: Key name for the question ID field in input dictionaries.
        question_text_key: Key name for the question text field in input dictionaries.
        context_key: Key name for the context list field in input dictionaries.
        auto_transform_context: If True, automatically convert dict context items to TextUnit objects.
    
    Returns:
        List of RetrievalResult objects created from the input dictionaries.
        
    Raises:
        ValueError: If any dictionary is missing required keys or has invalid values.
        KeyError: If required keys are missing from the dictionaries.
        
    Example:
        >>> data = [
        ...     {
        ...         "query_id": "q1",
        ...         "query_text": "What is renewable energy?",
        ...         "documents": [{"doc_id": "doc1", "content": "Solar energy is clean..."}]
        ...     }
        ... ]
        >>> # Using custom keys
        >>> retrieval_results = load_retrieval_results_from_dicts(
        ...     data, 
        ...     context_id_key="doc_id", 
        ...     context_text_key="content",
        ...     question_id_key="query_id",
        ...     question_text_key="query_text",
        ...     context_key="documents"
        ... )
        >>> len(retrieval_results)
        1
    """
    if not isinstance(data, list):
        raise ValueError("data must be a list of dictionaries")
    
    retrieval_results = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be a dictionary, got: {type(item)}")
        
        try:
            context = item[context_key]
            
            # Auto-transform dictionary context items to TextUnit objects if requested
            if auto_transform_context and isinstance(context, list):
                transformed_context = []
                for j, ctx_item in enumerate(context):
                    if isinstance(ctx_item, dict):
                        # Create TextUnit from dictionary
                        text_unit = TextUnit(
                            id=ctx_item[context_id_key],
                            short_id=ctx_item.get("short_id", str(ctx_item[context_id_key])[:8]),
                            text=ctx_item[context_text_key]
                        )
                        transformed_context.append(text_unit)
                    else:
                        # Keep existing TextUnit objects as-is
                        transformed_context.append(ctx_item)
                context = transformed_context
            
            retrieval_result = RetrievalResult(
                question_id=item[question_id_key],
                question_text=item[question_text_key],
                context=context,
                context_id_key=context_id_key,
                context_text_key=context_text_key,
            )
            retrieval_results.append(retrieval_result)
        except KeyError as e:
            raise KeyError(f"Item {i} missing required key: {e}") from e
        except ValueError as e:
            raise ValueError(f"Item {i} has invalid value: {e}") from e
    
    return retrieval_results
