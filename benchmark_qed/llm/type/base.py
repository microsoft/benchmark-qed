# Copyright (c) 2025 Microsoft Corporation.
"""Base classes and Protocols for Language Models (LLMs)."""

from typing import Any, Protocol

from pydantic import BaseModel, Field, PrivateAttr, computed_field


class Usage(BaseModel):
    """Base class for usage metrics."""

    model: str = Field(..., description="The name of the model.")

    _prompt_tokens: list[int] = PrivateAttr(default_factory=list)
    _completion_tokens: list[int] = PrivateAttr(default_factory=list)
    _prompt_cached_tokens: list[int] = PrivateAttr(default_factory=list)
    _completion_reasoning_tokens: list[int] = PrivateAttr(default_factory=list)
    _accepted_prediction_tokens: list[int] = PrivateAttr(default_factory=list)
    _rejected_prediction_tokens: list[int] = PrivateAttr(default_factory=list)

    def add_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        completion_reasoning_tokens: int = 0,
        accepted_prediction_tokens: int = 0,
        rejected_prediction_tokens: int = 0,
    ) -> None:
        """
        Add usage metrics to the instance.

        Args:
            prompt_tokens: The number of prompt tokens used.
            completion_tokens: The number of completion tokens used.
            cached_tokens: The number of cached tokens used.
            completion_reasoning_tokens: The number of reasoning tokens used in the completion.
            accepted_prediction_tokens: The number of accepted prediction tokens.
            rejected_prediction_tokens: The number of rejected prediction tokens.
        """
        self._prompt_tokens.append(prompt_tokens)
        self._completion_tokens.append(completion_tokens)
        self._prompt_cached_tokens.append(cached_tokens)
        self._completion_reasoning_tokens.append(completion_reasoning_tokens)
        self._accepted_prediction_tokens.append(accepted_prediction_tokens)
        self._rejected_prediction_tokens.append(rejected_prediction_tokens)

    @computed_field
    @property
    def prompt_tokens(self) -> int:
        """
        Get the total number of prompt tokens used.

        Returns
        -------
            The total number of prompt tokens used.
        """
        return sum(self._prompt_tokens)

    @computed_field
    @property
    def completion_tokens(self) -> int:
        """
        Get the total number of completion tokens used.

        Returns
        -------
            The total number of completion tokens used.
        """
        return sum(self._completion_tokens)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """
        Get the total number of tokens used.

        Returns
        -------
            The total number of tokens used.
        """
        return sum(self._prompt_tokens) + sum(self._completion_tokens)

    @computed_field
    @property
    def prompt_cached_tokens(self) -> int:
        """
        Get the total number of cached prompt tokens used.

        Returns
        -------
            The total number of cached prompt tokens used.
        """
        return sum(self._prompt_cached_tokens)

    @computed_field
    @property
    def completion_reasoning_tokens(self) -> int:
        """
        Get the total number of reasoning tokens used in the completion.

        Returns
        -------
            The total number of reasoning tokens used in the completion.
        """
        return sum(self._completion_reasoning_tokens)

    @computed_field
    @property
    def accepted_prediction_tokens(self) -> int:
        """
        Get the total number of accepted prediction tokens.

        Returns
        -------
            The total number of accepted prediction tokens.
        """
        return sum(self._accepted_prediction_tokens)

    @computed_field
    @property
    def rejected_prediction_tokens(self) -> int:
        """
        Get the total number of rejected prediction tokens.

        Returns
        -------
            The total number of rejected prediction tokens.
        """
        return sum(self._rejected_prediction_tokens)

    @computed_field
    @property
    def total_calls(self) -> int:
        """
        Get the total number of calls made.

        Returns
        -------
            The total number of calls made.
        """
        return len(self._prompt_tokens)


class BaseModelOutput(BaseModel):
    """Base class for LLM output."""

    content: str = Field(..., description="The textual content of the output.")
    """The textual content of the output."""


class BaseModelResponse(BaseModel):
    """Base class for a Model response."""

    output: BaseModelOutput
    """"""
    history: list[Any] = Field(default_factory=list)
    """History of the response."""
    usage: dict[str, Any] = Field(default_factory=dict)
    """Request/response metrics."""


class ChatModel(Protocol):
    """
    Protocol for a chat-based Language Model (LM).

    This protocol defines the methods required for a chat-based LM.
    Prompt is always required for the chat method, and any other keyword arguments are forwarded to the Model provider.
    """

    def get_usage(self) -> dict[str, Any]:
        """
        Get the usage metrics for the model.

        Returns
        -------
            A dictionary containing the usage metrics.
        """
        ...

    async def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> BaseModelResponse:
        """
        Generate a response for the given messages.

        Args:
            messages: The messages to generate a response for.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns
        -------
            A string representing the response.
        """
        ...


class EmbeddingModel(Protocol):
    """
    Protocol for an embedding-based Language Model (LM).

    This protocol defines the methods required for an embedding-based LM.
    """

    def get_usage(self) -> dict[str, Any]:
        """
        Get the usage metrics for the model.

        Returns
        -------
            A dictionary containing the usage metrics.
        """
        ...

    async def embed(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """
        Generate an embedding vector for the given list of strings.

        Args:
            text: The text to generate an embedding for.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns
        -------
            A collections of list of floats representing the embedding vector for each item in the batch.
        """
        ...
