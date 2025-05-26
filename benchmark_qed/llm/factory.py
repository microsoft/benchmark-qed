# Copyright (c) 2025 Microsoft Corporation.
"""A package containing a factory for supported llm types."""

from collections.abc import Callable
from typing import ClassVar

from benchmark_qed.config.llm_config import LLMConfig, LLMProvider
from benchmark_qed.llm.provider.azure import AzureInferenceChat, AzureInferenceEmbedding
from benchmark_qed.llm.provider.openai import (
    AzureOpenAIChat,
    AzureOpenAIEmbedding,
    OpenAIChat,
    OpenAIEmbedding,
)
from benchmark_qed.llm.type.base import ChatModel, EmbeddingModel


class ModelFactory:
    """A factory for creating Model instances."""

    _chat_registry: ClassVar[dict[str, Callable[..., ChatModel]]] = {}
    _embedding_registry: ClassVar[dict[str, Callable[..., EmbeddingModel]]] = {}

    @classmethod
    def register_chat(cls, model_type: str, creator: Callable[..., ChatModel]) -> None:
        """Register a ChatModel implementation."""
        cls._chat_registry[model_type] = creator

    @classmethod
    def register_embedding(
        cls, model_type: str, creator: Callable[..., EmbeddingModel]
    ) -> None:
        """Register an EmbeddingModel implementation."""
        cls._embedding_registry[model_type] = creator

    @classmethod
    def create_chat_model(cls, model_config: LLMConfig) -> ChatModel:
        """
        Create a ChatModel instance.

        Args:
            model_type: The type of ChatModel to create.
            **kwargs: Additional keyword arguments for the ChatModel constructor.

        Returns
        -------
            A ChatModel instance.
        """
        if model_config.llm_provider not in cls._chat_registry:
            msg = f"ChatMOdel implementation '{model_config.llm_provider}' is not registered."
            raise ValueError(msg)
        return cls._chat_registry[model_config.llm_provider](model_config)

    @classmethod
    def create_embedding_model(cls, model_config: LLMConfig) -> EmbeddingModel:
        """
        Create an EmbeddingModel instance.

        Args:
            model_type: The type of EmbeddingModel to create.
            **kwargs: Additional keyword arguments for the EmbeddingLLM constructor.

        Returns
        -------
            An EmbeddingLLM instance.
        """
        if model_config.llm_provider not in cls._embedding_registry:
            msg = f"EmbeddingModel implementation '{model_config.llm_provider}' is not registered."
            raise ValueError(msg)
        return cls._embedding_registry[model_config.llm_provider](model_config)


# --- Register default implementations ---
ModelFactory.register_chat(LLMProvider.OpenAIChat, lambda config: OpenAIChat(config))
ModelFactory.register_chat(
    LLMProvider.AzureOpenAIChat, lambda config: AzureOpenAIChat(config)
)
ModelFactory.register_chat(
    LLMProvider.AzureInferenceChat, lambda config: AzureInferenceChat(config)
)

ModelFactory.register_embedding(
    LLMProvider.OpenAIEmbedding, lambda config: OpenAIEmbedding(config)
)
ModelFactory.register_embedding(
    LLMProvider.AzureOpenAIEmbedding, lambda config: AzureOpenAIEmbedding(config)
)
ModelFactory.register_embedding(
    LLMProvider.AzureInferenceEmbedding, lambda config: AzureInferenceEmbedding(config)
)
