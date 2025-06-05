# Copyright (c) 2025 Microsoft Corporation.
"""A package containing a factory for supported llm types."""

import importlib
from collections.abc import Callable
from typing import ClassVar

from benchmark_qed.config.llm_config import LLMConfig, LLMProvider, ModelType
from benchmark_qed.llm.provider.azure import AzureInferenceChat, AzureInferenceEmbedding
from benchmark_qed.llm.provider.openai import (
    AzureOpenAIChat,
    AzureOpenAIEmbedding,
    OpenAIChat,
    OpenAIEmbedding,
)
from benchmark_qed.llm.type.base import ChatModel, EmbeddingModel


def __get_custom_provider_names(
    model_config: LLMConfig, model_type: ModelType
) -> list[str]:
    """Get the names of custom providers for the given model config."""
    return [
        provider.name
        for provider in model_config.custom_providers
        if provider.model_type == model_type
    ]


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
    def __register_custom_provider(
        cls, model_config: LLMConfig, model_type: ModelType
    ) -> None:
        provider = next(
            filter(
                lambda p: p.name == model_config.llm_provider,
                model_config.custom_providers,
            ),
            None,
        )
        if provider is not None:
            try:
                module = importlib.import_module(provider.module)
                model_class = getattr(module, provider.model_class)
            except ImportError as e:
                msg = (
                    f"Failed to import custom provider '{provider.name}' "
                    f"from module '{provider.module}'. Please check the module and class name."
                )
                raise ImportError(msg) from e
            if model_type == ModelType.Chat:
                cls.register_chat(provider.name, lambda config: model_class(config))
            elif model_type == ModelType.Embedding:
                cls.register_embedding(
                    provider.name, lambda config: model_class(config)
                )

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
        custom_provider_names = __get_custom_provider_names(
            model_config, ModelType.Chat
        )
        if (
            model_config.llm_provider not in cls._chat_registry
            and model_config.llm_provider not in custom_provider_names
        ):
            msg = f"ChatModel implementation '{model_config.llm_provider}' is not registered."
            raise ValueError(msg)
        if model_config.llm_provider in custom_provider_names:
            cls.__register_custom_provider(model_config, ModelType.Chat)
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
        custom_provider_names = __get_custom_provider_names(
            model_config, ModelType.Embedding
        )
        if (
            model_config.llm_provider not in cls._embedding_registry
            and model_config.llm_provider not in custom_provider_names
        ):
            msg = f"EmbeddingModel implementation '{model_config.llm_provider}' is not registered."
            raise ValueError(msg)
        if model_config.llm_provider in custom_provider_names:
            cls.__register_custom_provider(model_config, ModelType.Embedding)
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
