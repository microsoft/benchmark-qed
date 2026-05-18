# Copyright (c) 2025 Microsoft Corporation.
"""A factory for supported llm types backed by ``graphrag-llm``."""

import importlib
from typing import ClassVar

from graphrag_llm.completion import LLMCompletion, create_completion
from graphrag_llm.completion.completion_factory import register_completion
from graphrag_llm.embedding import LLMEmbedding, create_embedding
from graphrag_llm.embedding.embedding_factory import register_embedding

from benchmark_qed.config.llm_config import LLMConfig, ModelType


def _register_custom_provider(model_config: LLMConfig, model_type: ModelType) -> None:
    """Register a custom provider with the underlying graphrag-llm factory."""
    provider = next(
        (
            p
            for p in model_config.custom_providers
            if p.name == str(model_config.llm_provider) and p.model_type == model_type
        ),
        None,
    )
    if provider is None:
        return
    try:
        module = importlib.import_module(provider.module)
    except ImportError as e:
        msg = (
            f"Failed to import custom provider '{provider.name}' "
            f"from module '{provider.module}'. Please check the module and class name."
        )
        raise ImportError(msg) from e
    try:
        model_class = getattr(module, provider.model_class)
    except AttributeError as e:
        msg = (
            f"Failed to load custom provider '{provider.name}': class "
            f"'{provider.model_class}' not found in module '{provider.module}'. "
            "Please check the module and class name."
        )
        raise AttributeError(msg) from e

    if model_type == ModelType.Chat:
        register_completion(provider.name, model_class)
    else:
        register_embedding(provider.name, model_class)


class ModelFactory:
    """Factory for creating ``graphrag-llm`` model instances from an :class:`LLMConfig`."""

    _registered_chat: ClassVar[set[str]] = set()
    _registered_embedding: ClassVar[set[str]] = set()

    @classmethod
    def create_chat_model(cls, model_config: LLMConfig) -> LLMCompletion:
        """Create a chat completion client.

        Built-in providers (OpenAI, Azure OpenAI, Azure AI Inference) are
        served by ``LiteLLMCompletion``. Custom providers declared via
        :class:`benchmark_qed.config.llm_config.CustomLLMProvider` are
        registered with ``graphrag_llm.completion.completion_factory`` and
        instantiated through it.
        """
        provider = str(model_config.llm_provider)
        custom_names = {
            p.name
            for p in model_config.custom_providers
            if p.model_type == ModelType.Chat
        }
        if provider in custom_names and provider not in cls._registered_chat:
            _register_custom_provider(model_config, ModelType.Chat)
            cls._registered_chat.add(provider)
        return create_completion(model_config.to_model_config())

    @classmethod
    def create_embedding_model(cls, model_config: LLMConfig) -> LLMEmbedding:
        """Create an embedding client (see :meth:`create_chat_model`)."""
        provider = str(model_config.llm_provider)
        custom_names = {
            p.name
            for p in model_config.custom_providers
            if p.model_type == ModelType.Embedding
        }
        if provider in custom_names and provider not in cls._registered_embedding:
            _register_custom_provider(model_config, ModelType.Embedding)
            cls._registered_embedding.add(provider)
        return create_embedding(model_config.to_model_config())
