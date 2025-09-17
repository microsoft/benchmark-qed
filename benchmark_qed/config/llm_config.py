# Copyright (c) 2025 Microsoft Corporation.
"""LLM configuration module."""

import os
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, SecretStr, model_validator


class LLMProvider(StrEnum):
    """Enum for the LLM provider."""

    OpenAIChat = "openai.chat"
    OpenAIEmbedding = "openai.embedding"
    AzureOpenAIChat = "azure.openai.chat"
    AzureOpenAIEmbedding = "azure.openai.embedding"
    AzureInferenceChat = "azure.inference.chat"
    AzureInferenceEmbedding = "azure.inference.embedding"


class ModelType(StrEnum):
    """Enum for the model type."""

    Chat = "chat"
    Embedding = "embedding"


class CustomLLMProvider(BaseModel):
    """Custom LLM provider configuration."""

    model_type: ModelType = Field(
        ...,
        description="The type of model this custom provider implements.",
    )
    name: str = Field(
        ...,
        description="The name of the custom LLM provider.",
    )
    module: str = Field(
        ...,
        description="The module where the custom LLM provider is implemented",
    )
    model_class: str = Field(
        ...,
        description="Class name that implements the ChatModel or EmbeddingModel protocol.",
    )


class AuthType(StrEnum):
    """Enum for the authentication type."""

    API = "api_key"
    AzureManagedIdentity = "azure_managed_identity"


class RetryConfig(BaseModel):
    """Configuration for retrying failed requests."""

    retries: int = Field(
        default=3,
        description="The maximum number of retry attempts.",
    )
    base_delay: float = Field(
        default=1.0,
        description="Initial delay between retries in seconds.",
    )
    max_delay: float = Field(
        default=60.0,
        description="Maximum delay between retries in seconds.",
    )
    backoff_factor: float = Field(
        default=2.0,
        description="Multiplier for exponential backoff.",
    )
    jitter: bool = Field(
        default=True,
        description="Whether to add random jitter to delay times.",
    )


class LLMConfig(BaseModel):
    """Configuration for the LLM to use."""

    model: str = Field(
        default="gpt-4.1",
        description="The name of the model to use for scoring. This should be a valid model name.",
    )
    auth_type: AuthType = Field(
        default=AuthType.API,
        description="The type of authentication to use. This should be either 'api_key' or 'azure_managed_identity'.",
    )
    api_key: SecretStr = Field(
        default=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
        description="The API key to use for the model. This should be a valid API key.",
    )
    concurrent_requests: int = Field(
        default=4,
        description="The number of concurrent requests to send to the model. This should be a positive integer.",
    )
    retry_config: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Configuration for retry behavior.",
    )
    llm_provider: LLMProvider | str = Field(
        default=LLMProvider.OpenAIChat,
        description="The type of model to use.",
    )

    init_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments to pass to the model when initializing it.",
    )

    call_args: dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.0, "seed": 42},
        description="Additional arguments to pass to the model when calling it.",
    )

    custom_providers: list[CustomLLMProvider] = Field(
        default_factory=list,
        description="List of custom LLM providers to register.",
    )

    @model_validator(mode="after")
    def check_api_key(self) -> Self:
        """Check if the API key is set."""
        if self.auth_type == "api_key" and (
            self.api_key is None or self.api_key.get_secret_value().strip() == ""
        ):
            msg = "API key is required."
            raise ValueError(msg)
        return self
