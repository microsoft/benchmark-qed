# Copyright (c) 2025 Microsoft Corporation.
"""LLM configuration module."""

import os
from enum import StrEnum
from typing import Any, Self

from graphrag_llm.config import ModelConfig
from graphrag_llm.config.metrics_config import MetricsConfig
from graphrag_llm.config.rate_limit_config import RateLimitConfig
from graphrag_llm.config.types import AuthMethod, RateLimitType
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


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
        description="Class name that implements the LLMCompletion or LLMEmbedding interface.",
    )


class AuthType(StrEnum):
    """Enum for the authentication type."""

    API = "api_key"
    AzureManagedIdentity = "azure_managed_identity"


# Mapping from benchmark-qed LLMProvider values to litellm-style model_provider strings.
_PROVIDER_TO_MODEL_PROVIDER: dict[str, str] = {
    LLMProvider.OpenAIChat: "openai",
    LLMProvider.OpenAIEmbedding: "openai",
    LLMProvider.AzureOpenAIChat: "azure",
    LLMProvider.AzureOpenAIEmbedding: "azure",
    LLMProvider.AzureInferenceChat: "azure_ai",
    LLMProvider.AzureInferenceEmbedding: "azure_ai",
}


class LLMConfig(BaseModel):
    """Configuration for the LLM to use."""

    model_config = ConfigDict(extra="forbid")

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
    llm_provider: LLMProvider | str = Field(
        default=LLMProvider.OpenAIChat,
        description="The type of model to use.",
    )

    azure_identity_scopes: list[str] = Field(
        default_factory=lambda: ["https://cognitiveservices.azure.com/.default"],
        description="The Azure identity scopes to request when using azure_managed_identity auth. Passed to get_bearer_token_provider.",
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

    def to_model_config(self) -> ModelConfig:
        """Translate this benchmark-qed config into a graphrag-llm ``ModelConfig``.

        Built-in providers map to LiteLLM provider strings (``openai``,
        ``azure``, ``azure_ai``). Custom providers are passed through with
        their registered name as the ``ModelConfig.type`` so that
        ``register_completion`` / ``register_embedding`` callers can resolve
        them.
        """
        provider_key = str(self.llm_provider)
        is_builtin = provider_key in _PROVIDER_TO_MODEL_PROVIDER
        model_provider = _PROVIDER_TO_MODEL_PROVIDER.get(provider_key, provider_key)

        init_args = dict(self.init_args)
        api_base = init_args.pop("azure_endpoint", None)
        api_version = init_args.pop("api_version", None)

        auth_method = (
            AuthMethod.AzureManagedIdentity
            if self.auth_type == AuthType.AzureManagedIdentity
            else AuthMethod.ApiKey
        )
        api_key: str | None
        if auth_method == AuthMethod.AzureManagedIdentity:
            api_key = None
        else:
            api_key = self.api_key.get_secret_value()

        azure_deployment_name = self.model if model_provider == "azure" else None

        rate_limit: RateLimitConfig | None = None
        if self.concurrent_requests and self.concurrent_requests > 0:
            rate_limit = RateLimitConfig(
                type=RateLimitType.SlidingWindow,
                period_in_seconds=1,
                requests_per_period=max(1, self.concurrent_requests),
            )

        # Track usage by enabling a metrics processor with no writer (silent).
        metrics = MetricsConfig(writer=None)

        config_kwargs: dict[str, Any] = {
            "model_provider": model_provider,
            "model": self.model,
            "api_base": api_base,
            "api_version": api_version,
            "api_key": api_key,
            "auth_method": auth_method,
            "azure_deployment_name": azure_deployment_name,
            "call_args": dict(self.call_args),
            "rate_limit": rate_limit,
            "metrics": metrics,
            **init_args,
        }

        if not is_builtin:
            # Use the custom provider's name as the strategy/type so that the
            # corresponding ``register_completion`` / ``register_embedding`` entry
            # is selected by graphrag-llm's factories.
            config_kwargs["type"] = provider_key

        return ModelConfig(**config_kwargs)
