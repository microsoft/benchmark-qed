# Copyright (c) 2025 Microsoft Corporation.
"""LLM configuration module."""

import os
from enum import StrEnum
from typing import Any, ClassVar, Self

from graphrag_llm.config import ModelConfig
from graphrag_llm.config.metrics_config import MetricsConfig
from graphrag_llm.config.rate_limit_config import RateLimitConfig
from graphrag_llm.config.retry_config import RetryConfig
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
    CopilotChat = "github.copilot.chat"
    CopilotEmbedding = "github.copilot.embedding"


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
    Provider = "provider"


# Mapping from benchmark-qed LLMProvider values to litellm-style model_provider strings.
_PROVIDER_TO_MODEL_PROVIDER: dict[str, str] = {
    LLMProvider.OpenAIChat: "openai",
    LLMProvider.OpenAIEmbedding: "openai",
    LLMProvider.AzureOpenAIChat: "azure",
    LLMProvider.AzureOpenAIEmbedding: "azure",
    LLMProvider.AzureInferenceChat: "azure_ai",
    LLMProvider.AzureInferenceEmbedding: "azure_ai",
    LLMProvider.CopilotChat: "github_copilot",
    LLMProvider.CopilotEmbedding: "github_copilot",
}


class LLMConfig(BaseModel):
    """Configuration for the LLM to use."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    model: str = Field(
        default="gpt-4.1",
        description="The name of the model to use for scoring. This should be a valid model name.",
    )
    auth_type: AuthType = Field(
        default=AuthType.API,
        description="Authentication: 'api_key', 'azure_managed_identity', or 'provider' (managed by the provider).",
    )
    api_key: SecretStr = Field(
        default_factory=lambda data: SecretStr(
            ""
            if data.get("auth_type") == AuthType.Provider
            else os.environ.get("OPENAI_API_KEY", "")
        ),
        exclude_if=lambda value: value.get_secret_value() == "",
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
        default_factory=lambda data: (
            {}
            if data.get("llm_provider") == LLMProvider.CopilotEmbedding
            else {"temperature": 0.0, "seed": 42}
        ),
        description="Additional arguments to pass to the model when calling it.",
    )

    retry: RetryConfig | None = Field(
        default=None,
        description="Optional retry policy for transient provider errors (for example, HTTP 429).",
    )

    custom_providers: list[CustomLLMProvider] = Field(
        default_factory=list,
        description="List of custom LLM providers to register.",
    )

    @model_validator(mode="after")
    def check_api_key(self) -> Self:
        """Validate credentials without borrowing OpenAI credentials for provider auth."""
        if (
            _PROVIDER_TO_MODEL_PROVIDER.get(str(self.llm_provider)) == "github_copilot"
            and self.auth_type != AuthType.Provider
        ):
            msg = "GitHub Copilot requires auth_type='provider'; omit api_key."
            raise ValueError(msg)
        if self.auth_type == AuthType.Provider:
            if "api_key" in self.model_fields_set:
                msg = "Omit api_key when auth_type='provider'; the provider manages credentials."
                raise ValueError(msg)
            credential_overrides = {
                "api_key",
                "api_token",
                "azure_ad_token",
                "azure_ad_token_provider",
                "auth_method",
                "type",
                "model_provider",
            }
            if credential_overrides.intersection(self.init_args) or (
                credential_overrides.intersection(self.call_args)
            ):
                msg = "Provider-managed authentication cannot include credential, auth_method, or routing overrides in init_args/call_args."
                raise ValueError(msg)
        if self.auth_type == "api_key" and (
            self.api_key is None or self.api_key.get_secret_value().strip() == ""
        ):
            msg = "API key is required."
            raise ValueError(msg)
        return self

    def to_model_config(self) -> ModelConfig:
        """Translate this benchmark-qed config into a graphrag-llm ``ModelConfig``.

        Built-in providers map to LiteLLM provider strings (``openai``,
        ``azure``, ``azure_ai``, ``github_copilot``). Custom providers are passed through with
        their registered name as the ``ModelConfig.type`` so that
        ``register_completion`` / ``register_embedding`` callers can resolve
        them.
        """
        provider_key = str(self.llm_provider)
        is_builtin = provider_key in _PROVIDER_TO_MODEL_PROVIDER
        if not is_builtin and not any(
            provider.name == provider_key for provider in self.custom_providers
        ):
            msg = (
                f"Unknown LLM provider '{provider_key}'. Choose a supported llm_provider "
                f"({', '.join(_PROVIDER_TO_MODEL_PROVIDER)}) or declare it in custom_providers."
            )
            raise ValueError(msg)
        model_provider = _PROVIDER_TO_MODEL_PROVIDER.get(provider_key, provider_key)

        init_args = dict(self.init_args)
        api_base = init_args.pop("azure_endpoint", None)
        api_version = init_args.pop("api_version", None)

        if self.auth_type == AuthType.Provider and "provider" not in {
            method.value for method in AuthMethod
        }:
            msg = (
                "Provider-managed authentication requires a graphrag-llm release "
                "supporting auth_method='provider'. The pinned 3.3.0 release does "
                "not support it; update the dependency after upstream support is published."
            )
            raise ValueError(msg)
        auth_method = AuthMethod(self.auth_type.value)
        api_key: str | None
        if self.auth_type != AuthType.API:
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
            "auth_method": auth_method,
            "azure_deployment_name": azure_deployment_name,
            "call_args": dict(self.call_args),
            "retry": self.retry,
            "rate_limit": rate_limit,
            "metrics": metrics,
        }
        if self.auth_type != AuthType.Provider:
            config_kwargs["api_key"] = api_key
        config_kwargs.update(init_args)

        if not is_builtin:
            # Use the custom provider's name as the strategy/type so that the
            # corresponding ``register_completion`` / ``register_embedding`` entry
            # is selected by graphrag-llm's factories.
            config_kwargs["type"] = provider_key

        return ModelConfig(**config_kwargs)
