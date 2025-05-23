# Copyright (c) 2025 Microsoft Corporation.
"""LLM configuration module."""

import os
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class LLMProvider(StrEnum):
    """Enum for the LLM provider."""

    OpenAIChat = "openai.chat"
    OpenAIEmbedding = "openai.embedding"
    AzureOpenAIChat = "azure.openai.chat"
    AzureOpenAIEmbedding = "azure.openai.embedding"
    AzureInferenceChat = "azure.inference.chat"
    AzureInferenceEmbedding = "azure.inference.embedding"


class AuthType(StrEnum):
    """Enum for the authentication type."""

    API = "api_key"
    AzureManagedIdentity = "azure_managed_identity"


class LLMConfig(BaseModel):
    """Configuration for the LLM to use."""

    model: str = Field(
        default="gpt-4-turbo",
        description="The name of the model to use for scoring. This should be a valid model name.",
    )
    auth_type: str = Field(
        default="azure",
        description="The type of authentication to use. This should be either 'azure' or 'openai'.",
    )
    api_key: str = Field(
        default=os.environ.get("OPENAI_API_KEY", ""),
        description="The API key to use for the model. This should be a valid API key.",
    )
    concurrent_requests: int = Field(
        default=4,
        description="The number of concurrent requests to send to the model. This should be a positive integer.",
    )
    llm_provider: LLMProvider | str = Field(
        default=LLMProvider.OpenAIChat,
        description="The type of model to use. This should be either 'openai' or 'azure'.",
    )

    init_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments to pass to the model when initializing it.",
    )

    call_args: dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.0, "seed": 42},
        description="Additional arguments to pass to the model when calling it.",
    )

    @model_validator(mode="after")
    def check_api_key(self) -> Self:
        """Check if the API key is set."""
        if self.auth_type == "api_key" and (
            self.api_key is None or self.api_key.strip() == ""
        ):
            msg = "API key is required."
            raise ValueError(msg)
        return self
