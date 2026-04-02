# Copyright (c) 2025 Microsoft Corporation.
"""A module containing openai model provider definitions."""

import asyncio
import logging
import re
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
)

from benchmark_qed.config.llm_config import AuthType, LLMConfig
from benchmark_qed.llm.type.base import BaseModelOutput, BaseModelResponse, Usage

log: logging.Logger = logging.getLogger(__name__)

REASONING_MODELS = ["o3", "o4-mini", "o3-mini", "o1-mini", "o1", "o1-pro"]
_MAX_API_RETRIES = 3
_RETRY_DELAY_SECONDS = 2
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Matches surrogates and C0/C1 control chars (except tab, newline, carriage return)
_BAD_CHARS_RE = re.compile(r"[\ud800-\udfff\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class BaseOpenAIChat:
    """An OpenAI Chat Model provider."""

    def __init__(
        self, client: AsyncAzureOpenAI | AsyncOpenAI, llm_config: LLMConfig
    ) -> None:
        self._client = client
        self._model = llm_config.model
        self._semaphore = asyncio.Semaphore(llm_config.concurrent_requests)
        self._usage = Usage(model=llm_config.model)

    def get_usage(self) -> dict[str, Any]:
        """Get the usage of the Model."""
        return self._usage.model_dump()

    async def chat(
        self, messages: list[dict[str, str]], **kwargs: dict[str, Any]
    ) -> BaseModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if self._model in REASONING_MODELS and "temperature" in kwargs:
            kwargs.pop("temperature")

        # Strip surrogates and control chars that break JSON serialization
        sanitized_messages = [
            {
                k: _BAD_CHARS_RE.sub("", v) if isinstance(v, str) else v
                for k, v in m.items()
            }
            for m in messages
        ]

        for attempt in range(_MAX_API_RETRIES):
            try:
                async with self._semaphore:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=sanitized_messages,  # type: ignore
                        **kwargs,  # type: ignore
                    )
                break
            except (APIConnectionError, APITimeoutError) as e:
                if attempt < _MAX_API_RETRIES - 1:
                    log.warning(
                        "Chat API connection error (attempt %s/%s): %s",
                        attempt + 1,
                        _MAX_API_RETRIES,
                        e,
                    )
                    await asyncio.sleep(
                        _RETRY_DELAY_SECONDS * (attempt + 1),
                    )
                else:
                    raise
            except APIStatusError as e:
                if (
                    e.status_code in _RETRYABLE_STATUS_CODES
                    and attempt < _MAX_API_RETRIES - 1
                ):
                    log.warning(
                        "Chat API error %s (attempt %s/%s): %s",
                        e.status_code,
                        attempt + 1,
                        _MAX_API_RETRIES,
                        e,
                    )
                    await asyncio.sleep(
                        _RETRY_DELAY_SECONDS * (attempt + 1),
                    )
                else:
                    raise
        else:
            # Unreachable: loop always breaks or raises
            msg = "Chat request failed after retries."
            raise RuntimeError(msg)

        history = [
            *messages,
            {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
            },
        ]

        self._usage.add_usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cached_tokens=response.usage.prompt_tokens_details.cached_tokens
            if response.usage.prompt_tokens_details
            else 0,
            completion_reasoning_tokens=response.usage.completion_tokens_details.reasoning_tokens
            if response.usage.completion_tokens_details
            else 0,
            accepted_prediction_tokens=response.usage.completion_tokens_details.accepted_prediction_tokens
            if response.usage.completion_tokens_details
            else 0,
            rejected_prediction_tokens=response.usage.completion_tokens_details.rejected_prediction_tokens
            if response.usage.completion_tokens_details
            else 0,
        )

        return BaseModelResponse(
            output=BaseModelOutput(content=response.choices[0].message.content),
            history=history,
            usage=response.usage.to_dict(),
        )


class OpenAIChat(BaseOpenAIChat):
    """An OpenAI Chat Model provider."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._client = AsyncOpenAI(
            api_key=llm_config.api_key.get_secret_value(),
            **llm_config.init_args,
        )

        super().__init__(self._client, llm_config)


class AzureOpenAIChat(BaseOpenAIChat):
    """An Azure OpenAI Chat Model provider."""

    def __init__(self, llm_config: LLMConfig) -> None:
        azure_endpoint = llm_config.init_args.pop("azure_endpoint")
        api_version = llm_config.init_args.pop("api_version")

        if llm_config.auth_type == AuthType.AzureManagedIdentity:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            self._client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                azure_ad_token_provider=token_provider,
                **llm_config.init_args,
            )
        else:
            self._client = AsyncAzureOpenAI(
                api_key=llm_config.api_key.get_secret_value(),
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                **llm_config.init_args,
            )

        super().__init__(self._client, llm_config)


class BaseOpenAIEmbedding:
    """An OpenAI Embedding Model provider."""

    def __init__(
        self, client: AsyncOpenAI | AsyncAzureOpenAI, llm_config: LLMConfig
    ) -> None:
        self._client = client
        self._model = llm_config.model
        self._semaphore = asyncio.Semaphore(llm_config.concurrent_requests)
        self._usage = Usage(model=llm_config.model)

    def get_usage(self) -> dict[str, Any]:
        """Get the usage of the Model."""
        return self._usage.model_dump()

    async def embed(self, text_list: list[str], **kwargs: Any) -> list[list[float]]:
        """Generate an embedding vector for the given list of strings.

        Args:
            text_list: The texts to generate embeddings for.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns
        -------
            A list of embedding vectors, one per input text.
        """
        for attempt in range(_MAX_API_RETRIES):
            try:
                # Strip surrogates and control chars that break JSON serialization
                sanitized = [_BAD_CHARS_RE.sub("", t) for t in text_list]
                async with self._semaphore:
                    response = await self._client.embeddings.create(
                        model=self._model,
                        input=sanitized,
                        **kwargs,
                    )

                self._usage.add_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                )

                return [embedding.embedding for embedding in response.data]
            except (APIConnectionError, APITimeoutError) as e:
                if attempt < _MAX_API_RETRIES - 1:
                    log.warning(
                        "Embed API connection error (attempt %s/%s): %s",
                        attempt + 1,
                        _MAX_API_RETRIES,
                        e,
                    )
                    await asyncio.sleep(
                        _RETRY_DELAY_SECONDS * (attempt + 1),
                    )
                else:
                    raise
            except APIStatusError as e:
                if (
                    e.status_code in _RETRYABLE_STATUS_CODES
                    and attempt < _MAX_API_RETRIES - 1
                ):
                    log.warning(
                        "Embed API error %s (attempt %s/%s): %s",
                        e.status_code,
                        attempt + 1,
                        _MAX_API_RETRIES,
                        e,
                    )
                    await asyncio.sleep(
                        _RETRY_DELAY_SECONDS * (attempt + 1),
                    )
                else:
                    raise

        # Unreachable: loop always returns or raises on final attempt
        msg = "Embedding failed after retries."
        raise RuntimeError(msg)


class OpenAIEmbedding(BaseOpenAIEmbedding):
    """An OpenAI Embedding Model provider."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._client = AsyncOpenAI(
            api_key=llm_config.api_key.get_secret_value(),
            **llm_config.init_args,
        )

        super().__init__(self._client, llm_config)


class AzureOpenAIEmbedding(BaseOpenAIEmbedding):
    """An Azure OpenAI Embedding Model provider."""

    def __init__(self, llm_config: LLMConfig) -> None:
        azure_deployment = llm_config.init_args.pop("azure_deployment")
        azure_endpoint = llm_config.init_args.pop("azure_endpoint")
        api_version = llm_config.init_args.pop("api_version")

        if llm_config.auth_type == AuthType.AzureManagedIdentity:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )
            self._client = AsyncAzureOpenAI(
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                azure_ad_token_provider=token_provider,
                **llm_config.init_args,
            )
        else:
            self._client = AsyncAzureOpenAI(
                api_key=llm_config.api_key.get_secret_value(),
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                **llm_config.init_args,
            )

        super().__init__(self._client, llm_config)
