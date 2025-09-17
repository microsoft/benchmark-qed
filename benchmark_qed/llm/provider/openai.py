# Copyright (c) 2025 Microsoft Corporation.
"""A module containing openai model provider definitions."""

import asyncio
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import (
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from benchmark_qed.config.llm_config import AuthType, LLMConfig, RetryConfig
from benchmark_qed.llm.type.base import BaseModelOutput, BaseModelResponse, Usage

REASONING_MODELS = ["o3", "o4-mini", "o3-mini", "o1-mini", "o1", "o1-pro"]

# Common retryable exceptions for OpenAI services
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


def async_retry(retry_config: RetryConfig) -> AsyncRetrying:
    """Create a tenacity AsyncRetrying instance from LLMConfig.

    Args:
        llm_config: The LLM configuration containing retry settings.

    Returns
    -------
        AsyncRetrying instance configured with the provided settings.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(retry_config.retries),
        wait=wait_exponential_jitter(
            initial=retry_config.base_delay,
            max=retry_config.max_delay,
            jitter=retry_config.base_delay * 0.25
            if retry_config.jitter
            else 0,
            exp_base=retry_config.backoff_factor,
        ),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )


class BaseOpenAIChat:
    """An OpenAI Chat Model provider."""

    def __init__(
        self, client: AsyncAzureOpenAI | AsyncOpenAI, llm_config: LLMConfig
    ) -> None:
        self._client = client
        self._model = llm_config.model
        self._semaphore = asyncio.Semaphore(llm_config.concurrent_requests)
        self._usage = Usage(model=llm_config.model)
        self._retry_config = llm_config.retry_config

    def get_usage(self) -> dict[str, Any]:
        """Get the usage of the Model."""
        return self._usage.model_dump()

    async def _create_chat_completion(
        self, messages: list[dict[str, str]], **kwargs: dict[str, Any]
    ) -> Any:
        """Create a chat completion using the OpenAI client.

        Args:
            messages: The messages to send to the model.
            kwargs: Additional arguments to pass to the model.

        Returns
        -------
            The chat completion response.
        """
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            **kwargs,  # type: ignore
        )

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

        response = None
        async with self._semaphore:
            async for attempt in async_retry(self._retry_config):
                with attempt:
                    response = await self._create_chat_completion(messages, **kwargs)

        if response is None:
            msg = "No response received from Azure Chat API"
            raise ValueError(msg)

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
        self._retry_config = llm_config.retry_config

    def get_usage(self) -> dict[str, Any]:
        """Get the usage of the Model."""
        return self._usage.model_dump()

    async def _create_embeddings(self, text_list: list[str], **kwargs: Any) -> Any:
        """Create embeddings using the OpenAI client.

        Args:
            text_list: The list of text to generate embeddings for.
            kwargs: Additional arguments to pass to the model.

        Returns
        -------
            The embedding response.
        """
        return await self._client.embeddings.create(
            model=self._model,
            input=text_list,
            **kwargs,
        )

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
        response = None
        async with self._semaphore:
            async for attempt in async_retry(self._retry_config):
                with attempt:
                    response = await self._create_embeddings(text_list, **kwargs)

        if response is None:
            msg = "No response received from Azure Chat API"
            raise ValueError(msg)

        self._usage.add_usage(prompt_tokens=response.usage.prompt_tokens)

        return [embedding.embedding for embedding in response.data]


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
