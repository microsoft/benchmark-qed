# Copyright (c) 2025 Microsoft Corporation.
"""A module containing azure inference model provider definitions."""

import asyncio
from typing import Any, cast

from azure.ai.inference.aio import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.models import ChatCompletions, EmbeddingEncodingFormat
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from benchmark_qed.config.llm_config import AuthType, LLMConfig, RetryConfig
from benchmark_qed.llm.type.base import BaseModelOutput, BaseModelResponse, Usage

# Common retryable exceptions for Azure services
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (HttpResponseError,)


def _async_retry(retry_config: RetryConfig) -> AsyncRetrying:
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
            jitter=retry_config.base_delay * 0.25 if retry_config.jitter else 0,
            exp_base=retry_config.backoff_factor,
        ),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )


class AzureInferenceChat:
    """An Azure Chat Model provider."""

    def __init__(self, llm_config: LLMConfig) -> None:
        if llm_config.auth_type == AuthType.AzureManagedIdentity:
            credential = DefaultAzureCredential()
        else:
            credential = AzureKeyCredential(llm_config.api_key.get_secret_value())
        self._client = ChatCompletionsClient(
            endpoint=llm_config.init_args["azure_endpoint"],
            credential=credential,  # type: ignore
            **llm_config.init_args,
        )
        self._model = llm_config.model
        self._semaphore = asyncio.Semaphore(llm_config.concurrent_requests)
        self._usage = Usage(model=llm_config.model)
        self._retry_config = llm_config.retry_config

    def get_usage(self) -> dict[str, Any]:
        """Get the usage of the Model."""
        return self._usage.model_dump()

    async def _complete_chat(
        self, messages: list[dict[str, str]], **kwargs: dict[str, Any]
    ) -> ChatCompletions:
        """Complete a chat request using the Azure client.

        Args:
            messages: The messages to send to the model.
            kwargs: Additional arguments to pass to the model.

        Returns
        -------
            The chat completion response.
        """
        return cast(
            ChatCompletions,
            await self._client.complete(
                model=self._model,
                messages=messages,
                **kwargs,  # type: ignore
            ),  # type: ignore
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
        response = None
        async with self._semaphore:
            async for attempt in _async_retry(self._retry_config):
                with attempt:
                    response = await self._complete_chat(messages, **kwargs)

        if response is None:
            msg = "No response received from Azure Chat API"
            raise ValueError(msg)

        content = response.choices[0].message.content.replace(
            "<|im_start|>assistant<|im_sep|>", ""
        )

        history = [
            *messages,
            {"content": content, "role": response.choices[0].message.role},
        ]

        usage_dict = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        self._usage.add_usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        return BaseModelResponse(
            output=BaseModelOutput(
                content=content,
            ),
            history=history,
            usage=usage_dict,
        )


class AzureInferenceEmbedding:
    """An Azure Inference Embedding Model provider."""

    def __init__(self, llm_config: LLMConfig) -> None:
        if llm_config.auth_type == AuthType.AzureManagedIdentity:
            credential = DefaultAzureCredential()
        else:
            credential = AzureKeyCredential(llm_config.api_key.get_secret_value())
        self._client = EmbeddingsClient(
            endpoint=llm_config.init_args["azure_endpoint"],
            credential=credential,  # type: ignore
            **llm_config.init_args,
        )
        self._model = llm_config.model
        self._semaphore = asyncio.Semaphore(llm_config.concurrent_requests)
        self._usage = Usage(model=llm_config.model)
        self._retry_config = llm_config.retry_config

    def get_usage(self) -> dict[str, Any]:
        """Get the usage of the Model."""
        return self._usage.model_dump()

    async def _embed_text(self, text_list: list[str], **kwargs: Any) -> Any:
        """Generate embeddings using the Azure client.

        Args:
            text_list: The list of text to generate embeddings for.
            kwargs: Additional arguments to pass to the model.

        Returns
        -------
            The embedding response.
        """
        return await self._client.embed(
            model=self._model,
            input=text_list,
            encoding_format=EmbeddingEncodingFormat.FLOAT,
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
            async for attempt in _async_retry(self._retry_config):
                with attempt:
                    response = await self._embed_text(text_list, **kwargs)

        if response is None:
            msg = "No response received from Azure Embedding API"
            raise ValueError(msg)

        self._usage.add_usage(prompt_tokens=response.usage.prompt_tokens)

        return [cast(list[float], embedding.embedding) for embedding in response.data]
