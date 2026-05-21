# Copyright (c) 2025 Microsoft Corporation.
"""Module for LLM code."""

from collections.abc import AsyncIterator
from typing import Any

from graphrag_llm.completion import LLMCompletion
from graphrag_llm.embedding import LLMEmbedding
from graphrag_llm.types import (
    LLMCompletionMessagesParam,
    LLMCompletionResponse,
    LLMEmbeddingResponse,
    ResponseFormat,
)


async def chat(
    llm: LLMCompletion,
    messages: LLMCompletionMessagesParam,
    *,
    response_format: type[ResponseFormat] | None = None,
    **kwargs: Any,
) -> LLMCompletionResponse[ResponseFormat]:
    """Run a non-streaming chat completion and narrow away the streaming union.

    Thin wrapper around ``LLMCompletion.completion_async`` that asserts the
    response is not a streaming chunk iterator so callers can directly access
    ``.content`` / ``.formatted_response`` without further type narrowing.
    """
    response = await llm.completion_async(
        messages=messages,
        response_format=response_format,
        **kwargs,
    )
    if isinstance(response, AsyncIterator):
        msg = "Streaming completions are not supported by benchmark_qed.llm.chat."
        raise TypeError(msg)
    return response


async def embed(
    embedding: LLMEmbedding,
    texts: list[str],
    **kwargs: Any,
) -> LLMEmbeddingResponse:
    """Run an embedding request — passthrough wrapper kept for symmetry with ``chat``."""
    return await embedding.embedding_async(input=texts, **kwargs)
