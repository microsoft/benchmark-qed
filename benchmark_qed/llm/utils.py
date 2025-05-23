# Copyright (c) 2025 Microsoft Corporation.
"""Utils for llm models."""

import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from benchmark_qed.llm.type.base import ChatModel

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseModel")


async def chat_typed_response(
    llm: ChatModel,
    messages: list[dict[str, str]],
    data_model: Callable[..., T],
    retries: int = 3,
    **kwargs: Any,
) -> T:
    """Send a chat message to the model and parse the response."""
    for _ in range(retries):
        try:
            response = await llm.chat(
                messages=messages,
                **kwargs,
            )
            json_response = str(response.output.content)
            parsed_response = json.loads(json_response)
            return data_model(**parsed_response)
        except json.JSONDecodeError as e:
            msg = f"JSON decoding error: {e}, retrying..."
            logger.warning(msg)
        except ValidationError as e:
            msg = f"Validation error: {e}, retrying..."
            logger.warning(msg)
    msg = "Failed to get a valid response from the model after multiple retries."
    raise RuntimeError(msg)
