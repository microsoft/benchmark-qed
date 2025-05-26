# Copyright (c) 2025 Microsoft Corporation.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import tiktoken

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.llm.type.base import ChatModel


@dataclass
class SummarizationResult:
    summary: str | list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    llm_calls: int


class BaseSummarizer(ABC):
    def __init__(
        self,
        llm: ChatModel,
        token_encoder: tiktoken.Encoding | None = None,
    ):
        self.llm = llm
        self.token_encoder = token_encoder

    @abstractmethod
    async def asummarize(
        self, text_units: list[TextUnit], *args: Any, **kwargs: Any
    ) -> SummarizationResult:
        pass
