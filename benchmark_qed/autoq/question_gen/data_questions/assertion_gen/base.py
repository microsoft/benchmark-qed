# Copyright (c) 2025 Microsoft Corporation.
"""Base classes for assertion generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from string import Template
from typing import Any

from benchmark_qed.config.defaults import LLM_PARAMS
from benchmark_qed.llm.type.base import ChatModel


@dataclass
class AssertionGenerationResult:
    """Data class for assertion generation results."""

    assertions: list[dict[str, Any]]
    """The generated assertions."""

    total_assertions: int
    """Total number of assertions generated."""


class BaseAssertionGenerator(ABC):
    """
    Base class for generating factual assertions for evaluating answer accuracy in question-answering systems.
    
    Subclasses should implement specific methods for generating assertions from different input types.
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        system_prompt: Template | None = None,
    ) -> None:
        self.llm = llm
        self.llm_params = llm_params
        self.json_mode = json_mode
        self.system_prompt: Template = system_prompt  # type: ignore
        
        if self.json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

    @abstractmethod
    async def agenerate_assertions(
        self, question_text: str, **kwargs: Any
    ) -> AssertionGenerationResult:
        """Generate assertions for evaluating answer accuracy based on a question and additional inputs."""
        pass

    def _validate_assertions(self, parsed_assertions: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        """Validate and clean the assertions."""
        validated_assertions = []
        for assertion in parsed_assertions:
            if (
                assertion.get("statement", "").strip() != ""
                and isinstance(assertion.get("score", 0), int)
                and 0 < assertion.get("score", 0) <= 100
            ):
                validated_assertion = {
                    "statement": assertion.get("statement"),
                    "sources": assertion.get("sources", []),
                    "score": assertion.get("score", 50),
                }
                validated_assertions.append(validated_assertion)
        return validated_assertions
