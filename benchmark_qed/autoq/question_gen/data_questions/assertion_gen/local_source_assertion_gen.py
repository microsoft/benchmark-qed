# Copyright (c) 2025 Microsoft Corporation.
"""Generate assertions for evaluating answer accuracy directly from source texts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autoq.prompts import data_questions
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.base import (
    Assertion,
    AssertionGenerationResult,
    BaseAssertionGenerator,
)
from benchmark_qed.config.defaults import LLM_PARAMS
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

ASSERTION_GEN_PROMPTS_PATH = Path(data_questions.__file__).parent


class LocalSourceAssertionGenerator(BaseAssertionGenerator):
    """
    Generate factual assertions directly from source text passages for local questions.

    This generator bypasses the intermediate claim extraction step and generates
    assertions directly from the source text units that were used to create the question.
    This is a simpler, faster approach that preserves full context from source texts.

    Advantages over claim-based generation:
    - Single LLM call instead of two (claim extraction + assertion generation)
    - Full context preserved - no intermediate filtering
    - Avoids potential claim extraction errors propagating
    - Better for cases where source texts are already concise and relevant

    Use this generator when:
    - Source texts are relatively short and focused
    - You want to minimize LLM calls for cost/latency reasons
    - You want to preserve full context without intermediate abstraction
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        system_prompt: Template | None = None,
        max_assertions: int | None = 5,
    ) -> None:
        """
        Initialize the LocalSourceAssertionGenerator.

        Parameters
        ----------
        llm : ChatModel
            The language model to use for assertion generation.
        llm_params : dict[str, Any]
            Parameters to pass to the LLM.
        json_mode : bool
            Whether to use JSON mode for structured output.
        system_prompt : Template | None
            Custom system prompt template. If None, uses default.
        max_assertions : int | None
            Maximum number of assertions to generate per question.
            None means no limit.
        """
        super().__init__(llm, llm_params, json_mode, max_assertions)

        system_prompt = (
            system_prompt
            if system_prompt
            else load_template_file(
                ASSERTION_GEN_PROMPTS_PATH
                / "assertions"
                / "local_source_assertion_gen_prompt.txt"
            )
        )
        if isinstance(system_prompt, str):
            system_prompt = Template(system_prompt)
        self.system_prompt = system_prompt

        # Load max assertion instruction template for dynamic count limiting
        self._max_assertion_instruction_prompt = load_template_file(
            ASSERTION_GEN_PROMPTS_PATH
            / "assertions"
            / "local_source_max_assertion_instruction_prompt.txt"
        )

    async def agenerate_assertions(
        self, question_text: str, **kwargs: Any
    ) -> AssertionGenerationResult:
        """
        Generate assertions directly from source text units.

        Parameters
        ----------
        question_text : str
            The question text to generate assertions for.
        **kwargs : Any
            Additional parameters. Expects 'source_texts' as a list of TextUnit objects
            or a list of dicts with 'id' and 'text' keys.

        Returns
        -------
        AssertionGenerationResult
            Result containing the generated assertions.
        """
        source_texts: list[TextUnit | dict[str, Any]] = kwargs.get("source_texts", [])
        if not source_texts:
            log.warning("No source texts provided for assertion generation")
            return AssertionGenerationResult(assertions=[], total_assertions=0)

        # Build context from source texts
        context_data = self._build_context(source_texts)

        if self.system_prompt is None:
            msg = "System prompt cannot be None"
            raise ValueError(msg)

        # Build base prompt
        base_prompt = self.system_prompt.substitute(
            query=question_text, context_data=context_data
        )

        # Dynamically add count instruction if max_assertions is specified
        if self.max_assertions is not None and self.max_assertions > 0:
            count_instruction = self._max_assertion_instruction_prompt.substitute(
                max_assertions=self.max_assertions
            )
            prompt_content = base_prompt + "\n\n" + count_instruction
        else:
            prompt_content = base_prompt

        messages = [
            {
                "role": "system",
                "content": prompt_content,
            },
        ]

        result = await self.llm.chat(messages=messages, **self.llm_params)
        log.debug("Assertion results: %s", result)
        response, j = try_parse_json_object(result.output.content)
        if j == {}:
            msg = f"Invalid json response, returning empty assertion list: {response}"
            log.warning(msg)
            return AssertionGenerationResult(assertions=[], total_assertions=0)

        parsed_assertions = json.loads(response).get("assertions")
        if not parsed_assertions or not isinstance(parsed_assertions, list):
            log.warning(
                "No assertions found in the response, returning empty assertion list"
            )
            return AssertionGenerationResult(assertions=[], total_assertions=0)

        # Validate and clean the assertions
        validated_assertions = self._parse_assertions(
            parsed_assertions, source_texts=source_texts
        )

        # Apply ranking and limiting (None means no limit)
        validated_assertions = self._rank_and_limit_assertions(
            validated_assertions, self.max_assertions
        )

        return AssertionGenerationResult(
            assertions=validated_assertions,
            total_assertions=len(validated_assertions),
        )

    def _parse_assertions(
        self, parsed_assertions: list[dict[str, Any]], **kwargs: Any
    ) -> list[Assertion]:
        """
        Validate assertions and create Assertion objects with source text mapping.

        Parameters
        ----------
        parsed_assertions : list[dict[str, Any]]
            Raw assertions from LLM response as dictionaries.
        **kwargs : Any
            Additional parameters, expects 'source_texts' for source mapping.

        Returns
        -------
        list[Assertion]
            List of validated Assertion objects with mapped source texts.
        """
        source_texts: list[TextUnit | dict[str, Any]] = kwargs.get("source_texts", [])
        validated_assertions = []

        # Create source ID to text mapping
        source_id_to_text: dict[str, str] = {}
        for i, source in enumerate(source_texts):
            if isinstance(source, TextUnit):
                source_id = source.short_id or f"source_{i + 1}"
                source_id_to_text[source_id] = source.text
                # Also map the generated ID format
                source_id_to_text[f"source_{i + 1}"] = source.text
            else:
                # Handle dict format
                source_id = source.get("id", f"source_{i + 1}")
                source_id_to_text[source_id] = source.get("text", "")
                source_id_to_text[f"source_{i + 1}"] = source.get("text", "")

        for assertion in parsed_assertions:
            statement = assertion.get("statement", "").strip()
            score = assertion.get("score", 5)

            if statement != "" and isinstance(score, int) and 1 <= score <= 10:
                # Map source IDs to actual source texts
                sources = assertion.get("sources", [])
                source_chunk_texts: list[str] = []

                if source_id_to_text and sources:
                    for source in sources:
                        source_str = str(source).strip()
                        if source_str in source_id_to_text:
                            source_chunk_texts.append(source_id_to_text[source_str])
                        else:
                            # Keep original source if not found in mapping
                            log.debug("Source ID not found in mapping: %s", source_str)
                else:
                    # If no source texts provided or no sources, use empty list
                    source_chunk_texts = []

                # Create Assertion object
                try:
                    assertion_obj = Assertion(
                        statement=statement,
                        score=score,
                        sources=source_chunk_texts,
                        attributes={
                            **assertion.get("attributes", {}),
                            "source_ids": sources,
                        },
                    )
                    # Debug logging for source counts
                    source_count = len(source_chunk_texts) if source_chunk_texts else 0
                    if source_count == 0:
                        log.warning(
                            "Source assertion created with 0 sources: '%s...'",
                            statement[:100],
                        )
                        log.debug("  Original sources: %s", sources)
                        log.debug(
                            "  Available source IDs: %s", list(source_id_to_text.keys())
                        )
                    else:
                        log.debug(
                            "Source assertion created with %s sources", source_count
                        )

                    validated_assertions.append(assertion_obj)
                except ValueError as e:
                    log.warning("Skipping invalid assertion: %s", e)
                    continue

        return validated_assertions

    @staticmethod
    def _build_context(source_texts: list[TextUnit | dict[str, Any]]) -> str:
        """
        Format source texts for the assertion generation prompt.

        Parameters
        ----------
        source_texts : list[TextUnit | dict[str, Any]]
            List of source text units or dicts.

        Returns
        -------
        str
            Formatted context string for the prompt.
        """
        if not source_texts:
            return "No source texts provided."

        formatted_sources = []
        for i, source in enumerate(source_texts):
            source_id = f"source_{i + 1}"

            if isinstance(source, TextUnit):
                text = source.text
                source_id = source.short_id or source_id
            else:
                text = source.get("text", "")
                source_id = source.get("id", source_id)

            formatted_source = f"ID: {source_id}\nText: {text}\n"
            formatted_sources.append(formatted_source)

        return "\n".join(formatted_sources)
