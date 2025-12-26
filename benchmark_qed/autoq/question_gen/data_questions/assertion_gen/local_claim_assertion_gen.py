# Copyright (c) 2025 Microsoft Corporation.
"""Generate assertions for evaluating answer accuracy based on claims for local questions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autoq.prompts import data_questions
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.base import (
    Assertion,
    AssertionGenerationResult,
    BaseAssertionGenerator,
    ClaimDict,
)
from benchmark_qed.config.defaults import LLM_PARAMS, MAX_ASSERTIONS
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.validator import (
        AssertionValidator,
    )
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

ASSERTION_GEN_PROMPTS_PATH = Path(data_questions.__file__).parent


class LocalClaimAssertionGenerator(BaseAssertionGenerator):
    """
    Generate factual assertions for evaluating answer accuracy based on claims for local questions.

    This generator is designed for data_local_questions and handles simple claim lists.
    Takes a question and a list of relevant claims as input, and generates testable assertions
    that can be used as unit tests to verify the accuracy of answers to the question.

    Supports optional validation of generated assertions using AssertionValidator.

    Optimized for simple, direct processing without complex batching overhead.
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        system_prompt: Template | None = None,
        max_assertions: int | None = MAX_ASSERTIONS,
        validator: AssertionValidator | None = None,
        max_concurrent_questions: int | None = None,
    ) -> None:
        super().__init__(
            llm,
            llm_params,
            json_mode,
            max_assertions,
            validator,
            max_concurrent_questions,
        )

        system_prompt = system_prompt or load_template_file(
            ASSERTION_GEN_PROMPTS_PATH
            / "assertions"
            / "local_claim_assertion_gen_prompt.txt"
        )
        if isinstance(system_prompt, str):
            system_prompt = Template(system_prompt)
        self.system_prompt = system_prompt

        # Load max assertion instruction template for dynamic count limiting
        self._max_assertion_instruction_prompt = load_template_file(
            ASSERTION_GEN_PROMPTS_PATH
            / "assertions"
            / "local_max_assertion_instruction_prompt.txt"
        )

    async def agenerate_assertions(
        self, question_text: str, **kwargs: Any
    ) -> AssertionGenerationResult:
        """Generate assertions for data local questions.

        Args:
            question_text: The question text to generate assertions for.
            **kwargs: Additional parameters:
                - claims: List of claims to generate assertions from
        """
        claims: list[ClaimDict] = kwargs.get("claims", [])

        if not claims:
            log.warning("No claims provided for assertion generation")
            return AssertionGenerationResult(assertions=[], total_assertions=0)

        # Process all claims in a single batch for local questions
        claims_text = self._build_context(claims)

        if self.system_prompt is None:
            msg = "System prompt cannot be None"
            raise ValueError(msg)

        # Build base prompt
        base_prompt = self.system_prompt.substitute(
            query=question_text, context_data=claims_text
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

        # Parse and create Assertion objects from LLM response
        assertions = self._parse_assertions(parsed_assertions, claims=claims)

        # Validate assertions with LLM if validator is configured
        assertions = await self._validate_assertions(assertions, question_text)

        # Apply ranking and limiting (None means no limit)
        assertions = self._rank_and_limit_assertions(assertions, self.max_assertions)

        return AssertionGenerationResult(
            assertions=assertions,
            total_assertions=len(assertions),
        )

    def _parse_assertions(
        self, parsed_assertions: list[dict[str, Any]], **kwargs: Any
    ) -> list[Assertion]:
        """
        Parse assertions from LLM response and create Assertion objects with claim ID mapping.

        Assertions with hallucinated source IDs (not in claim mapping) are discarded.

        Args:
            parsed_assertions: Raw assertions from LLM response as dictionaries
            **kwargs: Additional parameters, expects 'claims' for claim mapping

        Returns
        -------
            List of validated Assertion objects with mapped claim sources
        """
        claims = kwargs.get("claims", [])

        # Build claim ID mapping
        claim_id_to_text = (
            {f"claim_{i + 1}": claim for i, claim in enumerate(claims)}
            if claims
            else {}
        )

        validated_assertions = []
        for assertion in parsed_assertions:
            result = self._process_single_assertion(assertion, claim_id_to_text)
            if result:
                validated_assertions.append(result)

        return validated_assertions

    def _process_single_assertion(
        self, assertion: dict[str, Any], claim_id_to_text: dict[str, Any]
    ) -> Assertion | None:
        """Process a single assertion and return Assertion object or None if invalid."""
        statement = assertion.get("statement", "").strip()
        score = assertion.get("score", 5)
        sources = assertion.get("sources", [])

        # Validate basic fields
        if not statement or not isinstance(score, int) or not (1 <= score <= 10):
            return None

        if not claim_id_to_text:
            log.warning(
                "No claims provided for source mapping, skipping: '%s...'",
                statement[:80],
            )
            return None

        if not sources:
            log.warning("Assertion has no sources, skipping: '%s...'", statement[:80])
            return None

        # Map sources and detect hallucinations
        source_claim_texts, source_chunks, hallucinated = self._map_claim_sources(
            sources, claim_id_to_text
        )

        # Discard if all sources are hallucinated
        if not source_claim_texts:
            log.warning(
                "Discarding assertion with all hallucinated sources: '%s...' (invalid: %s)",
                statement[:80],
                hallucinated,
            )
            return None

        # Log partial hallucinations
        if hallucinated:
            log.warning(
                "Assertion has %s hallucinated source(s): %s (keeping %s valid)",
                len(hallucinated),
                hallucinated,
                len(source_claim_texts),
            )

        if not source_chunks:
            log.warning(
                "Assertion has no traceable source texts, skipping: '%s...'",
                statement[:80],
            )
            return None

        try:
            return Assertion(
                statement=statement,
                score=score,
                sources=list(source_chunks),
                reasoning=assertion.get("reasoning", ""),
                attributes={
                    **assertion.get("attributes", {}),
                    "source_claims": source_claim_texts,
                },
            )
        except ValueError as e:
            log.warning("Skipping invalid assertion: %s", e)
            return None

    def _map_claim_sources(
        self, sources: list[Any], claim_id_to_text: dict[str, Any]
    ) -> tuple[list[str], set[str], list[str]]:
        """Map source IDs to claim texts and aggregate source chunks.

        Returns
        -------
            Tuple of (source_claim_texts, source_chunks, hallucinated_sources)
        """
        source_claim_texts = []
        source_chunks: set[str] = set()
        hallucinated = []

        for source in sources:
            source_str = str(source).strip()
            claim = claim_id_to_text.get(source_str)

            if claim:
                source_claim_texts.append(claim.get("statement", ""))
                source_chunks.update(s["text"] for s in claim.get("sources", []))
            else:
                hallucinated.append(source_str)

        return source_claim_texts, source_chunks, hallucinated

    @staticmethod
    def _build_context(claims: list[ClaimDict]) -> str:
        """Format claims list for the assertion generation prompt."""
        if not claims:
            return "No claims provided."

        formatted_claims = []
        for i, claim in enumerate(claims):
            claim_id = f"claim_{i + 1}"
            statement = claim.get("statement", "")
            score = claim.get("score", 0)

            formatted_claim = (
                f"ID: {claim_id}\nStatement: {statement}\nImportance Score: {score}\n"
            )
            formatted_claims.append(formatted_claim)

        return "\n".join(formatted_claims)
