# Copyright (c) 2025 Microsoft Corporation.
"""Generate assertions for evaluating answer accuracy based on claims for global questions."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from benchmark_qed.autod.data_processor.text_utils import (
    num_tokens,
    try_parse_json_object,
)
from benchmark_qed.autoq.prompts import data_questions
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.base import (
    Assertion,
    AssertionGenerationResult,
    BaseAssertionGenerator,
    ClaimDict,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.local_claim_assertion_gen import (
    LocalClaimAssertionGenerator,
)
from benchmark_qed.config.defaults import (
    ASSERTION_BATCH_SIZE,
    ASSERTION_MAX_DATA_TOKENS,
    LLM_PARAMS,
    MAX_ASSERTIONS,
)
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.validator import (
        AssertionValidator,
    )
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

ASSERTION_GEN_PROMPTS_PATH = Path(data_questions.__file__).parent


class GlobalClaimAssertionGenerator(BaseAssertionGenerator):
    """
    Generate factual assertions for evaluating answer accuracy based on claims for global questions.

    This generator is designed for data_global_questions and handles complex batch formats
    where each batch contains question text and ClaimExtractionResult objects.
    It aggregates claims across multiple question contexts and performs cross-question deduplication.

    Supports optional validation of generated assertions using AssertionValidator.

    Supports automatic batching for large claim sets with parallel processing:
    - If batch_size is None or <= 0, processes all claims in a single request
    - If claims exceed batch_size, automatically splits into parallel batches
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        map_system_prompt: Template | None = None,
        reduce_system_prompt: Template | None = None,
        max_assertions: int | None = MAX_ASSERTIONS,
        batch_size: int | None = ASSERTION_BATCH_SIZE,
        concurrent_coroutines: int = 8,
        max_data_tokens: int = ASSERTION_MAX_DATA_TOKENS,
        token_encoder: Any | None = None,
        validator: AssertionValidator | None = None,
    ) -> None:
        super().__init__(llm, llm_params, json_mode, max_assertions, validator)

        # Load prompt templates
        self.map_prompt = (
            map_system_prompt
            if map_system_prompt
            else load_template_file(
                ASSERTION_GEN_PROMPTS_PATH
                / "assertions"
                / "global_claim_assertion_map_prompt.txt"
            )
        )
        if isinstance(self.map_prompt, str):
            self.map_prompt = Template(self.map_prompt)

        self.reduce_prompt = (
            reduce_system_prompt
            if reduce_system_prompt
            else load_template_file(
                ASSERTION_GEN_PROMPTS_PATH
                / "assertions"
                / "global_claim_assertion_reduce_prompt.txt"
            )
        )
        if isinstance(self.reduce_prompt, str):
            self.reduce_prompt = Template(self.reduce_prompt)

        # Load max assertion instruction template for dynamic count limiting
        self._max_assertion_instruction_prompt = load_template_file(
            ASSERTION_GEN_PROMPTS_PATH
            / "assertions"
            / "global_max_assertion_instruction_prompt.txt"
        )

        # Batch processing parameters for complex global processing
        self.batch_size = batch_size
        self.concurrent_coroutines = concurrent_coroutines
        self._semaphore = asyncio.Semaphore(concurrent_coroutines)
        self.max_data_tokens = max_data_tokens
        self.token_encoder = token_encoder

        # Local generator for map phase processing
        self.local_generator = LocalClaimAssertionGenerator(
            llm=self.llm,
            llm_params=self.llm_params,
            json_mode=self.json_mode,
            system_prompt=self.map_prompt,
            max_assertions=None,  # No assertion limiting in map step - focus on quality and comprehensiveness
        )

    async def agenerate_assertions(
        self, question_text: str, **kwargs: Any
    ) -> AssertionGenerationResult:
        """Generate assertions using a map-reduce approach for global questions.

        Map Step: Divide claims into batches and generate initial assertions
        Reduce Step: Consolidate initial assertions into high-level global assertions
        Validation Step (optional): Validate assertions if validator is configured

        Args:
            question_text: The question text to generate assertions for
            **kwargs: Additional parameters, expects 'claims' list (can be simple or batch format)
        """
        # MAP PHASE
        claim_batches = self.build_map_context(kwargs.get("claims", []))
        if not claim_batches:
            log.warning("No claims provided for assertion generation")
            return AssertionGenerationResult(assertions=[], total_assertions=0)

        map_responses = await self.generate_map_responses(question_text, claim_batches)

        # REDUCE PHASE
        reduce_context = self.build_reduce_context(map_responses)
        final_assertions = await self.generate_reduce_response(
            question_text, reduce_context
        )

        # VALIDATION PHASE (optional)
        final_assertions = await self._validate_assertions(
            final_assertions, question_text
        )

        return AssertionGenerationResult(
            assertions=final_assertions, total_assertions=len(final_assertions)
        )

    def build_map_context(self, claims: list[ClaimDict]) -> list[list[ClaimDict]]:
        """Build map context by extracting claims and dividing into batches for parallel processing.

        Args:
            claims_input: Entire list of claims

        Returns
        -------
            List of claim batches, where each batch is a list ofclaims
        """
        if not claims:
            return []

        # create batches based on batch_size
        batch_size = (
            self.batch_size if self.batch_size and self.batch_size > 0 else len(claims)
        )

        # Split claims into batches
        batches = [
            claims[i : i + batch_size] for i in range(0, len(claims), batch_size)
        ]
        log.info(
            "MAP CONTEXT: Created %s batches from %s simple claims",
            len(batches),
            len(claims),
        )
        return batches

    async def generate_map_responses(
        self, question_text: str, claim_batches: list[list[ClaimDict]]
    ) -> list[list[Assertion]]:
        """Generate map responses by running LocalClaimAssertionGenerator in parallel on claim batches.

        Args:
            question_text: The question text for assertion generation
            claim_batches: List of claim batches to process in parallel

        Returns
        -------
            List of assertion lists, one per successfully processed batch
        """
        if not claim_batches:
            return []

        log.info("MAP RESPONSES: Processing %s batches in parallel", len(claim_batches))

        async def process_batch_with_semaphore(
            batch: list[ClaimDict],
        ) -> list[Assertion]:
            async with self._semaphore:
                # Use the shared local generator for map phase processing
                result = await self.local_generator.agenerate_assertions(
                    question_text, claims=batch
                )
                return result.assertions

        # Process batches in parallel
        batch_results = await asyncio.gather(
            *[process_batch_with_semaphore(batch) for batch in claim_batches],
            return_exceptions=True,
        )

        # Collect successful results
        map_responses = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                log.error("Error processing batch %s in MAP step: %s", i, result)
                continue
            if isinstance(result, list):
                map_responses.append(result)

        log.info(
            "MAP RESPONSES: Successfully processed %s out of %s batches",
            len(map_responses),
            len(claim_batches),
        )
        return map_responses

    def build_reduce_context(
        self, map_responses: list[list[Assertion]]
    ) -> tuple[list[Assertion], str]:
        """Build reduce context by merging map responses and formatting for consolidation.

        Args:
            question_text: The question text for context
            map_responses: List of assertion lists from map phase

        Returns
        -------
            Tuple of (unique_assertions, formatted_assertions_text) for reduce phase
        """
        # Flatten all initial assertions
        initial_assertions: list[Assertion] = []
        for assertion_list in map_responses:
            initial_assertions.extend(assertion_list)

        if not initial_assertions:
            return [], ""

        log.info(
            "REDUCE CONTEXT: Merging %s assertions from %s batches",
            len(initial_assertions),
            len(map_responses),
        )

        # Rank assertions by score (descending) and source count (descending)
        ranked_assertions = sorted(
            initial_assertions, key=lambda a: (-a.score, -len(a.sources))
        )

        log.info(
            "REDUCE CONTEXT: Ranked %s unique assertions by score and source count",
            len(ranked_assertions),
        )

        # Format assertions with token limit
        formatted_assertions = []
        selected_assertions = []
        current_tokens = 0

        for i, assertion in enumerate(ranked_assertions, 1):
            statement = assertion.statement
            score = assertion.score
            assertion_text = (
                f"ID: assertion_{i}\nStatement: {statement}\nScore: {score}"
            )
            assertion_tokens = num_tokens(assertion_text, self.token_encoder)

            # Check if adding this assertion would exceed token limit
            if current_tokens + assertion_tokens > self.max_data_tokens:
                log.info(
                    "REDUCE CONTEXT: Reached token limit at assertion %s, stopping at %s tokens",
                    i - 1,
                    current_tokens,
                )
                break

            formatted_assertions.append(assertion_text)
            selected_assertions.append(assertion)
            current_tokens += assertion_tokens

        formatted_text = "\n".join(formatted_assertions)

        log.info(
            "REDUCE CONTEXT: Selected %s of %s assertions within %s tokens (limit: %s)",
            len(selected_assertions),
            len(ranked_assertions),
            current_tokens,
            self.max_data_tokens,
        )
        return selected_assertions, formatted_text

    def _build_assertion_mapping(
        self, assertions: list[Assertion]
    ) -> dict[str, dict[str, Any]]:
        """Build mapping from assertion index to assertion data for source resolution.

        Args:
            assertions: List of assertions to create mapping for

        Returns
        -------
            Dictionary mapping assertion IDs to assertion data
        """
        assertion_id_to_data = {}
        for i, assertion in enumerate(assertions, 1):
            assertion_id = str(i)
            assertion_id_to_data[f"assertion_{assertion_id}"] = {
                "statement": assertion.statement,
                "sources": assertion.sources,
                "score": assertion.score,
            }
        return assertion_id_to_data

    def _map_sources_and_aggregate(
        self, sources: list[Any], assertion_mapping: dict[str, dict[str, Any]]
    ) -> tuple[list[Any], list[Any], list[str]]:
        """Map source assertion IDs to statements and aggregate source chunks.

        Args:
            sources: List of source IDs from consolidated assertion
            assertion_mapping: Mapping from assertion IDs to assertion data

        Returns
        -------
            Tuple of (mapped_sources, aggregated_source_chunks, hallucinated_sources)
        """
        mapped_sources = []
        aggregated_source_chunks = []
        hallucinated_sources = []

        if assertion_mapping and sources:
            for source in sources:
                source_str = str(source).strip()
                if source_str in assertion_mapping:
                    source_assertion = assertion_mapping[source_str]
                    mapped_sources.append(
                        {
                            "statement": source_assertion["statement"],
                            "score": source_assertion["score"],
                        }
                    )
                    # Aggregate source chunks from the original assertion
                    if source_assertion["sources"]:
                        aggregated_source_chunks.extend(source_assertion["sources"])
                else:
                    # Track hallucinated source IDs
                    hallucinated_sources.append(source_str)

        return mapped_sources, aggregated_source_chunks, hallucinated_sources

    def _validate_consolidated_assertions(
        self,
        consolidated_assertions: list[dict[str, Any]],
        assertion_mapping: dict[str, dict[str, Any]],
    ) -> list[Assertion]:
        """Validate and create Assertion objects from consolidated assertions.

        Args:
            consolidated_assertions: Raw consolidated assertions from LLM
            assertion_mapping: Mapping from assertion IDs to assertion data

        Returns
        -------
            List of validated Assertion objects
        """
        validated_assertions = []
        for assertion in consolidated_assertions:
            result = self._process_consolidated_assertion(assertion, assertion_mapping)
            if result:
                validated_assertions.append(result)
        return validated_assertions

    def _process_consolidated_assertion(
        self, assertion: dict[str, Any], assertion_mapping: dict[str, dict[str, Any]]
    ) -> Assertion | None:
        """Process a single consolidated assertion and return Assertion object or None if invalid."""
        statement = assertion.get("statement", "").strip()
        if not statement:
            return None

        sources = assertion.get("sources", [])
        mapped_sources, aggregated_chunks, hallucinated = (
            self._map_sources_and_aggregate(sources, assertion_mapping)
        )

        # Discard if all sources are hallucinated
        if not mapped_sources:
            if hallucinated:
                log.warning(
                    "Discarding assertion with all hallucinated sources: '%s...' (hallucinated: %s)",
                    statement[:100],
                    hallucinated,
                )
            return None

        # Log partial hallucinations
        if hallucinated:
            log.warning(
                "Assertion has %s hallucinated source(s): %s (keeping %s valid)",
                len(hallucinated),
                hallucinated,
                len(mapped_sources),
            )

        # Log missing source chunks
        if not aggregated_chunks:
            log.warning(
                "Global assertion has 0 source chunks: '%s...' (LLM sources: %s, mapping keys: %s)",
                statement[:100],
                sources,
                list(assertion_mapping.keys())[:10],
            )

        try:
            return Assertion(
                statement=statement,
                sources=aggregated_chunks,
                score=assertion.get("score", 5),
                reasoning=assertion.get("reasoning", ""),
                attributes={"source_assertions": mapped_sources},
            )
        except ValueError as e:
            log.warning("Skipping invalid consolidated assertion: %s", e)
            return None

    async def generate_reduce_response(
        self, question_text: str, reduce_context: tuple[list[Assertion], str]
    ) -> list[Assertion]:
        """Generate reduce response using LLM to consolidate assertions into high-level ones.

        Args:
            question_text: The question text for context
            reduce_context: Tuple of (unique_assertions, formatted_text) from build_reduce_context

        Returns
        -------
            List of consolidated high-level Assertion objects
        """
        unique_assertions, formatted_text = reduce_context

        if not unique_assertions:
            return []

        # Use LLM with reduce prompt to consolidate assertions
        log.info("REDUCE RESPONSE: Consolidating %s assertions", len(unique_assertions))

        # Build base prompt
        base_prompt = self.reduce_prompt.substitute(
            question_text=question_text, assertions_context=formatted_text
        )

        # Dynamically add count instruction if max_assertions is specified
        if self.max_assertions is not None and self.max_assertions > 0:
            count_instruction = self._max_assertion_instruction_prompt.substitute(
                max_assertions=self.max_assertions
            )
            prompt_content = base_prompt + "\n\n" + count_instruction
        else:
            prompt_content = base_prompt

        messages = [{"role": "user", "content": prompt_content}]

        result = await self.llm.chat(messages=messages, **self.llm_params)
        response, j = try_parse_json_object(result.output.content)

        if j == {}:
            log.warning(
                "Failed to parse consolidation response, returning original: %s",
                response,
            )
            return self._rank_and_limit_assertions(
                unique_assertions, self.max_assertions
            )

        parsed_result = json.loads(response)
        consolidated_assertions = parsed_result.get("assertions", [])

        if not consolidated_assertions:
            log.warning("No consolidated assertions returned, using original")
            return self._rank_and_limit_assertions(
                unique_assertions, self.max_assertions
            )

        # Build mapping from assertion IDs to assertion data
        assertion_mapping = self._build_assertion_mapping(unique_assertions)

        # Validate consolidated assertions with source mapping
        validated_assertions = self._validate_consolidated_assertions(
            consolidated_assertions, assertion_mapping
        )

        # rank and limit assertions
        validated_assertions = self._rank_and_limit_assertions(
            validated_assertions, self.max_assertions
        )

        log.info(
            "Successfully consolidated %s assertions into %s",
            len(unique_assertions),
            len(validated_assertions),
        )
        return validated_assertions
