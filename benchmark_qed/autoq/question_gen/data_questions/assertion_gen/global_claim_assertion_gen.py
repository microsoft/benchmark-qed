# Copyright (c) 2025 Microsoft Corporation.
"""Generate assertions for evaluating answer accuracy based on claims for global questions."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.text_utils import (
    num_tokens,
    try_parse_json_object,
)
from benchmark_qed.autod.sampler.clustering.constraint_kmeans import (
    ConstraintKmeansClustering,
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
    ASSERTION_MAP_DATA_TOKENS,
    ASSERTION_REDUCE_DATA_TOKENS,
    LLM_PARAMS,
    MAX_ASSERTIONS,
)
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
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

    Supports semantic claim grouping (optional):
    - When `enable_semantic_grouping=True` and `text_embedder` is provided, claims are grouped
      by semantic similarity using KMeans clustering before the map step
    - This reduces redundancy by ensuring similar claims are processed together
    - Results in more consolidated map assertions and better reduce step quality
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
        reduce_data_tokens: int = ASSERTION_REDUCE_DATA_TOKENS,
        map_data_tokens: int = ASSERTION_MAP_DATA_TOKENS,
        token_encoder: Any | None = None,
        map_validator: AssertionValidator | None = None,
        reduce_validator: AssertionValidator | None = None,
        max_concurrent_questions: int | None = None,
        text_embedder: TextEmbedder | None = None,
        enable_semantic_grouping: bool = False,
        validate_map_assertions: bool = False,
        validate_reduce_assertions: bool = True,
    ) -> None:
        super().__init__(
            llm,
            llm_params,
            json_mode,
            max_assertions,
            reduce_validator,  # Pass reduce_validator to base class as 'validator'
            max_concurrent_questions,
        )

        # Store separate validator for map assertions
        # Map assertions are factual -> use local_validation_prompt (fact-focused)
        # Reduce assertions are thematic -> use global_validation_prompt (via base class)
        self.map_validator = map_validator

        # Load prompt templates
        self.map_prompt: Template = map_system_prompt or load_template_file(
            ASSERTION_GEN_PROMPTS_PATH
            / "assertions"
            / "global_claim_assertion_map_prompt.txt"
        )
        if isinstance(self.map_prompt, str):
            self.map_prompt = Template(self.map_prompt)

        self.reduce_prompt: Template = reduce_system_prompt or load_template_file(
            ASSERTION_GEN_PROMPTS_PATH
            / "assertions"
            / "global_claim_assertion_reduce_prompt.txt"
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
        self.reduce_data_tokens = reduce_data_tokens
        self.map_data_tokens = map_data_tokens
        self.token_encoder = token_encoder

        # Semantic grouping parameters for improved claim batching
        self.text_embedder = text_embedder
        self.enable_semantic_grouping = (
            enable_semantic_grouping and text_embedder is not None
        )

        # Map assertion validation (optional, defaults to False)
        self.validate_map_assertions = validate_map_assertions

        # Reduce assertion validation (optional, defaults to True)
        self.validate_reduce_assertions = validate_reduce_assertions

        # Local generator for map phase processing
        # Uses map prompt to generate specific factual assertions
        # The reduce step will consolidate these into thematic global assertions
        self.local_generator: LocalClaimAssertionGenerator = (
            LocalClaimAssertionGenerator(
                llm=self.llm,
                llm_params=self.llm_params,
                json_mode=self.json_mode,
                system_prompt=self.map_prompt,
                max_assertions=None,  # No assertion limiting in map step
            )
        )

    async def agenerate_assertions(
        self, question_text: str, **kwargs: Any
    ) -> AssertionGenerationResult:
        """Generate assertions using a map-reduce approach for global questions.

        Map Step: Divide claims into batches and generate initial assertions
        Deduplication Step: Remove duplicate assertions from map step
        Reduce Step: Consolidate initial assertions into high-level global assertions
        Validation Step (optional): Validate assertions if validator is configured

        Args:
            question_text: The question text to generate assertions for
            **kwargs: Additional parameters, expects 'claims' list (can be simple or batch format)

        Returns
        -------
            AssertionGenerationResult containing:
            - assertions: Final consolidated global assertions
            - total_assertions: Number of final assertions
            - map_assertions: Intermediate assertions from map step (after deduplication)
        """
        # MAP PHASE - use async version when semantic grouping is enabled
        claim_batches = await self.abuild_map_context(kwargs.get("claims", []))
        if not claim_batches:
            log.warning("No claims provided for assertion generation")
            return AssertionGenerationResult(
                assertions=[], total_assertions=0, map_assertions=[]
            )

        map_responses = await self.generate_map_responses(question_text, claim_batches)

        # Flatten all map assertions
        all_map_assertions: list[Assertion] = []
        for assertion_list in map_responses:
            all_map_assertions.extend(assertion_list)

        # Debug: Check map assertions for empty sources
        empty_source_count = sum(1 for a in all_map_assertions if not a.sources)
        if empty_source_count > 0:
            log.warning(
                "MAP ASSERTIONS: %s/%s map assertions have EMPTY sources!",
                empty_source_count,
                len(all_map_assertions),
            )

        # DEDUPLICATION - Naive approach for now (string matching) to remove duplicate map assertions before reduce
        all_map_assertions = self._deduplicate_map_assertions(all_map_assertions)

        # MAP VALIDATION PHASE (optional) - validate map assertions before reduce
        if self.validate_map_assertions:
            log.info(
                "MAP VALIDATION: Validating %s map assertions", len(all_map_assertions)
            )
            all_map_assertions = await self._validate_map_assertions(
                all_map_assertions, question_text
            )
            log.info(
                "MAP VALIDATION: %s map assertions passed validation",
                len(all_map_assertions),
            )

        # REDUCE PHASE - use deduplicated (and optionally validated) assertions
        reduce_context = self.build_reduce_context_from_assertions(all_map_assertions)
        final_assertions = await self.generate_reduce_response(
            question_text, reduce_context
        )

        # REDUCE VALIDATION PHASE (optional, default True)
        if self.validate_reduce_assertions:
            final_assertions = await self._validate_assertions(
                final_assertions, question_text
            )

        return AssertionGenerationResult(
            assertions=final_assertions,
            total_assertions=len(final_assertions),
            map_assertions=all_map_assertions,
        )

    async def abuild_map_context(
        self, claims: list[ClaimDict]
    ) -> list[list[ClaimDict]]:
        """Build map context by extracting claims and dividing into batches for parallel processing.

        Uses semantic grouping if enabled to group similar claims together,
        which reduces redundancy in map assertions.

        Args:
            claims: Entire list of claims

        Returns
        -------
            List of claim batches, where each batch is a list of claims
        """
        if not claims:
            return []

        # Use semantic grouping if enabled, otherwise sequential batching
        if self.enable_semantic_grouping:
            return await self._group_claims_semantically(claims)

        # Default: create batches based on batch_size (sequential)
        return self._create_sequential_batches(claims)

    def _create_sequential_batches(
        self, claims: list[ClaimDict]
    ) -> list[list[ClaimDict]]:
        """Create sequential batches from claims based on batch_size.

        Args:
            claims: List of claims to batch

        Returns
        -------
            List of claim batches
        """
        batch_size = (
            self.batch_size if self.batch_size and self.batch_size > 0 else len(claims)
        )

        # Split claims into batches
        batches = [
            claims[i : i + batch_size] for i in range(0, len(claims), batch_size)
        ]
        log.info(
            "MAP CONTEXT: Created %s batches from %s claims (sequential)",
            len(batches),
            len(claims),
        )
        return batches

    async def _group_claims_semantically(
        self, claims: list[ClaimDict]
    ) -> list[list[ClaimDict]]:
        """Group semantically similar claims using constraint-based KMeans clustering.

        This method converts claims to TextUnits, embeds them efficiently using batch
        embedding, and clusters them using ConstraintKmeansClustering which ensures
        each cluster stays within a specified token limit.

        This reduces redundancy in the map step by ensuring similar claims are processed
        together, leading to more consolidated map assertions.

        Args:
            claims: List of claims to group

        Returns
        -------
            List of claim batches grouped by semantic similarity with token constraints
        """
        if not claims or self.text_embedder is None:
            return self._create_sequential_batches(claims)

        log.info("SEMANTIC GROUPING: Converting %s claims to TextUnits", len(claims))

        # Convert claims to TextUnits with unique IDs for tracking
        claim_text_units: list[TextUnit] = []
        claim_id_mapping: dict[str, ClaimDict] = {}

        for i, claim in enumerate(claims):
            claim_id = f"claim_{i}"
            statement = claim.get("statement", "")
            text_unit = TextUnit(
                id=claim_id,
                short_id=claim_id,
                text=statement,
                attributes={"original_claim": claim, "claim_index": i},
            )
            claim_text_units.append(text_unit)
            claim_id_mapping[claim_id] = claim

        # Embed all claims efficiently using batch embedding
        log.info(
            "SEMANTIC GROUPING: Embedding %s claims in batch", len(claim_text_units)
        )
        embedded_text_units = await self.text_embedder.embed_batch(claim_text_units)

        # Use ConstraintKmeansClustering to group claims with token limit
        # map_data_tokens controls the maximum size of each cluster in the map step
        clusterer = ConstraintKmeansClustering(token_encoder=self.token_encoder)

        log.info(
            "SEMANTIC GROUPING: Clustering claims with max %s tokens per cluster",
            self.map_data_tokens,
        )

        text_clusters = clusterer.cluster(
            text_units=embedded_text_units,
            max_cluster_token_size=self.map_data_tokens,
        )

        # Convert TextClusters back to claim batches
        batches: list[list[ClaimDict]] = []
        for cluster in text_clusters:
            cluster_claims = []
            for text_unit in cluster.text_units:
                if text_unit.id and text_unit.id in claim_id_mapping:
                    cluster_claims.append(claim_id_mapping[text_unit.id])
                elif text_unit.attributes and "original_claim" in text_unit.attributes:
                    cluster_claims.append(text_unit.attributes["original_claim"])
            if cluster_claims:
                batches.append(cluster_claims)

        # Log cluster sizes and token counts
        cluster_sizes = [len(batch) for batch in batches]
        log.info(
            "SEMANTIC GROUPING: Created %s semantic groups with sizes: %s",
            len(batches),
            cluster_sizes,
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

    async def _validate_map_assertions(
        self,
        assertions: list[Assertion],
        question_text: str,
    ) -> list[Assertion]:
        """Validate map assertions using the map_validator.

        Map assertions are factual and should be validated with a fact-focused
        prompt (local_validation_prompt).

        Args:
            assertions: List of map assertions to validate
            question_text: The question text for context in validation

        Returns
        -------
            List of assertions that passed validation, or all assertions if no validator
        """
        if not self.map_validator or not assertions:
            return assertions

        log.info("Validating %s map assertions...", len(assertions))
        summary = await self.map_validator.validate_assertions(
            assertions, question_text
        )

        log.info(
            "Map validation complete: %s/%s assertions passed (%.1f%%)",
            summary.valid_count,
            summary.total_count,
            summary.validation_rate * 100,
        )

        return summary.valid_assertions

    def _deduplicate_map_assertions(
        self, assertions: list[Assertion]
    ) -> list[Assertion]:
        """Remove duplicate assertions based on normalized statement text.

        When duplicates are found:
        - Combines sources from all duplicates into a unique list
        - Keeps the highest score among duplicates

        Args:
            assertions: List of assertions to deduplicate

        Returns
        -------
            List of unique assertions with combined sources
        """
        if not assertions:
            return []

        original_count = len(assertions)

        # Group assertions by normalized statement (lowercase, stripped)
        statement_to_assertions: dict[str, list[Assertion]] = {}
        for assertion in assertions:
            normalized = assertion.statement.lower().strip()
            if normalized not in statement_to_assertions:
                statement_to_assertions[normalized] = []
            statement_to_assertions[normalized].append(assertion)

        # For each group, merge into a single assertion with combined sources
        unique_assertions: list[Assertion] = []
        for group in statement_to_assertions.values():
            if len(group) == 1:
                unique_assertions.append(group[0])
            else:
                # Combine all sources into a unique list
                all_sources: list[str] = []
                seen_sources: set[str] = set()
                for assertion in group:
                    for source in assertion.sources:
                        if source not in seen_sources:
                            all_sources.append(source)
                            seen_sources.add(source)

                # Get the highest score among duplicates
                best_score = max(a.score for a in group)

                # Create merged assertion with combined sources
                # Use first assertion's statement (they're all the same after normalization)
                merged = Assertion(
                    statement=group[0].statement,
                    sources=all_sources,
                    score=best_score,
                )
                unique_assertions.append(merged)

        log.info(
            "DEDUPLICATION: Reduced %s map assertions to %s unique assertions",
            original_count,
            len(unique_assertions),
        )
        return unique_assertions

    def build_reduce_context_from_assertions(
        self, assertions: list[Assertion]
    ) -> tuple[list[Assertion], str]:
        """Build reduce context from a flat list of assertions.

        Args:
            assertions: Flat list of assertions (already deduplicated)

        Returns
        -------
            Tuple of (selected_assertions, formatted_assertions_text) for reduce phase
        """
        if not assertions:
            return [], ""

        log.info(
            "REDUCE CONTEXT: Processing %s assertions",
            len(assertions),
        )

        # Rank assertions by score (descending) and source count (descending)
        ranked_assertions = sorted(
            assertions, key=lambda a: (-a.score, -len(a.sources))
        )

        log.info(
            "REDUCE CONTEXT: Ranked %s assertions by score and source count",
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
            if current_tokens + assertion_tokens > self.reduce_data_tokens:
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
            self.reduce_data_tokens,
        )
        return selected_assertions, formatted_text

    def build_reduce_context(
        self, map_responses: list[list[Assertion]]
    ) -> tuple[list[Assertion], str]:
        """Build reduce context by merging map responses and formatting for consolidation.

        Note: This method is kept for backward compatibility. New code should use
        build_reduce_context_from_assertions after deduplication.

        Args:
            map_responses: List of assertion lists from map phase

        Returns
        -------
            Tuple of (unique_assertions, formatted_assertions_text) for reduce phase
        """
        # Flatten all initial assertions
        initial_assertions: list[Assertion] = []
        for assertion_list in map_responses:
            initial_assertions.extend(assertion_list)

        return self.build_reduce_context_from_assertions(initial_assertions)

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
            # Debug: log assertions with empty sources
            if not assertion.sources:
                log.warning(
                    "ASSERTION MAPPING: Map assertion %s has EMPTY sources: '%s...'",
                    assertion_id,
                    assertion.statement[:80],
                )
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
                    mapped_sources.append({
                        "statement": source_assertion["statement"],
                        "score": source_assertion["score"],
                    })
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
            final_assertion = Assertion(
                statement=statement,
                sources=aggregated_chunks,
                score=assertion.get("score", 5),
                reasoning=assertion.get("reasoning", ""),
                attributes={"supporting_assertions": mapped_sources},
            )
        except ValueError as e:
            log.warning("Skipping invalid consolidated assertion: %s", e)
            return None
        else:
            # Debug: log if final assertion has empty sources
            if not aggregated_chunks:
                log.warning(
                    "FINAL ASSERTION: Created with EMPTY sources: '%s...' "
                    "(mapped %s supporting assertions)",
                    statement[:80],
                    len(mapped_sources),
                )
            return final_assertion

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
