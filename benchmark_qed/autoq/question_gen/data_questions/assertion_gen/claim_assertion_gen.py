# Copyright (c) 2025 Microsoft Corporation.
"""Generate assertions for evaluating answer accuracy based on claims."""

import asyncio
import json
import logging
from pathlib import Path
from string import Template
from typing import Any

from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autoq.prompts import data_questions
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.base import (
    AssertionGenerationResult,
    BaseAssertionGenerator,
)
from benchmark_qed.config.defaults import LLM_PARAMS
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

ASSERTION_GEN_PROMPTS_PATH = Path(data_questions.__file__).parent


class ClaimAssertionGenerator(BaseAssertionGenerator):
    """
    Generate factual assertions for evaluating answer accuracy based on claims.
    
    Takes a question and a list of relevant claims as input, and generates testable assertions
    that can be used as unit tests to verify the accuracy of answers to the question.
    
    Supports automatic batching for large claim sets:
    - If batch_size is None or <= 0, processes all claims in a single request
    - If claims exceed batch_size, automatically splits into parallel batches
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        system_prompt: Template | None = None,
        batch_size: int | None = 10,
        concurrent_coroutines: int = 8,
    ) -> None:
        system_prompt = system_prompt or load_template_file(
            ASSERTION_GEN_PROMPTS_PATH / "claim_assertion_gen_system_prompt.txt"
        )
        super().__init__(llm, llm_params, json_mode, system_prompt)
        self.batch_size = batch_size
        self.concurrent_coroutines = concurrent_coroutines
        self._semaphore = asyncio.Semaphore(concurrent_coroutines)

    async def agenerate_assertions(
        self, question_text: str, **kwargs: Any
    ) -> AssertionGenerationResult:
        """Generate assertions for evaluating answer accuracy based on a question and relevant claims."""
        claims: list[dict[str, Any]] = kwargs.get("claims", [])
        
        # Check if batching should be used
        should_batch = (
            self.batch_size is not None 
            and self.batch_size > 0 
            and len(claims) > self.batch_size
        )
        
        if not should_batch:
            return await self._process_single_batch(question_text, claims)
        
        # Process claims in batches
        return await self._process_batched_claims(question_text, claims)

    async def _process_single_batch(
        self, question_text: str, claims: list[dict[str, Any]]
    ) -> AssertionGenerationResult:
        """Process a single batch of claims."""
        claims_text = _format_claims_for_prompt(claims)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.substitute(
                    query=question_text, context_data=claims_text
                ),
            },
        ]
        result = await self.llm.chat(messages=messages, **self.llm_params)
        response, j = try_parse_json_object(result.output.content)
        if j == {}:
            msg = f"Invalid json response, returning empty assertion list: {response}"
            log.warning(msg)
            return AssertionGenerationResult(assertions=[], total_assertions=0)
        
        parsed_assertions = json.loads(response).get("assertions")
        if not parsed_assertions or not isinstance(parsed_assertions, list):
            log.warning("No assertions found in the response, returning empty assertion list")
            return AssertionGenerationResult(assertions=[], total_assertions=0)

        # Validate and clean the assertions
        validated_assertions = self._validate_assertions(parsed_assertions, claims=claims)
        return AssertionGenerationResult(
            assertions=validated_assertions,
            total_assertions=len(validated_assertions),
        )

    def _validate_assertions(self, parsed_assertions: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        """
        Validate assertions and map claim IDs to actual claim statements.
        
        Args:
            parsed_assertions: Raw assertions from LLM response
            **kwargs: Additional parameters, expects 'claims' for claim mapping
        
        Returns:
            List of validated assertions with mapped claim sources
        """
        claims = kwargs.get("claims", [])
        validated_assertions = []
        
        # Create claim ID to statement mapping if claims are provided
        claim_id_to_statement = {}
        if claims:
            for i, claim in enumerate(claims):
                claim_id = f"claim_{i+1}"
                claim_id_to_statement[claim_id] = claim.get("statement", "")
        
        for assertion in parsed_assertions:
            if (
                assertion.get("statement", "").strip() != ""
                and isinstance(assertion.get("score", 0), int)
                and 0 < assertion.get("score", 0) <= 100
            ):
                # Map claim IDs to actual claim statements in sources
                sources = assertion.get("sources", [])
                mapped_sources = []
                
                if claim_id_to_statement and sources:
                    for source in sources:
                        source_str = str(source).strip()
                        if source_str in claim_id_to_statement:
                            mapped_sources.append({
                                "claim_id": source_str,
                                "statement": claim_id_to_statement[source_str]
                            })
                        else:
                            # Keep original source if not found in mapping
                            mapped_sources.append(source_str)
                else:
                    # If no claims provided or no sources, keep original sources
                    mapped_sources = sources
                
                validated_assertion = {
                    "statement": assertion.get("statement"),
                    "sources": mapped_sources,
                    "score": assertion.get("score", 50),
                }
                validated_assertions.append(validated_assertion)
        return validated_assertions

    async def _process_batched_claims(
        self, question_text: str, claims: list[dict[str, Any]]
    ) -> AssertionGenerationResult:
        """Process claims in batches with parallel execution."""
        # batch_size is guaranteed to be a positive int here due to should_batch check
        assert self.batch_size is not None and self.batch_size > 0
        batch_size = self.batch_size
        
        # Split claims into batches
        claim_batches = [
            claims[i:i + batch_size] 
            for i in range(0, len(claims), batch_size)
        ]
        
        log.info(f"Processing {len(claims)} claims in {len(claim_batches)} batches")
        
        async def process_batch_with_semaphore(batch: list[dict[str, Any]]) -> AssertionGenerationResult:
            async with self._semaphore:
                return await self._process_single_batch(question_text, batch)
        
        # Process batches in parallel
        batch_results = await asyncio.gather(
            *[process_batch_with_semaphore(batch) for batch in claim_batches],
            return_exceptions=True
        )
        
        # Combine results from all batches
        all_assertions = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                log.error(f"Error processing batch {i}: {result}")
                continue
            if isinstance(result, AssertionGenerationResult):
                all_assertions.extend(result.assertions)
        
        # Remove duplicate assertions based on statement content
        # Note: At this point, sources are already mapped to claim statements from individual batches
        unique_assertions = _deduplicate_assertions(all_assertions)
        
        return AssertionGenerationResult(
            assertions=unique_assertions,
            total_assertions=len(unique_assertions),
        )

    async def agenerate_assertions_from_claims(
        self, question_text: str, claims: list[dict[str, Any]]
    ) -> AssertionGenerationResult:
        """Convenience method to generate assertions from claims with a cleaner API."""
        return await self.agenerate_assertions(question_text, claims=claims)


def _format_claims_for_prompt(claims: list[dict[str, Any]]) -> str:
    """Format claims list for the assertion generation prompt."""
    if not claims:
        return "No claims provided."
    
    formatted_claims = []
    for i, claim in enumerate(claims):
        claim_id = f"claim_{i+1}"
        statement = claim.get("statement", "")
        score = claim.get("score", 0)
        
        formatted_claim = f"ID: {claim_id}\nStatement: {statement}\nImportance Score: {score}\n"
        formatted_claims.append(formatted_claim)
    
    return "\n".join(formatted_claims)


def _deduplicate_assertions(assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate assertions based on statement content."""
    seen_statements = set()
    unique_assertions = []
    
    for assertion in assertions:
        statement = assertion.get("statement", "").strip().lower()
        if statement and statement not in seen_statements:
            seen_statements.add(statement)
            unique_assertions.append(assertion)
    
    return unique_assertions
