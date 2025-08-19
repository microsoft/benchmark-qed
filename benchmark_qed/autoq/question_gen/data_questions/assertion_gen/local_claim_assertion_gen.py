# Copyright (c) 2025 Microsoft Corporation.
"""Generate assertions for evaluating answer accuracy based on claims for local questions."""

import json
import logging
from pathlib import Path
from string import Template
from typing import Any

from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autoq.prompts import data_questions
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.base import (
    BaseAssertionGenerator,
    AssertionGenerationResult,
    Assertion,
    ClaimDict,
)
from benchmark_qed.config.defaults import LLM_PARAMS
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

ASSERTION_GEN_PROMPTS_PATH = Path(data_questions.__file__).parent


class LocalClaimAssertionGenerator(BaseAssertionGenerator):
    """
    Generate factual assertions for evaluating answer accuracy based on claims for local questions.
    
    This generator is designed for data_local_questions and handles simple claim lists.
    Takes a question and a list of relevant claims as input, and generates testable assertions
    that can be used as unit tests to verify the accuracy of answers to the question.
    
    Optimized for simple, direct processing without complex batching overhead.
    """

    def __init__(
        self,
        llm: ChatModel,
        llm_params: dict[str, Any] = LLM_PARAMS,
        json_mode: bool = True,
        system_prompt: Template | None = None,
        max_assertions: int | None = 5,
    ) -> None:
        super().__init__(llm, llm_params, json_mode, max_assertions)

        system_prompt = system_prompt if system_prompt else load_template_file(
            ASSERTION_GEN_PROMPTS_PATH / "assertions" / "local_claim_assertion_gen_prompt.txt"
        )
        if isinstance(system_prompt, str):
            system_prompt = Template(system_prompt)
        self.system_prompt = system_prompt
        
        # Load max assertion instruction template for dynamic count limiting
        self._max_assertion_instruction_prompt = load_template_file(
            ASSERTION_GEN_PROMPTS_PATH / "assertions" / "local_max_assertion_instruction_prompt.txt"
        )

    async def agenerate_assertions(
        self, question_text: str, **kwargs: Any
    ) -> AssertionGenerationResult:
        """Generate assertions for data local questions."""
        claims: list[ClaimDict] = kwargs.get("claims", [])
        if not claims:
            log.warning("No claims provided for assertion generation")
            return AssertionGenerationResult(assertions=[], total_assertions=0)
        
        # Process all claims in a single batch for local questions
        claims_text = self._build_context(claims)
        
        if self.system_prompt is None:
            raise ValueError("System prompt cannot be None")
        
        # Build base prompt
        base_prompt = self.system_prompt.substitute(
            query=question_text, 
            context_data=claims_text
        )
        
        # Dynamically add count instruction if max_assertions is specified
        if self.max_assertions is not None and self.max_assertions > 0:
            count_instruction = self._max_assertion_instruction_prompt.substitute(max_assertions=self.max_assertions)
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
        log.debug(f"Assertion results: {result}")
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
        
        # Apply ranking and limiting (None means no limit)
        validated_assertions = self._rank_and_limit_assertions(validated_assertions, self.max_assertions)
        
        return AssertionGenerationResult(
            assertions=validated_assertions,
            total_assertions=len(validated_assertions),
        )

    def _validate_assertions(self, parsed_assertions: list[dict[str, Any]], **kwargs: Any) -> list[Assertion]:
        """
        Validate assertions and create Assertion objects with claim ID mapping.
        
        Args:
            parsed_assertions: Raw assertions from LLM response as dictionaries
            **kwargs: Additional parameters, expects 'claims' for claim mapping
        
        Returns:
            List of validated Assertion objects with mapped claim sources
        """
        claims = kwargs.get("claims", [])
        validated_assertions = []
        
        # Create claim ID to statement mapping if claims are provided
        claim_id_to_text = {}
        if claims:
            for i, claim in enumerate(claims):
                claim_id = f"claim_{i+1}"
                claim_id_to_text[claim_id] = claim
        
        for assertion in parsed_assertions:
            statement = assertion.get("statement", "").strip()
            score = assertion.get("score", 5)
            
            if (
                statement != ""
                and isinstance(score, int)
                and 1 <= score <= 10
            ):
                # Map claim IDs to actual claim text
                sources = assertion.get("sources", [])
                source_claim_texts = []
                source_chunks: set[str] = set() # source chunks of the associated claims
                
                if claim_id_to_text and sources:
                    for source in sources:
                        source_str = str(source).strip()
                        if source_str in claim_id_to_text:
                            source_claim_texts.append(claim_id_to_text[source_str].get("statement", ""))

                            # add sources of claims to source_chunks
                            claim_sources = claim_id_to_text[source_str].get("sources", [])
                            source_chunks.update([source["text"] for source in claim_sources])
                        else:
                            # Keep original source if not found in mapping
                            source_claim_texts.append(source_str)
                else:
                    # If no claims provided or no sources, keep original sources
                    source_claim_texts = sources
                
                # Create Assertion object with enhanced attributes
                try:
                    assertion_obj = Assertion(
                        statement=statement,
                        score=score,
                        sources=list(source_chunks),
                        attributes={
                            **assertion.get("attributes", {}),
                            "source_claims": source_claim_texts,
                        }
                    )
                    # Debug logging for source counts
                    source_count = len(source_chunks) if source_chunks else 0
                    if source_count == 0:
                        log.warning(f"Local assertion created with 0 sources: '{statement[:100]}...'")
                        log.debug(f"  Original sources: {sources}")
                        log.debug(f"  Source claim texts: {source_claim_texts}")
                        log.debug(f"  Source chunks: {source_chunks}")
                    else:
                        log.debug(f"Local assertion created with {source_count} sources")
                    
                    validated_assertions.append(assertion_obj)
                except ValueError as e:
                    log.warning(f"Skipping invalid assertion: {e}")
                    continue
        
        return validated_assertions

    @staticmethod
    def _build_context(claims: list[ClaimDict]) -> str:
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
