# Copyright (c) 2025 Microsoft Corporation.
"""Data-global question generation module."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.asyncio import tqdm_asyncio

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autod.sampler.sampling.mmr_sampler import MMRTextSampler
from benchmark_qed.autoq.data_model.enums import QuestionType
from benchmark_qed.autoq.data_model.question import Question
from benchmark_qed.autoq.prompts.data_questions import global_questions
from benchmark_qed.autoq.question_gen.base import BaseQuestionGen, QuestionGenResult
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen import (
    AssertionValidator,
    GlobalClaimAssertionGenerator,
)
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.global_claim_extractor import (
    DataGlobalClaimExtractor,
)
from benchmark_qed.autoq.question_gen.data_questions.question_validator import (
    GlobalQuestionValidator,
)
from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from string import Template

    import tiktoken

    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
    from benchmark_qed.autoq.config import AssertionConfig, AssertionPromptConfig
    from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.typing import (
        ClaimExtractionResult,
    )
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

DATA_GLOBAL_PROMPTS_PATH = Path(global_questions.__file__).parent


def soft_filter_questions(
    questions: list[Question],
    attribute_name: str,
    min_threshold: float,
    min_needed: int,
) -> list[Question]:
    """Apply soft filtering to questions based on an attribute threshold.

    Prefers questions meeting the threshold, but falls back to include
    below-threshold questions (sorted by attribute value) if needed.

    Args:
        questions: List of questions to filter.
        attribute_name: Name of the attribute to filter on.
        min_threshold: Minimum value required for the attribute.
        min_needed: Minimum number of questions needed in output.

    Returns
    -------
        Filtered list of questions, prioritizing those meeting the threshold.

    """
    above_threshold = [
        q
        for q in questions
        if q.attributes and q.attributes.get(attribute_name, 0) >= min_threshold
    ]
    below_threshold = [
        q
        for q in questions
        if q.attributes and q.attributes.get(attribute_name, 0) < min_threshold
    ]

    if len(above_threshold) >= min_needed:
        # Enough high-quality candidates - use only those
        log.info(
            "Found %s questions with %s >= %s (threshold met, need %s)",
            len(above_threshold),
            attribute_name,
            min_threshold,
            min_needed,
        )
        return above_threshold

    if len(above_threshold) > 0:
        # Not enough high-quality, supplement with best below-threshold
        below_threshold.sort(
            key=lambda q: q.attributes.get(attribute_name, 0) if q.attributes else 0,
            reverse=True,
        )
        needed = min_needed - len(above_threshold)
        log.warning(
            "Only %s questions meet %s >= %s (need %s). "
            "Including %s questions below threshold (sorted by value).",
            len(above_threshold),
            attribute_name,
            min_threshold,
            min_needed,
            min(needed, len(below_threshold)),
        )
        return above_threshold + below_threshold[:needed]

    # No questions meet threshold, sort by attribute and take best
    log.warning(
        "No questions meet %s >= %s. Selecting top %s by highest value instead.",
        attribute_name,
        min_threshold,
        min_needed,
    )
    questions_sorted = sorted(
        questions,
        key=lambda q: q.attributes.get(attribute_name, 0) if q.attributes else 0,
        reverse=True,
    )
    return questions_sorted[:min_needed]


@dataclass
class DataGlobalQuestionContext:
    """Data class for storing the context for generating global data questions."""

    category: str
    local_questions: list[str]
    context_text: str
    num_generated_questions: int = 5


class DataGlobalQuestionGen(BaseQuestionGen):
    """
    Generate data-global questions for a given dataset from a set of local questions.

    Supports optional assertion generation after claim extraction to create
    testable facts that can be used for answer accuracy evaluation. Configure
    assertion generation using the assertion_config parameter.
    """

    def __init__(
        self,
        llm: ChatModel,
        text_embedder: TextEmbedder,
        local_questions: list[Question],
        token_encoder: tiktoken.Encoding | None = None,
        question_sampler: QuestionSampler | None = None,
        claim_extractor_params: dict[str, Any] | None = None,
        assertion_config: AssertionConfig | None = None,
        assertion_prompt_config: AssertionPromptConfig | None = None,
        llm_params: dict[str, Any] = defs.LLM_PARAMS,
        json_mode: bool = True,
        generation_system_prompt: Template | None = None,
        generation_user_prompt: Template | None = None,
        concurrent_coroutines: int = 32,
        random_seed: int = defs.RANDOM_SEED,
        use_weighted_sampling: bool = True,
        min_questions_in_context: int = defs.MIN_QUESTIONS_IN_CONTEXT,
        min_claim_count: int = defs.MIN_CLAIM_COUNT,
        min_relevant_reference_count: int = defs.MIN_RELEVANT_REFERENCE_COUNT,
        enable_question_validation: bool = True,
    ) -> None:
        from benchmark_qed.autoq.config import AssertionConfig, AssertionPromptConfig

        if claim_extractor_params is None:
            claim_extractor_params = {}
        if assertion_config is None:
            assertion_config = AssertionConfig()
        if assertion_prompt_config is None:
            assertion_prompt_config = AssertionPromptConfig()

        self.assertion_config = assertion_config
        self.random_seed = random_seed
        self.use_weighted_sampling = use_weighted_sampling
        self.min_questions_in_context = min_questions_in_context
        self.min_claim_count = min_claim_count
        self.min_relevant_reference_count = min_relevant_reference_count
        if question_sampler is not None:
            question_sampler.random_seed = self.random_seed
        else:
            # Assertions are now generated BEFORE selection, so we can use assertion-based metrics
            question_sampler = QuestionSampler(
                sampler=MMRTextSampler(lambda_param=0.5),
                sampler_params={
                    "quality_attributes": [
                        "assertion_count",  # Number of validated assertions
                        "unique_source_count",  # Unique sources from assertions
                        "total_map_source_count",  # Total corpus coverage
                        "relevant_references_count",  # How many local questions contributed
                    ],
                },
                random_seed=self.random_seed,
            )
        super().__init__(llm, llm_params, question_sampler)
        self.text_embedder = text_embedder
        self.token_encoder = token_encoder

        # Assertion generation setup (max_assertions != 0 enables it)
        self.assertion_generator: GlobalClaimAssertionGenerator | None = None
        global_assertion_config = assertion_config.global_
        max_assertions = global_assertion_config.max_assertions
        if max_assertions is None or max_assertions > 0:
            # Create validators if validation is enabled
            # Map validator uses local prompt (fact-focused) for map assertions
            # Reduce validator uses global prompt (thematic) for reduce assertions
            map_validator = None
            reduce_validator = None
            if global_assertion_config.enable_validation:
                # Map validator for factual map assertions
                map_validator = AssertionValidator(
                    llm=llm,
                    llm_params=llm_params,
                    min_criterion_score=global_assertion_config.min_validation_score,
                    validation_prompt=assertion_prompt_config.local_validation_prompt.template,
                    concurrent_validations=global_assertion_config.concurrent_llm_calls,
                    max_source_count=global_assertion_config.max_source_count,
                )
                # Reduce validator for thematic reduce assertions
                reduce_validator = AssertionValidator(
                    llm=llm,
                    llm_params=llm_params,
                    min_criterion_score=global_assertion_config.min_validation_score,
                    validation_prompt=assertion_prompt_config.global_validation_prompt.template,
                    concurrent_validations=global_assertion_config.concurrent_llm_calls,
                    max_source_count=global_assertion_config.max_source_count,
                )

            self.assertion_generator = GlobalClaimAssertionGenerator(
                llm=llm,
                llm_params=llm_params,
                token_encoder=self.token_encoder,
                max_assertions=max_assertions,
                map_validator=map_validator,
                reduce_validator=reduce_validator,
                batch_size=global_assertion_config.batch_size,
                reduce_data_tokens=global_assertion_config.reduce_data_tokens,
                map_data_tokens=global_assertion_config.map_data_tokens,
                concurrent_coroutines=global_assertion_config.concurrent_llm_calls,
                max_concurrent_questions=global_assertion_config.max_concurrent_questions,
                map_system_prompt=assertion_prompt_config.global_assertion_map_prompt.template,
                reduce_system_prompt=assertion_prompt_config.global_assertion_reduce_prompt.template,
                text_embedder=text_embedder,
                enable_semantic_grouping=global_assertion_config.enable_semantic_grouping,
                validate_map_assertions=global_assertion_config.validate_map_assertions,
                validate_reduce_assertions=global_assertion_config.validate_reduce_assertions,
            )

        # Question validation setup
        self.question_validator: GlobalQuestionValidator | None = None
        if enable_question_validation:
            self.question_validator = GlobalQuestionValidator(
                llm=llm,
                llm_params=llm_params,
            )

        self.json_mode = json_mode
        if json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

        self.extraction_prompt: Template = (
            generation_system_prompt
            or load_template_file(
                DATA_GLOBAL_PROMPTS_PATH / "data_global_gen_system_prompt.txt"
            )
        )
        self.extraction_input_prompt: Template = (
            generation_user_prompt
            or load_template_file(
                DATA_GLOBAL_PROMPTS_PATH / "data_global_gen_user_prompt.txt"
            )
        )

        self.local_questions = local_questions
        self.concurrent_coroutines = concurrent_coroutines
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self.concurrent_coroutines
        )

        self.claim_extractor_params = claim_extractor_params
        self.claim_extractor: DataGlobalClaimExtractor = DataGlobalClaimExtractor(
            llm=llm, local_questions=local_questions, **claim_extractor_params
        )

    async def agenerate(
        self,
        num_questions: int = defs.NUM_QUESTIONS,
        oversample_factor: float = defs.OVERSAMPLE_FACTOR,
    ) -> QuestionGenResult:
        """Async function to generate data-global questions from a set of pre-generated local questions."""
        num_candidate_questions = math.ceil(num_questions * oversample_factor)
        question_contexts = self._generate_question_context(num_candidate_questions)

        results: list[Question] = []
        for i in range(0, len(question_contexts), self.concurrent_coroutines):
            batch = question_contexts[i : i + self.concurrent_coroutines]
            log.info(
                "Processing categories %s to %s of %s categories...",
                i,
                min(i + self.concurrent_coroutines, len(question_contexts)),
                len(question_contexts),
            )
            batch_results = await tqdm_asyncio.gather(*[
                self._agenerate_single_chain(question_context=context)
                for context in batch
            ])
            batch_questions = [
                question for result in batch_results for question in result
            ]
            results.extend(batch_questions)
        log.info(
            "Generated %s candidate questions from %s local questions",
            len(results),
            len(self.local_questions),
        )

        # Step 1: Hard filter by minimum claim count
        pre_filter_count = len(results)
        results = [
            q
            for q in results
            if q.attributes
            and q.attributes.get("claim_count", 0) >= self.min_claim_count
        ]
        if pre_filter_count != len(results):
            log.info(
                "Filtered to %s questions with claim_count >= %s (removed %s)",
                len(results),
                self.min_claim_count,
                pre_filter_count - len(results),
            )

        # Step 2: Soft filter by relevant_references_count (from claims, before assertions)
        # This reduces the number of questions we validate and generate assertions for
        results = soft_filter_questions(
            questions=results,
            attribute_name="relevant_references_count",
            min_threshold=self.min_relevant_reference_count,
            min_needed=num_questions
            * 2,  # Keep 2x to allow for validation/assertion failures
        )

        # Step 3: Validate question quality BEFORE assertion generation (saves LLM costs)
        if self.question_validator is not None:
            log.info("Validating %s questions for quality...", len(results))
            pre_validation_count = len(results)
            results = await self.question_validator.filter_valid_questions(results)
            if pre_validation_count != len(results):
                log.info(
                    "Filtered to %s questions after validation (removed %s)",
                    len(results),
                    pre_validation_count - len(results),
                )

        # Step 4: Generate assertions for validated candidates
        max_assertions = (
            self.assertion_config.global_.max_assertions
            if self.assertion_config
            else None
        )
        if (
            max_assertions is None or max_assertions > 0
        ) and self.assertion_generator is not None:
            log.info("Generating assertions for %s candidate questions", len(results))
            await self.assertion_generator.agenerate_assertions_for_questions(results)

        # Step 5: Compute unique source count (from assertions) for all candidates
        self._compute_source_coverage(results)

        # Step 6: Filter out questions with no valid assertions
        pre_filter_count = len(results)
        results = [
            q
            for q in results
            if q.attributes and q.attributes.get("assertion_count", 0) > 0
        ]
        if pre_filter_count != len(results):
            log.info(
                "Filtered to %s questions with assertions (removed %s with no assertions)",
                len(results),
                pre_filter_count - len(results),
            )

        # Step 7: Select best questions using assertion-based quality metrics
        selected_questions = self.select(
            candidate_questions=results, top_k=num_questions
        )
        log.info(
            "Selected %s questions (from %s candidates)",
            len(selected_questions),
            len(results),
        )

        return QuestionGenResult(
            selected_questions=selected_questions,
            candidate_questions=results,
        )

    def _compute_source_coverage(self, questions: list[Question]) -> None:
        """Compute source coverage metrics for each question.

        Computes two metrics:
        1. unique_source_count: Sources from final (reduced) assertions only.
           Measures how well-grounded the selected assertions are.
        2. total_map_source_count: Sources from ALL map assertions.
           Measures total corpus coverage the question touches.

        Args:
            questions: List of questions to compute source coverage for.
                       Each question should have assertions and map_assertions in attributes.
        """
        for question in questions:
            if question.attributes is None:
                continue

            # Metric 1: Count unique sources from final assertions only
            assertions = question.attributes.get("assertions", [])
            final_sources = {
                source
                for assertion in assertions
                for source in assertion.get("sources", [])
                if isinstance(source, str)
            }
            question.attributes["unique_source_count"] = len(final_sources)

            # Metric 2: Count unique sources from ALL map assertions
            map_assertions = question.attributes.get("map_assertions", [])
            map_sources = {
                source
                for assertion in map_assertions
                for source in assertion.get("sources", [])
                if isinstance(source, str)
            }
            question.attributes["total_map_source_count"] = len(map_sources)

    def _generate_question_context(
        self,
        num_questions: int = 50,
    ) -> list[DataGlobalQuestionContext]:
        """
        Generate the context for the question generation.

        Each context consists of a set of local questions sharing the same abstract category.
        """
        category_to_questions = self._group_questions_by_category()
        valid_categories = [
            c
            for c in category_to_questions
            if len(category_to_questions[c]) >= self.min_questions_in_context
        ]

        log.info(
            "Categories: %s total, %s valid (>=%s questions)",
            len(category_to_questions),
            len(valid_categories),
            self.min_questions_in_context,
        )

        category_counts = self._compute_category_counts(
            valid_categories, category_to_questions, num_questions
        )

        return [
            self._create_context(category, count, category_to_questions[category])
            for category, count in category_counts.items()
        ]

    def _group_questions_by_category(self) -> dict[str, list[str]]:
        """Group local questions by their abstract categories."""
        category_to_questions: dict[str, list[str]] = defaultdict(list)
        for question in self.local_questions:
            if question.attributes and "abstract_categories" in question.attributes:
                for category in question.attributes["abstract_categories"]:
                    category_to_questions[category].append(question.text)
        return category_to_questions

    def _compute_category_counts(
        self,
        valid_categories: list[str],
        category_to_questions: dict[str, list[str]],
        num_questions: int,
    ) -> dict[str, int]:
        """Determine how many questions to generate per category."""
        if self.use_weighted_sampling:
            return self._weighted_sample_categories(
                valid_categories, category_to_questions, num_questions
            )
        return self._even_distribute_categories(valid_categories, num_questions)

    def _weighted_sample_categories(
        self,
        valid_categories: list[str],
        category_to_questions: dict[str, list[str]],
        num_questions: int,
    ) -> dict[str, int]:
        """Sample categories with probability proportional to log(question_count)."""
        rng = random.Random(self.random_seed)  # noqa: S311
        # Use log(count + 1) to handle edge cases and compress weights
        weights = [
            math.log(len(category_to_questions[c]) + 1) for c in valid_categories
        ]
        sampled = rng.choices(valid_categories, weights=weights, k=num_questions)

        counts: dict[str, int] = defaultdict(int)
        for cat in sampled:
            counts[cat] += 1

        log.info(
            "Weighted sampling (seed=%s): %s unique categories from %s, questions/category: %s-%s",
            self.random_seed,
            len(counts),
            len(valid_categories),
            min(counts.values()),
            max(counts.values()),
        )
        return counts

    def _even_distribute_categories(
        self, valid_categories: list[str], num_questions: int
    ) -> dict[str, int]:
        """Distribute questions evenly across all valid categories."""
        num_per_category = math.ceil(num_questions / len(valid_categories))
        log.info("Even distribution: %s questions per category", num_per_category)
        return dict.fromkeys(valid_categories, num_per_category)

    def _create_context(
        self, category: str, num_to_generate: int, local_questions: list[str]
    ) -> DataGlobalQuestionContext:
        """Create a single question generation context."""
        questions_text = "\n".join(local_questions)
        context_text = (
            f"Category: {category}\n\nLocal questions:\n\n{questions_text}\n\n---\n\n"
        )
        return DataGlobalQuestionContext(
            category=category,
            local_questions=local_questions,
            num_generated_questions=num_to_generate,
            context_text=context_text,
        )

    async def _agenerate_single_chain(
        self,
        question_context: DataGlobalQuestionContext,
    ) -> list[Question]:
        """Generate questions for a single input text."""
        try:
            async with self.semaphore:
                # Generate questions from LLM
                unique_questions = await self._generate_unique_questions(
                    question_context
                )
                if not unique_questions:
                    return []

                # Extract claims and embeddings in parallel
                claim_results, embeddings = await self._extract_question_metadata(
                    unique_questions, question_context.local_questions
                )

                # Build Question objects
                return self._assemble_question_objects(
                    unique_questions, claim_results, embeddings, question_context
                )

        except Exception:
            log.exception("Exception for category: %s", question_context.category)
            return []

    async def _generate_unique_questions(
        self, question_context: DataGlobalQuestionContext
    ) -> list[str]:
        """Generate questions from LLM and deduplicate."""
        extraction_messages = [
            {
                "role": "system",
                "content": self.extraction_prompt.substitute(
                    num_questions=question_context.num_generated_questions,
                ),
            },
            {
                "role": "user",
                "content": self.extraction_input_prompt.substitute(
                    input_text=question_context.context_text,
                    num_questions=question_context.num_generated_questions,
                ),
            },
        ]
        questions_result = await self.llm.chat(
            messages=extraction_messages, **self.llm_params
        )
        questions, j = try_parse_json_object(questions_result.output.content)
        if j == {}:
            msg = f"Error parsing JSON response: {questions}"
            log.error(msg)
            return []

        parsed_questions = json.loads(questions)

        # Deduplicate
        unique_questions = []
        question_set: set[str] = set()
        for question in parsed_questions["questions"]:
            if question not in question_set:
                question_set.add(question)
                unique_questions.append(question)
        return unique_questions

    async def _extract_question_metadata(
        self, questions: list[str], local_questions: list[str]
    ) -> tuple[list[ClaimExtractionResult], list[list[float]]]:
        """Extract claims and compute embeddings for all questions in parallel."""
        claim_semaphore = asyncio.Semaphore(self.concurrent_coroutines)

        async def extract_claims(question: str) -> ClaimExtractionResult:
            async with claim_semaphore:
                return await self.claim_extractor.aextract_claims(
                    question, local_questions
                )

        claim_tasks = [extract_claims(q) for q in questions]
        embedding_tasks = [self.text_embedder.embed_raw_text(q) for q in questions]

        claim_results, embeddings = await asyncio.gather(
            asyncio.gather(*claim_tasks),
            asyncio.gather(*embedding_tasks),
        )
        return claim_results, embeddings

    def _assemble_question_objects(
        self,
        questions: list[str],
        claim_results: list[ClaimExtractionResult],
        embeddings: list[list[float]],
        question_context: DataGlobalQuestionContext,
    ) -> list[Question]:
        """Assemble Question objects from generated questions and metadata."""
        results: list[Question] = []
        for question, claims, embedding in zip(
            questions, claim_results, embeddings, strict=True
        ):
            results.append(
                Question(
                    id=str(uuid.uuid4()),
                    text=question,
                    question_type=QuestionType.DATA_GLOBAL,
                    embedding=embedding,
                    references=question_context.local_questions,
                    attributes={
                        "abstract_categories": question_context.category,
                        "claims": claims.claims,
                        "claim_count": len(claims.claims),
                        "reference_coverage": claims.reference_coverage,
                        "relevant_references_count": claims.relevant_references_count,
                        "input_questions_count": len(question_context.local_questions),
                        "assertions": [],
                        "assertion_count": 0,
                        "unique_source_count": 0,  # Updated after assertion generation
                        "total_map_source_count": 0,  # Updated after assertion generation
                    },
                )
            )
        return results
