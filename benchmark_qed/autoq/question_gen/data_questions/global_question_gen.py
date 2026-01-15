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
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.typing import (
    ClaimExtractionResult,
)
from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from string import Template

    import tiktoken

    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
    from benchmark_qed.autoq.config import AssertionConfig, AssertionPromptConfig
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

DATA_GLOBAL_PROMPTS_PATH = Path(global_questions.__file__).parent


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
        if question_sampler is not None:
            question_sampler.random_seed = self.random_seed
        else:
            question_sampler = QuestionSampler(
                sampler=MMRTextSampler(lambda_param=0.5),
                sampler_params={
                    "quality_attributes": [
                        "relevant_references_count",
                        "input_questions_count",
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
            # Create validator if validation is enabled
            validator = None
            if global_assertion_config.enable_validation:
                validator = AssertionValidator(
                    llm=llm,
                    llm_params=llm_params,
                    min_criterion_score=global_assertion_config.min_validation_score,
                    validation_prompt=assertion_prompt_config.global_validation_prompt.template,
                    concurrent_validations=global_assertion_config.concurrent_llm_calls,
                )

            self.assertion_generator = GlobalClaimAssertionGenerator(
                llm=llm,
                llm_params=llm_params,
                token_encoder=self.token_encoder,
                max_assertions=max_assertions,
                validator=validator,
                batch_size=global_assertion_config.batch_size,
                max_data_tokens=global_assertion_config.max_data_tokens,
                concurrent_coroutines=global_assertion_config.concurrent_llm_calls,
                max_concurrent_questions=global_assertion_config.max_concurrent_questions,
                map_system_prompt=assertion_prompt_config.global_assertion_map_prompt.template,
                reduce_system_prompt=assertion_prompt_config.global_assertion_reduce_prompt.template,
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
                f"Processing categories {i} to {min(i + self.concurrent_coroutines, len(question_contexts))} "
                f"of {len(question_contexts)} categories..."
            )
            batch_results = await tqdm_asyncio.gather(*[
                self._agenerate_single_chain(question_context=context)
                for context in batch
            ])
            batch_questions = [
                question for result in batch_results for question in result
            ]
            results.extend(batch_questions)
        log.info(f"Generated {len(results)} candidate questions from {len(self.local_questions)} local questions")

        # select a subset of questions if needed
        final_questions = self.select(candidate_questions=results, top_k=num_questions)

        # Generate assertions only for the final selected questions
        max_assertions = (
            self.assertion_config.global_.max_assertions
            if self.assertion_config
            else None
        )
        if (
            max_assertions is None or max_assertions > 0
        ) and self.assertion_generator is not None:
            log.info(
                "Generating assertions for %s final questions", len(final_questions)
            )
            await self.assertion_generator.agenerate_assertions_for_questions(
                final_questions
            )

        return QuestionGenResult(
            selected_questions=final_questions,
            candidate_questions=results,
        )

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
            f"Categories: {len(category_to_questions)} total, "
            f"{len(valid_categories)} valid (>={self.min_questions_in_context} questions)"
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
        rng = random.Random(self.random_seed)
        # Use log(count + 1) to handle edge cases and compress weights
        weights = [math.log(len(category_to_questions[c]) + 1) for c in valid_categories]
        sampled = rng.choices(valid_categories, weights=weights, k=num_questions)

        counts: dict[str, int] = defaultdict(int)
        for cat in sampled:
            counts[cat] += 1

        log.info(
            f"Weighted sampling (seed={self.random_seed}): "
            f"{len(counts)} unique categories from {len(valid_categories)}, "
            f"questions/category: {min(counts.values())}-{max(counts.values())}"
        )
        return counts

    def _even_distribute_categories(
        self, valid_categories: list[str], num_questions: int
    ) -> dict[str, int]:
        """Distribute questions evenly across all valid categories."""
        num_per_category = math.ceil(num_questions / len(valid_categories))
        log.info(f"Even distribution: {num_per_category} questions per category")
        return {cat: num_per_category for cat in valid_categories}

    def _create_context(
        self, category: str, num_to_generate: int, local_questions: list[str]
    ) -> DataGlobalQuestionContext:
        """Create a single question generation context."""
        questions_text = "\n".join(local_questions)
        context_text = (
            f"Category: {category}\n\n"
            f"Local questions:\n\n"
            f"{questions_text}\n\n---\n\n"
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
                unique_questions = await self._generate_unique_questions(question_context)
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
            log.exception(f"Exception for category: {question_context.category}")
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
                return await self.claim_extractor.aextract_claims(question, local_questions)

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
        for question, claims, embedding in zip(questions, claim_results, embeddings, strict=True):
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
                    },
                )
            )
        return results
