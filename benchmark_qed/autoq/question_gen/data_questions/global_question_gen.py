# Copyright (c) 2025 Microsoft Corporation.
"""Data-global question generation module."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autod.sampler.enums import ClusterRepresentativeSelectionType
from benchmark_qed.autod.sampler.sampling.kmeans_sampler import KmeansTextSampler
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
    ) -> None:
        # Import here to avoid circular imports
        from benchmark_qed.autoq.config import AssertionConfig, AssertionPromptConfig

        if claim_extractor_params is None:
            claim_extractor_params = {}
        if assertion_config is None:
            assertion_config = AssertionConfig()
        if assertion_prompt_config is None:
            assertion_prompt_config = AssertionPromptConfig()

        self.assertion_config = assertion_config
        self.random_seed = random_seed
        if question_sampler is not None:
            question_sampler.random_seed = self.random_seed
        else:
            question_sampler = QuestionSampler(
                sampler=KmeansTextSampler(),
                sampler_params={
                    "sample_selection": ClusterRepresentativeSelectionType.ATTRIBUTE_RANKING,
                    "ranking_attributes": [
                        "relevant_references_count",
                        "input_questions_count",
                    ],
                    "ascending": False,
                },
                random_seed=self.random_seed,
            )
        super().__init__(llm, llm_params, question_sampler)
        self.text_embedder = text_embedder
        self.token_encoder = token_encoder

        # Assertion generation setup (max_assertions != 0 enables it)
        self.assertion_generator: GlobalClaimAssertionGenerator | None = None
        max_assertions = assertion_config.max_assertions
        if max_assertions is None or max_assertions > 0:
            # Create validator if validation is enabled
            validator = None
            if assertion_config.enable_validation:
                validator = AssertionValidator(
                    llm=llm,
                    llm_params=llm_params,
                    min_criterion_score=assertion_config.min_validation_score,
                    validation_prompt=assertion_prompt_config.global_validation_prompt.template,
                )

            self.assertion_generator = GlobalClaimAssertionGenerator(
                llm=llm,
                llm_params=llm_params,
                token_encoder=self.token_encoder,
                max_assertions=max_assertions,
                validator=validator,
                batch_size=assertion_config.batch_size,
                max_data_tokens=assertion_config.max_data_tokens,
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
            msg = f"Processing categories {i} to {min(i + self.concurrent_coroutines, len(question_contexts))} of {len(question_contexts)} categories..."
            log.info(msg)
            batch = question_contexts[i : i + self.concurrent_coroutines]
            batch_results = await tqdm_asyncio.gather(*[
                self._agenerate_single_chain(question_context=context)
                for context in batch
            ])
            batch_questions = [
                question for result in batch_results for question in result
            ]
            results.extend(batch_questions)
        msg = f"Generated {len(results)} candidate questions from {len(self.local_questions)} local questions."
        log.info(msg)

        # select a subset of questions if needed
        final_questions = self.select(candidate_questions=results, top_k=num_questions)

        # Generate assertions only for the final selected questions
        max_assertions = (
            self.assertion_config.max_assertions if self.assertion_config else None
        )
        if (
            max_assertions is None or max_assertions > 0
        ) and self.assertion_generator is not None:
            await self._agenerate_assertions_for_questions(final_questions)

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
        category_to_question = defaultdict(list)
        for question in self.local_questions:
            if question.attributes and "abstract_categories" in question.attributes:
                categories = question.attributes["abstract_categories"]
                for category in categories:
                    category_to_question[category].append(question.text)

        # filter out categories with only one question and sort the categories by the number of questions
        sorted_cats = sorted(
            [c for c in category_to_question if len(category_to_question[c]) > 1],
            key=lambda x: len(category_to_question[x]),
            reverse=True,
        )

        # Calculate the number of questions per category
        num_questions_per_category = math.ceil(num_questions / len(sorted_cats))
        msg = (
            f"Number of initial categories: {len(category_to_question)}\n"
            f"Number of valid candidate categories (i.e. categories with more than one input question): {len(sorted_cats)}\n"
            f"Number of questions to generate per candidate category: {num_questions_per_category}"
        )
        log.info(msg)

        contexts: list[DataGlobalQuestionContext] = []
        for category in sorted_cats:
            context_text = (
                f"Category: {category}"
                + "\n\nLocal questions:\n\n"
                + "\n".join(category_to_question[category])
                + "\n\n---\n\n"
            )
            contexts.append(
                DataGlobalQuestionContext(
                    category=category,
                    local_questions=category_to_question[category],
                    num_generated_questions=num_questions_per_category,
                    context_text=context_text,
                )
            )

        return contexts

    async def _agenerate_single_chain(
        self,
        question_context: DataGlobalQuestionContext,
    ) -> list[Question]:
        """Generate questions for a single input text."""
        try:
            async with self.semaphore:
                # get an initial set of question from the input text
                extraction_messages = [
                    {
                        "role": "system",
                        "content": self.extraction_prompt.substitute(
                            num_questions=question_context.num_generated_questions
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

                results: list[Question] = []
                question_set = set()
                for question in parsed_questions["questions"]:
                    if question in question_set:
                        msg = f"Duplicate question to be filtered out: {question}"
                        log.info(msg)
                        continue
                    question_set.add(question)

                    # extract claims for the question
                    claim_extraction_results = (
                        await self.claim_extractor.aextract_claims(
                            question, question_context.local_questions
                        )
                    )

                    question_attributes = {
                        "abstract_categories": question_context.category,
                        "claims": claim_extraction_results.claims,
                        "claim_count": len(claim_extraction_results.claims),
                        "reference_coverage": claim_extraction_results.reference_coverage,
                        "relevant_references_count": claim_extraction_results.relevant_references_count,
                        "input_questions_count": len(question_context.local_questions),
                    }

                    # Initialize empty assertions - will be generated later for final questions only
                    question_attributes.update({
                        "assertions": [],
                        "assertion_count": 0,
                    })

                    results.append(
                        Question(
                            id=str(uuid.uuid4()),
                            text=question,
                            question_type=QuestionType.DATA_GLOBAL,
                            embedding=await self.text_embedder.embed_raw_text(question),
                            references=question_context.local_questions,
                            attributes=question_attributes,
                        )
                    )
                return results

        except Exception:
            msg = f"Exception occurred while generating questions for category: {question_context.category}"
            log.exception(msg)
            return []

    async def _agenerate_assertions_for_questions(
        self, questions: list[Question]
    ) -> None:
        """Generate assertions for a list of questions and update their attributes."""
        if not self.assertion_generator:
            return

        log.info("Generating assertions for %s final questions", len(questions))

        for question in tqdm(questions, desc="Generating assertions", unit="question"):
            try:
                # Get claims from question attributes
                claims = (
                    question.attributes.get("claims", []) if question.attributes else []
                )
                if not claims:
                    log.warning("No claims found for question: %s", question.text)
                    continue

                assertion_result = await self.assertion_generator.agenerate_assertions(
                    question_text=question.text,
                    claims=claims,
                )
                assertions = [
                    asdict(assertion) for assertion in assertion_result.assertions
                ]

                # Update question attributes with assertions
                if question.attributes is None:
                    question.attributes = {}
                question.attributes["assertions"] = assertions
                question.attributes["assertion_count"] = len(assertions)

                log.info(
                    "Generated %s assertions for question: %s",
                    len(assertions),
                    question.text,
                )

            except Exception as e:  # noqa: BLE001
                log.warning(
                    "Failed to generate assertions for question '%s': %s",
                    question.text,
                    e,
                )
                # Ensure attributes exist even on failure
                if question.attributes is None:
                    question.attributes = {}
                question.attributes["assertions"] = []
                question.attributes["assertion_count"] = 0
