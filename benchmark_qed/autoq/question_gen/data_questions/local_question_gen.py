# Copyright (c) 2025 Microsoft Corporation.
"""Data-local question generation module."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm.asyncio import tqdm_asyncio

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autod.sampler.enums import ClusterRepresentativeSelectionType
from benchmark_qed.autod.sampler.neighboring.semantic_neighbors import (
    compute_intra_inter_references_similarity_ratio,
    compute_similarity_to_references,
)
from benchmark_qed.autod.sampler.sample_gen import load_texts_to_clusters
from benchmark_qed.autod.sampler.sampling.kmeans_sampler import KmeansTextSampler
from benchmark_qed.autoq.data_model.enums import QuestionType
from benchmark_qed.autoq.data_model.question import Question
from benchmark_qed.autoq.prompts.data_questions import local_questions
from benchmark_qed.autoq.question_gen.base import BaseQuestionGen, QuestionGenResult
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen import (
    AssertionValidator,
    LocalClaimAssertionGenerator,
)
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.local_claim_extractor import (
    DataLocalClaimExtractor,
)
from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
from benchmark_qed.config.utils import load_template_file

if TYPE_CHECKING:
    from string import Template

    from benchmark_qed.autod.data_model.text_unit import TextUnit
    from benchmark_qed.autod.data_processor.embedding import TextEmbedder
    from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
    from benchmark_qed.autoq.config import AssertionConfig, AssertionPromptConfig
    from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

DATA_LOCAL_PROMPTS_PATH = Path(local_questions.__file__).parent


class DataLocalQuestionGen(BaseQuestionGen):
    """
    Generate local data questions for a given dataset.

    Supports optional assertion generation after claim extraction to create
    testable facts that can be used for answer accuracy evaluation. Configure
    assertion generation using the assertion_config parameter.
    """

    def __init__(
        self,
        llm: ChatModel,
        text_embedder: TextEmbedder,
        text_units: list[TextUnit],
        question_sampler: QuestionSampler | None = None,
        claim_extractor_params: dict[str, Any] | None = None,
        assertion_config: AssertionConfig | None = None,
        assertion_prompt_config: AssertionPromptConfig | None = None,
        llm_params: dict[str, Any] = defs.LLM_PARAMS,
        json_mode: bool = True,
        generation_system_prompt: Template | None = None,
        generation_user_prompt: Template | None = None,
        expansion_system_prompt: Template | None = None,
        concurrent_coroutines: int = 32,
        random_seed: int = defs.RANDOM_SEED,
    ) -> None:
        """
        Initialize the DataLocalQuestionGen.

        Parameters
        ----------
        llm : ChatModel
            The language model to use.
        text_embedder : TextEmbedder
            Text embedder for computing embeddings.
        text_units : list[TextUnit]
            The text units to generate questions from.
        question_sampler : QuestionSampler | None
            Optional sampler for selecting questions.
        claim_extractor_params : dict[str, Any] | None
            Parameters for the claim extractor.
        assertion_config : AssertionConfig | None
            Configuration for assertion generation and validation. If None, uses default
            configuration from AssertionConfig(). Set max_assertions to 0 to disable.
        assertion_prompt_config : AssertionPromptConfig | None
            Custom prompts for assertion generation. If None, uses default prompts.
        llm_params : dict[str, Any]
            Parameters for the LLM.
        json_mode : bool
            Whether to use JSON mode.
        generation_system_prompt : Template | None
            Custom system prompt for question generation.
        generation_user_prompt : Template | None
            Custom user prompt for question generation.
        expansion_system_prompt : Template | None
            Custom prompt for question expansion.
        concurrent_coroutines : int
            Number of concurrent coroutines.
        random_seed : int
            Random seed for reproducibility.
        """
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
            question_sampler.random_seed = random_seed
        else:
            question_sampler = QuestionSampler(
                sampler=KmeansTextSampler(),
                sampler_params={
                    "num_samples_per_cluster": 1,
                    "representative_selection": ClusterRepresentativeSelectionType.ATTRIBUTE_RANKING,
                    "ranking_attributes": [
                        "intra_inter_similarity_ratio",
                        "reference_coverage",
                    ],
                    "ascending": False,
                },
                random_seed=self.random_seed,
            )
        super().__init__(llm, llm_params, question_sampler)

        self.claim_extractor_params = claim_extractor_params
        self.claim_extractor: DataLocalClaimExtractor = DataLocalClaimExtractor(
            llm=llm, **claim_extractor_params
        )

        # Initialize assertion generator if enabled (max_assertions != 0)
        self.assertion_generator: LocalClaimAssertionGenerator | None = None
        local_assertion_config = assertion_config.local
        max_assertions = local_assertion_config.max_assertions
        if max_assertions is None or max_assertions > 0:
            # Create validator if validation is enabled
            validator = None
            if local_assertion_config.enable_validation:
                validator = AssertionValidator(
                    llm=llm,
                    llm_params=llm_params,
                    min_criterion_score=local_assertion_config.min_validation_score,
                    validation_prompt=assertion_prompt_config.local_validation_prompt.template,
                    concurrent_validations=local_assertion_config.concurrent_llm_calls,
                )

            self.assertion_generator = LocalClaimAssertionGenerator(
                llm=llm,
                llm_params=llm_params,
                max_assertions=max_assertions,
                validator=validator,
                system_prompt=assertion_prompt_config.local_assertion_gen_prompt.template,
                max_concurrent_questions=local_assertion_config.max_concurrent_questions,
            )

        self.json_mode = json_mode
        if json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

        self.extraction_prompt: str = (
            generation_system_prompt
            or load_template_file(
                DATA_LOCAL_PROMPTS_PATH / "data_local_gen_system_prompt.txt"
            )
        ).template
        self.text_input_prompt: Template = generation_user_prompt or load_template_file(
            DATA_LOCAL_PROMPTS_PATH / "data_local_gen_user_prompt.txt"
        )
        self.generation_prompt: str = (
            expansion_system_prompt
            or load_template_file(
                DATA_LOCAL_PROMPTS_PATH / "data_local_expansion_system_prompt.txt"
            )
        ).template
        self.text_units = text_units
        self.text_embedder = text_embedder
        self.concurrent_coroutines = concurrent_coroutines
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self.concurrent_coroutines
        )
        self.random_seed: int = random_seed

    async def agenerate(
        self,
        num_questions: int = defs.NUM_QUESTIONS,
        oversample_factor: float = defs.OVERSAMPLE_FACTOR,
    ) -> QuestionGenResult:
        """Async function to generate data-local questions from the given sample of clustered text units."""
        text_clusters = load_texts_to_clusters(self.text_units)
        num_candidate_questions = math.ceil(num_questions * oversample_factor)
        questions_per_cluster = max(
            math.ceil(num_candidate_questions / len(text_clusters)), 1
        )

        results: list[Question] = []
        for i in range(0, len(text_clusters), self.concurrent_coroutines):
            msg = f"Processing clusters {i} to {min(i + self.concurrent_coroutines, len(text_clusters))} of {len(text_clusters)} clusters..."
            log.info(msg)
            batch = text_clusters[i : i + self.concurrent_coroutines]
            batch_results = await tqdm_asyncio.gather(*[
                self._agenerate_local_questions(
                    text_cluster=cluster,
                    num_questions=questions_per_cluster,
                )
                for cluster in batch
            ])
            batch_questions = [
                question for result in batch_results for question in result
            ]
            results.extend(batch_questions)

        # select a subset of questions if needed
        msg = f"Generated {len(results)} candidate questions from {len(text_clusters)} clusters."
        log.info(msg)
        final_questions = self.select(candidate_questions=results, top_k=num_questions)

        # Generate assertions only for the final selected questions
        max_assertions = (
            self.assertion_config.local.max_assertions
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

    async def _agenerate_local_questions(
        self,
        text_cluster: TextCluster,
        num_questions: int = 5,
    ) -> list[Question]:
        """Generate local questions for each text cluster in the dataset."""
        async with self.semaphore:
            return await self._agenerate_single_chain(
                text_units=text_cluster.text_units,
                num_questions=num_questions,
            )

    async def _agenerate_single_chain(
        self,
        text_units: list[TextUnit],
        num_questions: int,
    ) -> list[Question]:
        """Generate questions for a single input text cluster using a chain of thought process."""
        results: list[Question] = []
        try:
            # get an initial set of question from the input text
            input_list = [
                f"Input Text {unit.short_id or index + 1}: {unit.text}"
                for index, unit in enumerate(text_units)
            ]
            input_text = "\n\n".join(input_list)
            extraction_messages = [
                {"role": "system", "content": self.extraction_prompt},
                {
                    "role": "user",
                    "content": self.text_input_prompt.substitute(
                        input_text=input_text, num_questions=num_questions
                    ),
                },
            ]
            initial_questions_result = await self.llm.chat(
                messages=extraction_messages, **self.llm_params
            )

            # expand the original set of questions with additional information
            generation_messages = [
                *extraction_messages,
                {
                    "role": "assistant",
                    "content": initial_questions_result.output.content,
                },
                {"role": "user", "content": self.generation_prompt},
            ]

            final_questions_result = await self.llm.chat(
                messages=generation_messages, **self.llm_params
            )

            final_questions, j = try_parse_json_object(
                final_questions_result.output.content
            )
            if j == {}:
                msg = f"Error parsing questions output: {final_questions}"
                log.error(msg)
                return []

            parsed_final_questions = json.loads(final_questions)

            question_set = set()
            for question in parsed_final_questions["questions"]:
                # extract claims and generate reference for each question
                question_text = question.get("output_question", "")
                if question_text.strip() != "":
                    if question_text in question_set:
                        msg = f"Duplicate question: {question_text}"
                        log.warning(msg)
                        continue
                    question_set.add(question_text.lower().strip())
                    claims = await self.claim_extractor.aextract_claims(
                        question_text=question_text,
                        question_references=text_units,
                    )

                    if claims.reference_coverage > 0:
                        # Initialize empty assertions - will be generated later for final questions only
                        assertions = []

                        # calculate the similarity of the question to the references
                        (
                            question_embedding,
                            reference_similarity,
                            intra_inter_similarity_ratio,
                        ) = await self.calculate_question_similarity(
                            question_text=question_text, text_units=text_units
                        )
                        msg = f"Question: {question_text}. Intra-inter Similarity: {intra_inter_similarity_ratio}. Reference Coverage: {claims.reference_coverage}"
                        log.info(msg)

                        results.append(
                            Question(
                                id=str(uuid.uuid4()),
                                text=question_text,
                                question_type=QuestionType.DATA_LOCAL,
                                embedding=question_embedding,
                                references=input_list,
                                attributes={
                                    "input_question": question.get(
                                        "input_question", ""
                                    ),
                                    "period": question.get("period", ""),
                                    "location": question.get("location", ""),
                                    "named_entities": question.get(
                                        "named_entities", ""
                                    ),
                                    "abstract_categories": question.get(
                                        "abstract_categories", ""
                                    ),
                                    "background_information": question.get(
                                        "background_information", ""
                                    ),
                                    "reference_coverage": claims.reference_coverage,
                                    "relevant_reference_count": claims.relevant_references_count,
                                    "reference_count": len(input_list),
                                    "min_reference_similarity": reference_similarity.get(
                                        "min_similarity", 0
                                    ),
                                    "max_reference_similarity": reference_similarity.get(
                                        "max_similarity", 0
                                    ),
                                    "mean_reference_similarity": reference_similarity.get(
                                        "mean_similarity", 0
                                    ),
                                    "intra_inter_similarity_ratio": intra_inter_similarity_ratio,
                                    "claim_count": len(claims.claims),
                                    "claims": claims.claims,
                                    "assertions": assertions or [],
                                    "assertion_count": len(assertions)
                                    if assertions
                                    else 0,
                                },
                            )
                        )

        except Exception:
            log.exception("Exception occurred while generating questions")
            return []
        else:
            return results

    async def calculate_question_similarity(
        self,
        question_text: str,
        text_units: list[TextUnit],
    ) -> tuple[list[float], dict[str, float], float]:
        """Calculate two similarity measures.

        - reference_similarity:
        """
        hashed_text_units = {unit.id: unit for unit in text_units}
        question_embedding = await self.text_embedder.embed_raw_text(question_text)
        reference_similarity = compute_similarity_to_references(
            text_embedding=question_embedding, references=text_units
        )
        intra_inter_similarity_ratio = compute_intra_inter_references_similarity_ratio(
            text_embedding=question_embedding,
            in_references=text_units,
            out_references=[
                unit for unit in self.text_units if unit.id not in hashed_text_units
            ],
        )
        return question_embedding, reference_similarity, intra_inter_similarity_ratio
