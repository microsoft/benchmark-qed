# Copyright (c) 2025 Microsoft Corporation.
"""Data-local question generation module."""

import asyncio
from dataclasses import asdict
import json
import logging
import math
import uuid
from pathlib import Path
from string import Template
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from benchmark_qed.autoq.question_gen.data_questions.assertion_gen.base import Assertion
import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
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
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.local_claim_extractor import (
    DataLocalClaimExtractor,
)
from benchmark_qed.autoq.question_gen.data_questions.assertion_gen import (
    LocalClaimAssertionGenerator,
)
from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

DATA_LOCAL_PROMPTS_PATH = Path(local_questions.__file__).parent


class DataLocalQuestionGen(BaseQuestionGen):
    """
    Generate local data questions for a given dataset.
    
    Supports optional assertion generation after claim extraction to create
    testable facts that can be used for answer accuracy evaluation. Set max_assertions
    to None or 0 to skip assertion generation, or to a positive integer to limit
    the number of assertions per question.
    """

    def __init__(
        self,
        llm: ChatModel,
        text_embedder: TextEmbedder,
        text_units: list[TextUnit],
        question_sampler: QuestionSampler | None = None,
        claim_extractor_params: dict[str, Any] | None = None,
        assertion_generator_params: dict[str, Any] | None = None,
        max_assertions: int = 5,
        llm_params: dict[str, Any] = defs.LLM_PARAMS,
        json_mode: bool = True,
        generation_system_prompt: Template | None = None,
        generation_user_prompt: Template | None = None,
        expansion_system_prompt: Template | None = None,
        concurrent_coroutines: int = 32,
        random_seed: int = defs.RANDOM_SEED,
    ) -> None:
        if claim_extractor_params is None:
            claim_extractor_params = {}
        if assertion_generator_params is None:
            assertion_generator_params = {}
            
        self.max_assertions = max_assertions
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

        # Initialize assertion generator if enabled
        self.assertion_generator: LocalClaimAssertionGenerator | None = None
        if self.max_assertions is None or self.max_assertions > 0:
            # Pass max_assertions to the assertion generator (None means unlimited)
            assertion_generator_params["max_assertions"] = self.max_assertions
            self.assertion_generator = LocalClaimAssertionGenerator(
                llm=llm, **assertion_generator_params
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
        if self.max_assertions > 0 and self.assertion_generator is not None:
            await self._agenerate_assertions_for_questions(final_questions)
        
        return QuestionGenResult(
            selected_questions=final_questions,
            candidate_questions=results,
        )

    async def _agenerate_assertions_for_questions(self, questions: list[Question]) -> None:
        """Generate assertions for a list of questions and update their attributes."""
        if not self.assertion_generator:
            return
            
        log.info(f"Generating assertions for {len(questions)} final questions")
        
        for question in questions:
            try:
                # Get claims from question attributes
                claims = question.attributes.get("claims", []) if question.attributes else []
                if not claims:
                    log.warning(f"No claims found for question: {question.text}")
                    continue
                    
                assertion_result = await self.assertion_generator.agenerate_assertions(
                    question_text=question.text,
                    claims=claims
                )
                assertions = [asdict(assertion) for assertion in assertion_result.assertions]
                
                # Update question attributes with assertions
                if question.attributes is None:
                    question.attributes = {}
                question.attributes["assertions"] = assertions
                question.attributes["assertion_count"] = len(assertions)
                
                log.info(f"Generated {len(assertions)} assertions for question: {question.text}")
                
            except Exception as e:
                log.warning(f"Failed to generate assertions for question '{question.text}': {e}")
                # Ensure attributes exist even on failure
                if question.attributes is None:
                    question.attributes = {}
                question.attributes["assertions"] = []
                question.attributes["assertion_count"] = 0

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
                                    "assertions": assertions if assertions else [],
                                    "assertion_count": len(assertions) if assertions else 0,
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
