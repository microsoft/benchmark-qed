# Copyright (c) 2025 Microsoft Corporation.

import asyncio
import json
import logging
import math
import uuid
from typing import Any

from tqdm.asyncio import tqdm_asyncio

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
from benchmark_qed.autoq.prompts.data_questions.local_question_prompts import (
    LOCAL_EXTRACTION_PROMPT,
    LOCAL_GENERATION_PROMPT,
    LOCAL_TEXT_INPUT_PROMPT,
)
from benchmark_qed.autoq.question_gen.base import BaseQuestionGen, QuestionGenResult
from benchmark_qed.autoq.question_gen.data_questions.claim_extractor.local_claim_extractor import (
    DataLocalClaimExtractor,
)
from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)


class DataLocalQuestionGen(BaseQuestionGen):
    """Generate local data questions for a given dataset"""

    def __init__(
        self,
        llm: ChatModel,
        text_embedder: TextEmbedder,
        text_units: list[TextUnit],
        question_sampler: QuestionSampler | None = None,
        claim_extractor_params: dict[str, Any] = {},
        llm_params: dict[str, Any] = defs.LLM_PARAMS,
        json_mode: bool = True,
        extraction_prompt: str = LOCAL_EXTRACTION_PROMPT,
        text_input_prompt: str = LOCAL_TEXT_INPUT_PROMPT,
        generation_prompt: str = LOCAL_GENERATION_PROMPT,
        concurrent_coroutines: int = 32,
        random_seed: int = defs.RANDOM_SEED,
    ):
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

        self.json_mode = json_mode
        if json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

        self.extraction_prompt = extraction_prompt
        self.text_input_prompt = text_input_prompt
        self.generation_prompt = generation_prompt
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
            log.info(
                f"Processing clusters {i} to {min(i + self.concurrent_coroutines, len(text_clusters))} of {len(text_clusters)} clusters..."
            )
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
        log.info(
            f"Generated {len(results)} candidate questions from {len(text_clusters)} clusters."
        )
        final_questions = self.select(candidate_questions=results, top_k=num_questions)
        return QuestionGenResult(
            selected_questions=final_questions,
            candidate_questions=results,
        )

    async def _agenerate_local_questions(
        self,
        text_cluster: TextCluster,
        num_questions: int = 5,
    ) -> list[Question]:
        """Generates local questions for each text cluster in the dataset"""
        async with self.semaphore:
            results = await self._agenerate_single_chain(
                text_units=text_cluster.text_units,
                num_questions=num_questions,
            )
            return results

    async def _agenerate_single_chain(
        self,
        text_units: list[TextUnit],
        num_questions: int,
    ) -> list[Question]:
        """Generates questions for a single input text cluster using a chain of thought process"""
        try:
            # get an initial set of question from the input text
            input_list = [
                f"Input Text {unit.short_id if unit.short_id else index + 1}: {unit.text}"
                for index, unit in enumerate(text_units)
            ]
            input_text = "\n\n".join(input_list)
            extraction_messages = [
                {"role": "system", "content": self.extraction_prompt},
                {
                    "role": "user",
                    "content": self.text_input_prompt.format(
                        input_text=input_text, num_questions=num_questions
                    ),
                },
            ]
            initial_questions_result = await self.llm.chat(
                messages=extraction_messages, **self.llm_params
            )

            # expand the original set of questions with additional information
            generation_messages = extraction_messages + [
                {
                    "role": "assistant",
                    "content": initial_questions_result.output.content,
                },
                {"role": "user", "content": self.generation_prompt},
            ]

            final_questions_result = await self.llm.chat(
                messages=generation_messages, **self.llm_params
            )

            final_questions, _j = try_parse_json_object(
                final_questions_result.output.content
            )
            if _j == {}:
                log.error(f"Error parsing questions output: {final_questions}")
                return []

            parsed_final_questions = json.loads(final_questions)

            results: list[Question] = []
            question_set = set()
            for question in parsed_final_questions["questions"]:
                # extract claims and generate reference for each question
                question_text = question.get("output_question", "")
                if question_text.strip() != "":
                    if question_text in question_set:
                        log.warning(f"Duplicate question: {question_text}")
                        continue
                    question_set.add(question_text.lower().strip())
                    claims = await self.claim_extractor.aextract_claims(
                        question_text=question_text,
                        question_references=text_units,
                    )

                    if claims.reference_coverage > 0:
                        # calculate the similarity of the question to the references
                        (
                            question_embedding,
                            reference_similarity,
                            intra_inter_similarity_ratio,
                        ) = await self.calculate_question_similarity(
                            question_text=question_text, text_units=text_units
                        )
                        log.info(
                            f"Question: {question_text}. Intra-inter Similarity: {intra_inter_similarity_ratio}. Reference Coverage: {claims.reference_coverage}"
                        )

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
                                },
                            )
                        )
            return results

        except Exception as e:
            log.exception(f"Exception: {e}")
            return []

    async def calculate_question_similarity(
        self,
        question_text: str,
        text_units: list[TextUnit],
    ) -> tuple[list[float], dict[str, float], float]:
        """
        Calculate two similarity measures:
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
