# Copyright (c) 2025 Microsoft Corporation.
"""Activity-local question generation module in AutoQ."""

import asyncio
import json
import logging
import math
import random
import uuid
from pathlib import Path
from string import Template
from typing import Any

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autod.sampler.enums import ClusterRepresentativeSelectionType
from benchmark_qed.autod.sampler.neighboring.semantic_neighbors import (
    compute_intra_inter_references_similarity_ratio,
    compute_similarity_to_references,
)
from benchmark_qed.autod.sampler.sampling.kmeans_sampler import KmeansTextSampler
from benchmark_qed.autoq.data_model.activity import ActivityContext, Entity, TaskContext
from benchmark_qed.autoq.data_model.enums import QuestionType
from benchmark_qed.autoq.data_model.question import Question
from benchmark_qed.autoq.prompts.activity_questions import local_questions
from benchmark_qed.autoq.question_gen.base import BaseQuestionGen, QuestionGenResult
from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

ACTIVITY_LOCAL_PROMPTS_PATH = Path(local_questions.__file__).parent


class ActivityLocalQuestionGen(BaseQuestionGen):
    """Generate local activity-based questions for a given dataset given the dataset description."""

    def __init__(
        self,
        llm: ChatModel,
        activity_context: ActivityContext,
        text_embedder: TextEmbedder,
        question_sampler: QuestionSampler | None = None,
        llm_params: dict[str, Any] = defs.LLM_PARAMS,
        json_mode: bool = True,
        generation_system_prompt: Template | None = None,
        generation_user_prompt: Template | None = None,
        concurrent_coroutines: int = 32,
        random_seed: int = defs.RANDOM_SEED,
    ) -> None:
        self.random_seed = random_seed
        random.seed(self.random_seed)

        if question_sampler is not None:
            question_sampler.random_seed = self.random_seed
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

        self.text_embedder = text_embedder

        self.json_mode = json_mode
        if json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

        self.activity_context = activity_context
        self.activity_entities: list[Entity] = activity_context.get_all_entities()
        self.generation_system_prompt: Template = (
            generation_system_prompt
            or load_template_file(
                ACTIVITY_LOCAL_PROMPTS_PATH / "activity_local_gen_system_prompt.txt"
            )
        )
        self.generation_user_prompt: Template = (
            generation_user_prompt
            or load_template_file(
                ACTIVITY_LOCAL_PROMPTS_PATH
                / "activity_local_gen_user_prompt.txt"
            )
        )
        self.concurrent_coroutines = concurrent_coroutines
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self.concurrent_coroutines
        )

    async def agenerate(
        self,
        num_questions: int = defs.NUM_QUESTIONS,
        oversample_factor: float = defs.OVERSAMPLE_FACTOR,
    ) -> QuestionGenResult:
        """Async function to generate activity local questions from [person, task, entities] combination."""
        # Generate questions based on the generated context
        num_candidate_questions = math.ceil(num_questions * oversample_factor)
        num_questions_per_task = max(
            math.ceil(
                num_candidate_questions / len(self.activity_context.task_contexts)
            ),
            1,
        )
        results = await asyncio.gather(*[
            self._agenerate_single_task(
                task_context=context,
                num_questions=num_questions_per_task,
            )
            for context in self.activity_context.task_contexts
        ])
        results = [question for result in results for question in result]
        msg = f"Generated {len(results)} candidate questions"
        log.info(msg)

        # select a subset of questions based on entity-question distribution
        entity_questions: dict[str, list[Question]] = {}
        for question in results:
            entity_names = (
                [
                    entity.split(":")[0].strip()
                    for entity in question.attributes.get("entities", [])
                ]
                if question.attributes is not None
                else []
            )
            for entity in entity_names:
                if entity not in entity_questions:
                    entity_questions[entity] = []
                entity_questions[entity].append(question)

        # Sort entities based on the number of questions
        entity_questions = dict(
            sorted(
                entity_questions.items(), key=lambda item: len(item[1]), reverse=True
            )
        )

        # Distribute the number of questions per entity to ensure we don't have too many questions associated with a single entity
        num_questions_per_entity = max(
            math.floor(num_questions / len(entity_questions)), 1
        )
        msg = (
            f"Number of questions per entity: {num_questions_per_entity}\n"
            f"Number of entities: {len(entity_questions)}"
        )
        log.info(msg)

        filtered_results = []
        unique_question_texts = set()
        entity_question_counts = dict.fromkeys(entity_questions.keys(), 0)
        for entity, questions in entity_questions.items():
            # prioritize questions with higher similarity ratios and better reference coverage.
            questions.sort(
                key=lambda x: (
                    (x.attributes or {}).get("intra_inter_similarity_ratio", 0),
                    (x.attributes or {}).get("reference_coverage", 0),
                ),
                reverse=True,
            )

            for question in questions:
                question_entities = (
                    [
                        related_entity.split(":")[0].strip()
                        for related_entity in question.attributes.get("entities", [])
                    ]
                    if question.attributes is not None
                    else []
                )

                # Check if any question entity has exceeded the number of questions per entity
                # If not, add question to the filtered results, and update counters
                if any(
                    entity_question_counts[entity] >= num_questions_per_entity
                    for entity in question_entities
                ):
                    continue

                if question.text not in unique_question_texts:
                    filtered_results.append(question)
                    unique_question_texts.add(question.text)
                    for related_entity in question_entities:
                        entity_question_counts[related_entity] += 1
                    if entity_question_counts[entity] >= num_questions_per_entity:
                        break

        # Fill in the missing questions if needed
        msg = f"Retained {len(filtered_results)} questions post-filtering by entity distribution"
        log.info(msg)
        missing_questions = num_questions - len(filtered_results)
        if missing_questions > 0:
            msg = f"Filling in {missing_questions} missing questions"
            log.info(msg)
            remaining_questions = [
                question
                for question in results
                if question.text not in unique_question_texts
            ]
            selected_questions = filtered_results + self.select(
                remaining_questions, missing_questions
            )
        else:
            selected_questions = self.select(filtered_results, num_questions)
        return QuestionGenResult(
            selected_questions=selected_questions,
            candidate_questions=results,
        )

    async def _agenerate_single_task(
        self,
        task_context: TaskContext,
        num_questions: int,
    ) -> list[Question]:
        """Generate questions for a single task context using a chain of thought process."""
        results: list[Question] = []
        try:
            if len(task_context.entities) == 0:
                msg = f"No entities found for task: {task_context.task}"
                log.warning(msg)
                return []

            # generate a set of questions from the entity list
            random.shuffle(task_context.entities)
            entity_description = "\n\n".join([
                entity.to_str() for entity in task_context.entities
            ])
            question_input_prompt = self.generation_user_prompt.substitute(
                entity_description=entity_description,
                persona=task_context.persona,
                task=task_context.task,
                num_questions=num_questions,
            )
            extraction_messages = [
                {
                    "role": "system",
                    "content": self.generation_system_prompt.substitute(
                        num_questions=num_questions
                    ),
                },
                {"role": "user", "content": question_input_prompt},
            ]

            model_response = await self.llm.chat(
                messages=extraction_messages, **self.llm_params
            )

            questions, j = try_parse_json_object(model_response.output.content)
            if j == {}:
                msg = f"Error parsing JSON response: {questions}"
                log.error(msg)
                return []
            parsed_questions = json.loads(questions).get("questions", [])
            msg = f"Parsed questions: {parsed_questions}"
            log.debug(msg)

            question_set: set[str] = set()
            for question in parsed_questions:
                # calculate intra-inter-similarity ratio and reference coverage
                question_text = question.get("output_question", "")
                if question_text != "" and question_text not in question_set:
                    question_set.add(question_text)
                    question_embedding = await self.text_embedder.embed_raw_text(
                        question_text
                    )

                    hashed_entities = {
                        entity.name: entity for entity in task_context.entities
                    }
                    in_reference_entity_texts = [
                        TextUnit(
                            id=str(index),
                            short_id=str(index),
                            text=entity.name,
                            text_embedding=entity.embedding,
                        )
                        for index, entity in enumerate(task_context.entities)
                    ]
                    reference_similarity = compute_similarity_to_references(
                        text_embedding=question_embedding,
                        references=in_reference_entity_texts,
                    )

                    out_reference_entity_texts = [
                        TextUnit(
                            id=str(index),
                            short_id=str(index),
                            text=entity.name,
                            text_embedding=entity.embedding,
                        )
                        for index, entity in enumerate(self.activity_entities)
                        if entity.name not in hashed_entities
                    ]
                    intra_inter_similarity_ratio = (
                        compute_intra_inter_references_similarity_ratio(
                            text_embedding=question_embedding,
                            in_references=in_reference_entity_texts,
                            out_references=out_reference_entity_texts,
                        )
                    )

                    # calculate reference coverage
                    related_entities = question.get("entities", [])
                    if isinstance(related_entities, str):
                        related_entities = [related_entities]
                    reference_coverage = (
                        len(related_entities) / len(task_context.entities)
                        if len(task_context.entities) > 0
                        else 0
                    )

                    results.append(
                        Question(
                            id=str(uuid.uuid4()),
                            text=question_text,
                            question_type=QuestionType.ACTIVITY_LOCAL,
                            embedding=question_embedding,
                            references=[
                                entity.to_str() for entity in task_context.entities
                            ],
                            attributes={
                                "input_question": question.get("input_question", ""),
                                "persona": task_context.persona,
                                "task": task_context.task,
                                "entities": related_entities,
                                "reference_count": len(task_context.entities),
                                "reference_coverage": reference_coverage,
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
                            },
                        )
                    )

        except Exception:
            log.exception("Exception occurred while generating questions.")
            return []
        else:
            return results
