# Copyright (c) 2025 Microsoft Corporation.
"""Activity-global question generation module in AutoQ."""

import asyncio
import json
import logging
import math
import uuid
from pathlib import Path
from string import Template
from typing import Any

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.data_processor.text_utils import try_parse_json_object
from benchmark_qed.autod.sampler.enums import ClusterRepresentativeSelectionType
from benchmark_qed.autod.sampler.sampling.kmeans_sampler import KmeansTextSampler
from benchmark_qed.autoq.data_model.activity import ActivityContext, TaskContext
from benchmark_qed.autoq.data_model.enums import QuestionType
from benchmark_qed.autoq.data_model.question import Question
from benchmark_qed.autoq.prompts.activity_questions import global_questions
from benchmark_qed.autoq.question_gen.base import BaseQuestionGen, QuestionGenResult
from benchmark_qed.autoq.sampler.question_sampler import QuestionSampler
from benchmark_qed.config.utils import load_template_file
from benchmark_qed.llm.type.base import ChatModel

log: logging.Logger = logging.getLogger(__name__)

ACTIVITY_GLOBAL_QUESTIONS_PATH = Path(global_questions.__file__).parent


class ActivityGlobalQuestionGen(BaseQuestionGen):
    """Generate global activity-based questions for a given dataset given the dataset description."""

    def __init__(
        self,
        llm: ChatModel,
        text_embedder: TextEmbedder,
        activity_context: ActivityContext,
        question_sampler: QuestionSampler | None = None,
        llm_params: dict[str, Any] = defs.LLM_PARAMS,
        json_mode: bool = True,
        generation_system_prompt: Template | None = None,
        generation_user_prompt: Template | None = None,
        concurrent_coroutines: int = 32,
        random_seed: int = defs.RANDOM_SEED,
    ) -> None:
        self.random_seed = random_seed
        if question_sampler is not None:
            question_sampler.random_seed = random_seed
        else:
            question_sampler = QuestionSampler(
                sampler=KmeansTextSampler(),
                sampler_params={
                    "sample_selection": ClusterRepresentativeSelectionType.CENTROID,
                    "num_samples_per_cluster": 1,
                },
            )
        super().__init__(llm, llm_params, question_sampler)

        self.text_embedder = text_embedder

        self.json_mode = json_mode
        if json_mode:
            self.llm_params["response_format"] = {"type": "json_object"}
        else:
            self.llm_params.pop("response_format", None)

        self.generation_system_prompt: str = (
            generation_system_prompt
            or load_template_file(
                ACTIVITY_GLOBAL_QUESTIONS_PATH
                / "global_generation_generation_system_prompt.txt"
            )
        ).template
        self.generation_user_prompt: Template = (
            generation_user_prompt
            or load_template_file(
                ACTIVITY_GLOBAL_QUESTIONS_PATH
                / "global_generation_generation_user_prompt.txt"
            )
        )
        self.activity_context = activity_context
        self.concurrent_coroutines = concurrent_coroutines
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self.concurrent_coroutines
        )

    async def agenerate(
        self,
        num_questions: int = defs.NUM_QUESTIONS,
        oversample_factor: float = defs.OVERSAMPLE_FACTOR,
    ) -> QuestionGenResult:
        """Async function to generate activity global questions for the dataset descriptions and activity context."""
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
                dataset_description=self.activity_context.dataset_description,
                task_context=context,
                num_questions_per_task=num_questions_per_task,
            )
            for context in self.activity_context.task_contexts
        ])

        results = [question for result in results for question in result]
        msg = f"Generated {len(results)} candidate questions for {len(self.activity_context.task_contexts)} tasks"
        log.info(msg)

        selected_questions = self.select(
            candidate_questions=results, top_k=num_questions
        )
        return QuestionGenResult(
            candidate_questions=results,
            selected_questions=selected_questions,
        )

    async def _agenerate_single_task(
        self,
        dataset_description: str,
        task_context: TaskContext,
        num_questions_per_task: int = 5,
    ) -> list[Question]:
        """Generate questions based on dataset description, persona and task descriptions."""
        results: list[Question] = []
        try:
            question_input_prompt = self.generation_user_prompt.substitute(
                dataset_description=dataset_description,
                persona=task_context.persona,
                task=task_context.task,
                num_questions=num_questions_per_task,
            )
            generation_messages = [
                {"role": "system", "content": self.generation_system_prompt},
                {"role": "user", "content": question_input_prompt},
            ]
            model_response = await self.llm.chat(
                messages=generation_messages, **self.llm_params
            )
            questions, j = try_parse_json_object(model_response.output.content)
            if j == {}:
                msg = f"Failed to parse questions response: {model_response.output.content}"
                log.error(msg)
                return []
            parsed_questions = json.loads(questions)

            question_set = set()
            for question in parsed_questions.get("questions", []):
                question_text = question.get("output_question", "")
                if question_text != "" and question_text not in question_set:
                    question_set.add(question_text)
                    results.append(
                        Question(
                            id=str(uuid.uuid4()),
                            text=question_text,
                            question_type=QuestionType.ACTIVITY_GLOBAL,
                            embedding=await self.text_embedder.embed_raw_text(
                                question_text
                            ),
                            references=[f"Dataset description: {dataset_description}"],
                            attributes={
                                "persona": task_context.persona,
                                "task": task_context.task,
                                "reasoning": question.get("reasoning", ""),
                            },
                        )
                    )

        except Exception:
            log.exception("Exception during question generation")
            return []
        else:
            return results
