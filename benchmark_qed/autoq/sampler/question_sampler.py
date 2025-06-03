# Copyright (c) 2025 Microsoft Corporation.
"""Sampling functions to select a sample of text units from a dataset."""

import logging
import random
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.sampling.base import BaseTextSampler
from benchmark_qed.autoq.data_model.question import Question

log: logging.Logger = logging.getLogger(__name__)


class QuestionSampler:
    """A class to sample questions from a question corpus using a text sampler."""

    def __init__(
        self,
        sampler: BaseTextSampler,
        sampler_params: dict[str, Any] | None = None,
        random_seed: int | None = 42,
    ) -> None:
        if sampler_params is None:
            sampler_params = {}
        self.sampler = sampler
        self.sampler_params = sampler_params
        self.random_seed = random_seed
        self.sampler.random_seed = random_seed
        random.seed(random_seed)

    def sample(
        self, questions: list[Question], sample_size: int | None
    ) -> list[Question]:
        """Select a subset of questions from the question corpus."""
        # convert questions to text units
        text_units = [
            TextUnit(
                id=question.id,
                short_id=str(index),
                text=question.text,
                text_embedding=question.embedding,
                attributes=question.attributes,
            )
            for index, question in enumerate(questions)
        ]

        # sample text units
        sampled_text_units = self.sampler.sample(
            text_units, sample_size, **self.sampler_params
        )
        msg = f"Selected {len(sampled_text_units)} questions from {len(text_units)} candidates."
        log.info(msg)

        # convert text units back to questions
        question_map = {question.id: question for question in questions}
        return [question_map[text_unit.id] for text_unit in sampled_text_units]
