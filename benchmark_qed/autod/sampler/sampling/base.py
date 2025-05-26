# Copyright (c) 2025 Microsoft Corporation.
"""Base class for methods to select a sample of text units from a dataset."""

import random
from abc import ABC, abstractmethod
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.config.defaults import RANDOM_SEED


class BaseTextSampler(ABC):
    def __init__(self, random_seed: int | None = RANDOM_SEED):
        self.random_seed = random_seed
        random.seed(random_seed)

    @abstractmethod
    def sample(
        self,
        text_units: list[TextUnit],
        sample_size: int | None,
        *args: Any,
        **kwargs: Any,
    ) -> list[TextUnit]:
        """
        Select a subset of text units from the text corpus.
        """
