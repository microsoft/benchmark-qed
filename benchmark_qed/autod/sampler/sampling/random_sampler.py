# Copyright (c) 2025 Microsoft Corporation.
"""Sampler that randomly samples text units from a corpus."""

import random
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.sampling.base import BaseTextSampler


class RandomTextSampler(BaseTextSampler):
    """Sampler that randomly samples text units from a corpus."""

    def sample(
        self, text_units: list[TextUnit], sample_size: int | None, **_kwargs: Any
    ) -> list[TextUnit]:
        """Randomly sample text units from the text corpus."""
        if sample_size is None or sample_size >= len(text_units):
            return text_units

        return random.sample(text_units, sample_size)
