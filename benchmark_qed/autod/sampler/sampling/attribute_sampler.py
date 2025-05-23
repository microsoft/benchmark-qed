import logging
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.sampling.base import BaseTextSampler

log: logging.Logger = logging.getLogger(__name__)


class AttributeRankingSampler(BaseTextSampler):
    def sample(
        self,
        text_units: list[TextUnit],
        sample_size: int | None,
        ranking_attributes: list[str],
        ascending: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> list[TextUnit]:
        """
        Sort text units based on a list of attributes and select the top N text units.
        """
        if sample_size is None or sample_size >= len(text_units):
            log.warning(
                "Warning: Insufficient number of text units in the corpus. Returning the entire corpus."
            )
            return text_units

        return sorted(
            text_units,
            key=lambda x: tuple(
                x.attributes[attr]
                for attr in ranking_attributes
                if x.attributes is not None and attr in x.attributes
            ),
            reverse=not ascending,
        )[:sample_size]
