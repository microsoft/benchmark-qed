# Copyright (c) 2025 Microsoft Corporation.
"""Data models for scoring."""

from .condition_pair import ConditionPair
from .pairwise import PairwiseLLMResponse
from .reference import ReferenceLLMResponse

__all__ = ["ConditionPair", "PairwiseLLMResponse", "ReferenceLLMResponse"]
