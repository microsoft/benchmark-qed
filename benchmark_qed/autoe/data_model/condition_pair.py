# Copyright (c) 2025 Microsoft Corporation.
"""Common data structures for scoring."""

from typing import NamedTuple


class ConditionPair(NamedTuple):
    """Pair of conditions for scoring."""

    question_id: str
    question_text: str
    answer_base: str
    answer_other: str
