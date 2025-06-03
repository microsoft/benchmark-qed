# Copyright (c) 2025 Microsoft Corporation.
"""Enums for question classes in AutoQ."""

from enum import StrEnum


class QuestionType(StrEnum):
    """Enum for question classes."""

    DATA_LOCAL = "data_local"
    DATA_GLOBAL = "data_global"
    ACTIVITY_LOCAL = "activity_local"
    ACTIVITY_GLOBAL = "activity_global"
