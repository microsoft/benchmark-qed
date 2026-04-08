# Copyright (c) 2025 Microsoft Corporation.
"""Enums for input data types."""

from enum import StrEnum

from graphrag_input import InputType


class InputDataType(StrEnum):
    """Enum for input data types."""

    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    PARQUET = "parquet"  # not provided by graphrag-input
