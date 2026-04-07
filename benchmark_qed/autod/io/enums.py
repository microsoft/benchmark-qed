# Copyright (c) 2025 Microsoft Corporation.
"""Enums for input data types."""

from enum import StrEnum

from graphrag_input import InputType


class InputDataType(StrEnum):
    """Enum for input data types."""

    JSON = InputType.Json
    CSV = InputType.Csv
    TEXT = InputType.Text
    PARQUET = "parquet"  # not provided by graphrag-input
