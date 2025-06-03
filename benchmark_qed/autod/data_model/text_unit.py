# Copyright (c) 2025 Microsoft Corporation.
"""A package containing the TextUnit model."""

from dataclasses import dataclass
from typing import Any

from benchmark_qed.autod.data_model.identified import Identified


@dataclass
class TextUnit(Identified):
    """A class representing a text unit."""

    text: str
    """The text of the unit."""

    document_id: str | None = None
    """The ID of the document this text unit belongs to (optional)."""

    n_tokens: int | None = None
    """The number of tokens in the text (optional)."""

    text_embedding: list[float] | None = None
    """The embedding of the text unit (optional)."""

    cluster_id: str | None = None
    """The ID of the cluster this text unit belongs to (optional)."""

    attributes: dict[str, Any] | None = None
    """A dictionary of additional attributes associated with the text unit (optional)."""
