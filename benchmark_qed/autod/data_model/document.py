# Copyright (c) 2025 Microsoft Corporation.
"""A package containing the Document model."""

from dataclasses import dataclass
from typing import Any

from benchmark_qed.autod.data_model.identified import Identified


@dataclass
class Document(Identified):
    """A protocol for a document in the system."""

    title: str
    """"The title of the document."""

    type: str = "text"
    """Type of the document."""

    text: str = ""
    """The raw text content of the document."""

    attributes: dict[str, Any] | None = None
    """A dictionary of structured attributes such as author, etc (optional)."""
