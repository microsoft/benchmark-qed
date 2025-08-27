# Copyright (c) 2025 Microsoft Corporation.
"""A package containing the 'Identified' protocol."""

from dataclasses import dataclass


@dataclass
class Identified:
    """A protocol for an item with an ID."""

    id: str
    """The ID of the item."""

    short_id: str | None
    """Human readable ID used to refer to this item in prompts or texts displayed to users, such as in a report text (optional)."""
