from dataclasses import dataclass
from typing import Any


@dataclass
class ClaimExtractionResult:
    claims: list[dict[str, Any]]
    """The extracted claims."""

    reference_coverage: float
    """Percentage of input texts referenced in the claims."""

    relevant_references_count: int
    """Count of input texts referenced in the claims"""
