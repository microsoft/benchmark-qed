# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to calculate source coverage of extracted claims"""

from typing import Any


def compute_source_coverage(
    claims: list[dict[str, Any]], question_references: list[str]
) -> float:
    """Compute the source coverage of the extracted claims"""
    covered_sources = get_relevant_references(claims)
    return (
        covered_sources / len(question_references)
        if len(question_references) > 0
        else 0.0
    )


def get_relevant_references(claims: list[dict[str, Any]]) -> int:
    """Get the relevant references from the extracted claims"""
    relevant_references = set()
    for claim in claims:
        relevant_references.update(claim.get("source_ids", []))
    return len(relevant_references)
