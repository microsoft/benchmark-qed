# Copyright (c) 2025 Microsoft Corporation.
"""Statistics generation for assertion files.

This module provides functions to compute and report statistics on generated assertions,
including total counts, per-question metrics, source distributions, and supporting assertion metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class DistributionStats:
    """Statistics for a numeric distribution."""

    count: int = 0
    total: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "count": self.count,
            "total": self.total,
            "mean": round(self.mean, 3),
            "std": round(self.std, 3),
            "min": self.min,
            "max": self.max,
            "median": round(self.median, 3),
        }

    @classmethod
    def from_values(cls, values: Sequence[int | float]) -> DistributionStats:
        """Create distribution stats from a list of values."""
        if not values:
            return cls()

        arr = np.array(values)
        return cls(
            count=len(arr),
            total=int(arr.sum()),
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            max=float(arr.max()),
            median=float(np.median(arr)),
        )


@dataclass
class AssertionStats:
    """Statistics for a collection of assertions."""

    file_path: str = ""
    assertion_type: str = ""  # "global", "map", or "local"
    total_questions: int = 0
    total_assertions: int = 0
    assertions_per_question: DistributionStats = field(
        default_factory=DistributionStats
    )
    sources_per_assertion: DistributionStats = field(default_factory=DistributionStats)
    unique_sources_per_question: DistributionStats | None = None
    unique_claim_sources_per_question: DistributionStats | None = None
    supporting_assertions_per_assertion: DistributionStats | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "file_path": self.file_path,
            "assertion_type": self.assertion_type,
            "total_questions": self.total_questions,
            "total_assertions": self.total_assertions,
            "assertions_per_question": self.assertions_per_question.to_dict(),
            "sources_per_assertion": self.sources_per_assertion.to_dict(),
        }
        if self.unique_sources_per_question is not None:
            result["unique_sources_per_question"] = (
                self.unique_sources_per_question.to_dict()
            )
        if self.unique_claim_sources_per_question is not None:
            result["unique_claim_sources_per_question"] = (
                self.unique_claim_sources_per_question.to_dict()
            )
        if self.supporting_assertions_per_assertion is not None:
            result["supporting_assertions_per_assertion"] = (
                self.supporting_assertions_per_assertion.to_dict()
            )
        return result

    def __str__(self) -> str:
        """Return a formatted string representation of the stats."""
        lines = [
            f"Assertion Statistics ({self.assertion_type})",
            "=" * 50,
            f"File: {self.file_path}",
            f"Total questions: {self.total_questions}",
            f"Total assertions: {self.total_assertions}",
            "",
            "Assertions per question:",
            f"  Mean: {self.assertions_per_question.mean:.2f}",
            f"  Std:  {self.assertions_per_question.std:.2f}",
            f"  Min:  {self.assertions_per_question.min:.0f}",
            f"  Max:  {self.assertions_per_question.max:.0f}",
            "",
            "Sources per assertion:",
            f"  Mean: {self.sources_per_assertion.mean:.2f}",
            f"  Std:  {self.sources_per_assertion.std:.2f}",
            f"  Min:  {self.sources_per_assertion.min:.0f}",
            f"  Max:  {self.sources_per_assertion.max:.0f}",
        ]

        if self.unique_sources_per_question is not None:
            lines.extend([
                "",
                "Unique sources per question (from assertions):",
                f"  Mean: {self.unique_sources_per_question.mean:.2f}",
                f"  Std:  {self.unique_sources_per_question.std:.2f}",
                f"  Min:  {self.unique_sources_per_question.min:.0f}",
                f"  Max:  {self.unique_sources_per_question.max:.0f}",
            ])

        if self.unique_claim_sources_per_question is not None:
            lines.extend([
                "",
                "Unique sources per question (from claims):",
                f"  Mean: {self.unique_claim_sources_per_question.mean:.2f}",
                f"  Std:  {self.unique_claim_sources_per_question.std:.2f}",
                f"  Min:  {self.unique_claim_sources_per_question.min:.0f}",
                f"  Max:  {self.unique_claim_sources_per_question.max:.0f}",
            ])

        if self.supporting_assertions_per_assertion is not None:
            lines.extend([
                "",
                "Supporting assertions per global assertion:",
                f"  Mean: {self.supporting_assertions_per_assertion.mean:.2f}",
                f"  Std:  {self.supporting_assertions_per_assertion.std:.2f}",
                f"  Min:  {self.supporting_assertions_per_assertion.min:.0f}",
                f"  Max:  {self.supporting_assertions_per_assertion.max:.0f}",
            ])

        return "\n".join(lines)


def compute_assertion_stats(
    assertions_data: list[dict[str, Any]],
    assertion_type: str = "global",
    file_path: str = "",
    sources_data: list[dict[str, Any]] | None = None,
) -> AssertionStats:
    """Compute statistics for a collection of assertions.

    Args:
        assertions_data: List of question dictionaries containing assertions.
            Expected format:
            - For global assertions: [{"question_id": ..., "assertions": [...]}]
            - For map assertions: [{"question_id": ..., "map_assertions": [...]}]
        assertion_type: Type of assertions ("global", "map", or "local").
        file_path: Path to the source file (for reference).
        sources_data: Optional list of source data for computing unique sources per question.
            Expected format: [{"question_id": ..., "assertion_sources": [{"sources": [...]}]}]

    Returns
    -------
        AssertionStats object containing computed statistics.
    """
    # Determine the assertions key based on type
    assertions_key = "map_assertions" if assertion_type == "map" else "assertions"
    sources_key = (
        "map_assertion_sources" if assertion_type == "map" else "assertion_sources"
    )

    # Build a lookup for sources by question_id if sources_data is provided
    sources_by_question: dict[str, list[dict[str, Any]]] = {}
    if sources_data:
        for question_sources in sources_data:
            qid = question_sources.get("question_id", "")
            sources_by_question[qid] = question_sources.get(sources_key, [])

    # Collect per-question assertion counts
    assertions_per_question: list[int] = []
    all_source_counts: list[int] = []
    all_supporting_counts: list[int] = []
    unique_sources_per_question: list[int] = []
    unique_claim_sources_per_question: list[int] = []
    has_supporting = False

    for question_data in assertions_data:
        assertions = question_data.get(assertions_key, [])
        assertions_per_question.append(len(assertions))

        # Collect unique sources for this question
        question_id = question_data.get("question_id", "")
        question_unique_sources: set[str] = set()

        for assertion in assertions:
            # Source count - handle both 'source_count' and 'sources' fields
            source_count = assertion.get("source_count")
            if source_count is None:
                sources = assertion.get("sources", [])
                source_count = len(sources) if isinstance(sources, list) else 0
            all_source_counts.append(source_count)

            # Supporting assertions (only for global assertions)
            supporting = assertion.get("supporting_assertions", [])
            if supporting:
                has_supporting = True
                all_supporting_counts.append(len(supporting))

        # Get unique sources from sources_data if available
        if question_id in sources_by_question:
            for assertion_source in sources_by_question[question_id]:
                sources = assertion_source.get("sources", [])
                for source in sources:
                    # Use hash of source text to identify unique sources
                    if isinstance(source, str):
                        question_unique_sources.add(source)
            unique_sources_per_question.append(len(question_unique_sources))

        # Compute unique sources from claims (pre-assertion metric)
        claims = question_data.get("claims", [])
        claim_sources: set[str] = set()
        for claim in claims:
            for source_id in claim.get("source_ids", []):
                if isinstance(source_id, str):
                    claim_sources.add(source_id)
        if claims:  # Only add if there are claims
            unique_claim_sources_per_question.append(len(claim_sources))

    stats = AssertionStats(
        file_path=file_path,
        assertion_type=assertion_type,
        total_questions=len(assertions_data),
        total_assertions=sum(assertions_per_question),
        assertions_per_question=DistributionStats.from_values(assertions_per_question),
        sources_per_assertion=DistributionStats.from_values(all_source_counts),
    )

    # Only include unique sources stats if we have sources data
    if unique_sources_per_question:
        stats.unique_sources_per_question = DistributionStats.from_values(
            unique_sources_per_question
        )

    # Only include unique claim sources stats if we have claims data
    if unique_claim_sources_per_question:
        stats.unique_claim_sources_per_question = DistributionStats.from_values(
            unique_claim_sources_per_question
        )

    # Only include supporting assertion stats if they exist
    if has_supporting and all_supporting_counts:
        stats.supporting_assertions_per_assertion = DistributionStats.from_values(
            all_supporting_counts
        )

    return stats


def compute_stats_from_file(
    file_path: Path | str,
    assertion_type: str | None = None,
) -> AssertionStats:
    """Load assertion file and compute statistics.

    Also tries to load the corresponding sources file (assertion_sources.json or
    map_assertion_sources.json) to compute unique sources per question.

    Args:
        file_path: Path to the JSON assertion file.
        assertion_type: Type of assertions. If None, inferred from filename.
            - "global" for assertions.json or files with "global" in name
            - "map" for map_assertions.json or files with "map" in name
            - "local" for other files

    Returns
    -------
        AssertionStats object containing computed statistics.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        msg = f"Assertion file not found: {file_path}"
        raise FileNotFoundError(msg)

    # Infer assertion type from filename if not specified
    if assertion_type is None:
        filename = file_path.name.lower()
        if "map" in filename:
            assertion_type = "map"
        elif "global" in filename or filename == "assertions.json":
            assertion_type = "global"
        else:
            assertion_type = "local"

    # Load assertions
    with file_path.open(encoding="utf-8") as f:
        assertions_data = json.load(f)

    # Try to load corresponding sources file for unique source stats
    sources_data = None
    if assertion_type == "map":
        sources_file = file_path.parent / "map_assertion_sources.json"
    else:
        sources_file = file_path.parent / "assertion_sources.json"

    if sources_file.exists():
        with sources_file.open(encoding="utf-8") as f:
            sources_data = json.load(f)
        log.info("Loaded sources from %s for unique source stats", sources_file)

    return compute_assertion_stats(
        assertions_data=assertions_data,
        assertion_type=assertion_type,
        file_path=str(file_path),
        sources_data=sources_data,
    )


def save_stats_to_file(
    stats: AssertionStats,
    output_path: Path | str,
) -> None:
    """Save assertion statistics to a JSON file.

    Args:
        stats: AssertionStats object to save.
        output_path: Path to the output JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    log.info("Saved assertion stats to %s", output_path)


def generate_stats_for_assertion_file(
    assertions_path: Path | str,
    output_path: Path | str | None = None,
    assertion_type: str | None = None,
    print_stats: bool = True,
) -> AssertionStats:
    """Generate and optionally save statistics for an assertion file.

    This is the main entry point for generating stats on existing assertion files.

    Args:
        assertions_path: Path to the assertion JSON file.
        output_path: Path to save stats JSON. If None, saves as {assertions_path}_stats.json.
        assertion_type: Type of assertions. If None, inferred from filename.
        print_stats: Whether to print statistics to console.

    Returns
    -------
        AssertionStats object containing computed statistics.
    """
    assertions_path = Path(assertions_path)

    # Compute stats
    stats = compute_stats_from_file(assertions_path, assertion_type)

    # Print if requested
    if print_stats:
        print(stats)
        print()

    # Determine output path
    if output_path is None:
        output_path = assertions_path.parent / f"{assertions_path.stem}_stats.json"
    else:
        output_path = Path(output_path)

    # Save stats
    save_stats_to_file(stats, output_path)

    return stats


def generate_stats_for_directory(
    directory: Path | str,
    output_dir: Path | str | None = None,
    print_stats: bool = True,
) -> dict[str, AssertionStats]:
    """Generate statistics for all assertion files in a directory.

    Looks for assertions.json and map_assertions.json files.

    Args:
        directory: Path to directory containing assertion files.
        output_dir: Directory to save stats files. If None, saves alongside originals.
        print_stats: Whether to print statistics to console.

    Returns
    -------
        Dictionary mapping file names to their AssertionStats.
    """
    directory = Path(directory)
    output_dir = Path(output_dir) if output_dir else directory

    results: dict[str, AssertionStats] = {}

    # Look for standard assertion files
    assertion_files = [
        ("assertions.json", "global"),
        ("map_assertions.json", "map"),
    ]

    for filename, assertion_type in assertion_files:
        file_path = directory / filename
        if file_path.exists():
            output_path = output_dir / f"{file_path.stem}_stats.json"
            stats = generate_stats_for_assertion_file(
                assertions_path=file_path,
                output_path=output_path,
                assertion_type=assertion_type,
                print_stats=print_stats,
            )
            results[filename] = stats

    return results
