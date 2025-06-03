# Copyright (c) 2025 Microsoft Corporation.
"""Enums for methods used in AutoD to select cluster representatives and distance metrics."""

from enum import StrEnum


class ClusterRepresentativeSelectionType(StrEnum):
    """Enum for methods used to select cluster representatives."""

    NEIGHBOR_DISTANCE = "neighbor_distance"
    CENTROID = "centroid"
    ATTRIBUTE_RANKING = "attribute_ranking"


class DistanceMetricType(StrEnum):
    """Enum for distance metrics used in clustering."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
