# Copyright (c) 2025 Microsoft Corporation.
from enum import StrEnum


class ClusterRepresentativeSelectionType(StrEnum):
    NEIGHBOR_DISTANCE = "neighbor_distance"
    CENTROID = "centroid"
    ATTRIBUTE_RANKING = "attribute_ranking"


class DistanceMetricType(StrEnum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
