# Copyright (c) 2025 Microsoft Corporation.
"""Module for assessing cluster relevance by testing representative chunks."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_serializer
from tqdm.asyncio import tqdm_asyncio

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.sampler.clustering.base import BaseClustering
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
from benchmark_qed.autod.sampler.clustering.kmeans import KmeansClustering
from benchmark_qed.autod.sampler.neighboring.semantic_neighbors import (
    get_semantic_neighbors_from_text,
)
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentResponse
from benchmark_qed.autoe.retrieval_metrics.relevance_assessment.base import (
    RelevanceRater,
)
from benchmark_qed.autoq.data_model.question import Question

log = logging.getLogger(__name__)


def _clean_assessment_response(response: RelevanceAssessmentResponse) -> dict[str, Any]:
    """Remove embeddings from RelevanceAssessmentResponse for serialization.

    This function creates a copy without embeddings to avoid Pydantic warnings
    about numpy arrays being serialized as list[float], while preserving the
    original objects' embeddings for subsequent operations.
    """
    # Use model_dump with exclude to create a copy without embeddings
    # This avoids mutating the original TextUnit objects
    return response.model_dump(
        exclude={
            "assessment": {
                "__all__": {"text_unit": {"text_embedding"}}
            }
        }
    )


class ClusterRelevanceResult(BaseModel):
    """Result of cluster relevance assessment."""

    cluster_id: str = Field(description="ID of the assessed cluster.")
    cluster_size: int = Field(description="Total number of text units in the cluster.")
    num_assessments: int = Field(description="Total number of relevance assessments performed.")
    num_overlapping_chunks: int = Field(
        description="Number of chunks that appeared in both semantic and centroid neighbor sets."
    )
    semantic_neighbor_assessments: RelevanceAssessmentResponse = Field(
        description="Detailed assessment results for semantically similar chunks."
    )
    centroid_neighbor_assessments: RelevanceAssessmentResponse = Field(
        description="Detailed assessment results for centroid neighbor chunks."
    )
    all_assessments: RelevanceAssessmentResponse = Field(
        description="All relevance assessments for this cluster."
    )

    @field_serializer("semantic_neighbor_assessments", "centroid_neighbor_assessments", "all_assessments")
    def serialize_assessments(self, response: RelevanceAssessmentResponse) -> dict[str, Any]:
        """Serialize assessments without embeddings."""
        return _clean_assessment_response(response)


class QueryClusterReferenceResult(BaseModel):
    """Container for query cluster reference results."""

    question_id: str = Field(description="Unique identifier for the question.")
    question_text: str = Field(description="The text of the question.")
    cluster_results: list[ClusterRelevanceResult] = Field(
        description="List of cluster relevance results for this question."
    )


class ClusterRelevanceRater:
    """A class for assessing cluster relevance by testing representative chunks."""

    def __init__(
        self,
        text_embedder: TextEmbedder,
        relevance_rater: RelevanceRater,
        corpus: list[TextCluster] | list[TextUnit],
        clusterer: BaseClustering | None = None,
        semantic_neighbors: int = 10,
        centroid_neighbors: int = 5,
        max_concurrent_clusters: int = 8,
        num_clusters: int | None = None,
    ) -> None:
        """
        Initialize the ClusterRelevanceRater.

        Args:
            text_embedder: Text embedder for generating query embeddings.
            relevance_rater: Relevance rater instance for assessing chunk relevance.
            corpus: Either pre-computed clusters or text units to cluster once.
                If TextUnit list is provided, clustering will be performed once during init.
            clusterer: Clustering algorithm to use for grouping text units. If None, use Kmeans
            semantic_neighbors: Default number of semantically similar chunks to query to test.
            centroid_neighbors: Default number of neighbors around centroid to test.
            max_concurrent_clusters: Maximum number of clusters to assess concurrently.
            relevance_threshold: Minimum relevance score to consider a chunk relevant.
            num_clusters: Number of clusters to create when clustering TextUnits (auto-determined if None).
        """
        self.text_embedder = text_embedder
        self.semantic_neighbors = semantic_neighbors
        self.centroid_neighbors = centroid_neighbors
        self.semaphore = asyncio.Semaphore(max_concurrent_clusters)
        self.rater = relevance_rater
        self.clusterer = clusterer
        self.num_clusters = num_clusters

        # Process the data parameter
        self.clusters: list[TextCluster] = []

        if len(corpus) == 0:
            log.warning("Empty data list provided")
        elif isinstance(corpus[0], TextCluster):
            self.clusters = corpus  # type: ignore - we know this is list[TextCluster]
            log.info("Initialized with %d pre-computed clusters", len(self.clusters))
        elif isinstance(corpus[0], TextUnit):
            log.info("Performing one-time clustering of %d text units", len(corpus))
            self.clusters = self._cluster_corpus(corpus, self.clusterer, self.num_clusters)  # type: ignore - we know this is list[TextUnit]
            log.info("Created %d clusters", len(self.clusters))
        else:
            msg = f"Data must be list of TextCluster or TextUnit objects, got {type(corpus[0])}"
            raise ValueError(msg)

    def _cluster_corpus(
            self,
            text_units: list[TextUnit],
            clusterer: BaseClustering | None = None,
            num_clusters: int | None = None
        ) -> list[TextCluster]:
        """Perform initial clustering of text units."""
        # Validate that text units have embeddings
        units_with_embeddings = [unit for unit in text_units if unit.text_embedding is not None]
        if not units_with_embeddings:
            msg = "Text units must have embeddings for clustering"
            raise ValueError(msg)

        if len(units_with_embeddings) < len(text_units):
            log.warning(
                "Only %d out of %d text units have embeddings",
                len(units_with_embeddings),
                len(text_units),
            )

        # Perform clustering
        if clusterer is None:
            clusterer = KmeansClustering(random_seed=42)
        try:
            clusters = clusterer.cluster(
                text_units=units_with_embeddings,
                num_clusters=num_clusters
            )

            # Log cluster statistics
            cluster_sizes = [len(cluster.text_units) for cluster in clusters]
            log.info(
                "Cluster size statistics: min=%d, max=%d, mean=%.1f",
                min(cluster_sizes),
                max(cluster_sizes),
                sum(cluster_sizes) / len(cluster_sizes),
            )
        except Exception as e:
            log.exception("Clustering failed")
            msg = f"Failed to perform clustering: {e}"
            raise ValueError(msg) from e
        else:
            return clusters

    async def _assess_single_cluster(
        self,
        query: str,
        cluster: TextCluster,
    ) -> ClusterRelevanceResult:
        """
        Assess cluster relevance by testing representative chunks.

        Args:
            query: The query to assess relevance against.
            cluster: The text cluster to evaluate.

        Returns
        -------
            ClusterRelevanceResult containing relevance assessment results and metadata.
        """
        # Get representative chunks
        representative_chunks = await self._get_representative_chunks(
            cluster=cluster,
            query=query,
        )

        log.info(
            "Testing cluster relevance with %d semantically similar chunks and %d "
            "centroid neighbor chunks (%d overlapping)",
            len(representative_chunks["semantic_neighbors"]),
            len(representative_chunks["centroid_neighbors"]),
            len(representative_chunks["overlapping_chunks"]),
        )

        # Combine all chunks, removing duplicates only for the LLM call
        semantic_chunks = representative_chunks["semantic_neighbors"]
        centroid_chunks = representative_chunks["centroid_neighbors"]

        # Create a deduplicated list for the LLM call
        all_chunks_dict = {chunk.id: chunk for chunk in semantic_chunks + centroid_chunks}
        all_chunks = list(all_chunks_dict.values())

        # Create sets of IDs for proper separation of results
        semantic_ids = {chunk.id for chunk in semantic_chunks}
        centroid_ids = {chunk.id for chunk in centroid_chunks}

        # Make single rate_relevance call for all chunks
        cluster_result = await self.rater.rate_relevance(
            query=query, text_units=all_chunks
        )

        # Separate results by chunk type based on IDs, not ordering
        semantic_assessments = [a for a in cluster_result.assessment if a.text_unit and a.text_unit.id in semantic_ids]
        centroid_assessments = [a for a in cluster_result.assessment if a.text_unit and a.text_unit.id in centroid_ids]

        return ClusterRelevanceResult(
            cluster_id=cluster.id,
            cluster_size=len(cluster.text_units),
            num_assessments=len(cluster_result.assessment),
            num_overlapping_chunks=len(representative_chunks["overlapping_chunks"]),
            semantic_neighbor_assessments=RelevanceAssessmentResponse(assessment=semantic_assessments),
            centroid_neighbor_assessments=RelevanceAssessmentResponse(assessment=centroid_assessments),
            all_assessments=cluster_result
        )

    async def assess_clusters(
        self,
        query: str,
    ) -> list[ClusterRelevanceResult]:
        """
        Assess relevance for multiple clusters concurrently.

        Args:
            query: The query to assess relevance against.
            clusters: Optional list of text clusters to evaluate.
                     If None, uses clusters from initialization data.

        Returns
        -------
            List of ClusterRelevanceResult objects for each cluster.
        """
        # Use pre-computed clusters if available
        log.info("Starting concurrent assessment of %d clusters", len(self.clusters))

        # Create tasks for concurrent execution
        async def assess_single_cluster_with_semaphore(cluster: TextCluster, index: int) -> ClusterRelevanceResult:
            async with self.semaphore:
                log.info(
                    "Assessing cluster %d/%d: cluster ID %s",
                    index + 1,
                    len(self.clusters),
                    cluster.id,
                )
                return await self._assess_single_cluster(query=query, cluster=cluster)

        tasks = [
            assess_single_cluster_with_semaphore(cluster, i)
            for i, cluster in enumerate(self.clusters)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        log.info("Completed concurrent assessment for %d clusters", len(self.clusters))
        return results

    async def assess_batch(
        self,
        queries: list[Question],
    ) -> list[QueryClusterReferenceResult]:
        """
        Assess relevance for multiple queries in parallel using pre-computed clusters.

        Args:
            queries: List of queries to assess relevance for.

        Returns
        -------
            List of QueryClusterReferenceResult objects, each containing:
            - question_id: The ID of the question
            - question_text: The text of the question
            - cluster_results: List of ClusterRelevanceResult objects
        """
        if not queries:
            log.warning("No queries provided for assessment")
            return []

        log.info(
            "Starting parallel assessment of %d queries against %d clusters",
            len(queries),
            len(self.clusters),
        )

        # Create tasks for each query
        async def assess_single_query(query: Question, pbar: Any) -> QueryClusterReferenceResult:
            try:
                cluster_results = await self.assess_clusters(query=query.text)
                log.debug("Completed assessment for query ID: %s", query.id)
                pbar.update(1)
                return QueryClusterReferenceResult(
                    question_id=query.id,
                    question_text=query.text,
                    cluster_results=cluster_results
                )
            except Exception:
                log.exception("Failed to assess query %s", query.id)
                pbar.update(1)
                return QueryClusterReferenceResult(
                    question_id=query.id,
                    question_text=query.text,
                    cluster_results=[]
                )

        # Run all queries in parallel with progress bar
        with tqdm_asyncio(total=len(queries), desc="Assessing queries", unit="query") as pbar:
            tasks = [assess_single_query(query, pbar) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle any exceptions
        query_results = []
        successful_assessments = 0

        for result in results:
            if isinstance(result, Exception):
                log.error("Query assessment failed with exception: %s", result)
                continue
            query_results.append(result)  # type: ignore
            successful_assessments += 1

        log.info(
            "Completed parallel assessment: %d/%d queries processed successfully",
            successful_assessments,
            len(queries),
        )
        return query_results

    async def _get_representative_chunks(
        self,
        cluster: TextCluster,
        query: str,
    ) -> dict[str, list[TextUnit]]:
        """
        Get representative chunks from the cluster for relevance testing.

        Args:
            cluster: The text cluster to sample from.
            query: Query text string to find semantic neighbors for.
            semantic_neighbors: Number of semantically similar chunks to query to select.
            centroid_neighbors: Number of neighbors around centroid to select.

        Returns
        -------
            Dictionary with 'semantic_neighbors', 'centroid_neighbors', and 'overlapping_chunks' lists.
        """
        # Filter chunks that have embeddings
        chunks_with_embeddings = [
            unit for unit in cluster.text_units if unit.text_embedding is not None
        ]

        if not chunks_with_embeddings:
            log.warning("Cluster %s has no chunks with embeddings", cluster.id)
            return {"semantic_neighbors": [], "centroid_neighbors": [], "overlapping_chunks": []}

        # Get semantic neighbors (chunks most similar to the query)
        semantic_neighbor_chunks = await get_semantic_neighbors_from_text(
            text=query,
            corpus=chunks_with_embeddings,
            text_embedder=self.text_embedder,
            n=self.semantic_neighbors,
        )

        # Get centroid neighbors
        centroid_neighbor_chunks = cluster.get_centroid_neighbors(n=self.centroid_neighbors)

        # Find overlapping chunks (appear in both lists)
        semantic_ids = {chunk.id for chunk in semantic_neighbor_chunks}
        overlapping_chunks = [
            chunk for chunk in centroid_neighbor_chunks
            if chunk.id in semantic_ids
        ]

        return {
            "semantic_neighbors": semantic_neighbor_chunks,
            "centroid_neighbors": centroid_neighbor_chunks,
            "overlapping_chunks": overlapping_chunks,
        }


def save_cluster_references_to_json(
    reference_results: list[QueryClusterReferenceResult],
    filepath: str | Path,
    include_clusters: bool = True,
    clusters: list[TextCluster] | None = None
) -> None:
    """
    Save query cluster reference results to a JSON file.

    Args:
        reference_results: List of QueryClusterReferenceResult objects from assess_batch.
        filepath: Path to the JSON file to save to.
        include_clusters: If True, include the actual cluster data in the JSON (for backward compatibility).
        clusters: List of TextCluster objects to save. If None and include_clusters is True,
                 clusters will be extracted from the reference results.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert QueryClusterReferenceResult objects to dictionaries
    json_data = [result.model_dump() for result in reference_results]

    data = {
        "total_questions": len(reference_results),
        "references": json_data,
    }

    # Add cluster data if requested
    if include_clusters and clusters is not None:
        cluster_data = []
        for cluster in clusters:
            cluster_dict = {
                "id": cluster.id,
                "text_units": [
                    {
                        "id": unit.id,
                        "short_id": unit.short_id,
                        "text": unit.text,
                    }
                    for unit in cluster.text_units
                ]
            }
            cluster_data.append(cluster_dict)
        data["clusters"] = cluster_data

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    save_msg = f"Saved batch assessment results for {len(reference_results)} questions"
    if include_clusters and clusters is not None:
        save_msg += f" with {len(clusters)} clusters"
    save_msg += f" to {filepath}"
    log.info(save_msg)


def load_cluster_references_from_json(
    filepath: str | Path
) -> tuple[list[QueryClusterReferenceResult], list[TextCluster] | None]:
    """
    Load query cluster reference results and clusters from a JSON file.

    Args:
        filepath: Path to the JSON file to load from.

    Returns
    -------
        Tuple of (QueryClusterReferenceResult objects, TextCluster objects or None if no clusters were saved).
    """
    filepath = Path(filepath)

    if not filepath.exists():
        msg = f"File not found: {filepath}"
        raise FileNotFoundError(msg)

    with filepath.open(encoding="utf-8") as f:
        data = json.load(f)

    # Convert back to QueryClusterReferenceResult objects
    batch_results = [
        QueryClusterReferenceResult.model_validate(result_data)
        for result_data in data["references"]
    ]

    # Load clusters if they exist
    clusters = None
    if "clusters" in data:
        clusters = []
        for cluster_data in data["clusters"]:
            text_units = []
            for unit_data in cluster_data["text_units"]:
                text_unit = TextUnit(
                    id=unit_data["id"],
                    short_id=unit_data.get("short_id"),
                    text=unit_data["text"],
                )
                text_units.append(text_unit)

            cluster = TextCluster(id=cluster_data["id"], text_units=text_units)
            clusters.append(cluster)

    log.info(
        "Loaded query cluster reference results for %d questions from %s",
        len(batch_results),
        filepath,
    )
    if clusters is not None:
        log.info("Loaded %d clusters from %s", len(clusters), filepath)

    return batch_results, clusters
