# Copyright (c) 2025 Microsoft Corporation.
"""Module for assessing cluster relevance by testing representative chunks."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
from benchmark_qed.autod.sampler.clustering.base import BaseClustering
from benchmark_qed.autod.sampler.clustering.kmeans import KmeansClustering
from benchmark_qed.autod.sampler.neighboring.semantic_neighbors import get_semantic_neighbors_from_text
from benchmark_qed.autoe.retrieval_scores.relevance_assessment.base import RelevanceRater
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentResponse
from benchmark_qed.autoq.data_model.question import Question

log = logging.getLogger(__name__)


@dataclass
class ClusterRelevanceResult:
    """Result of cluster relevance assessment."""

    cluster_id: str
    """ID of the assessed cluster."""
    
    cluster_size: int
    """Total number of text units in the cluster."""
    
    num_assessments: int
    """Total number of relevance assessments performed."""
    
    num_overlapping_chunks: int
    """Number of chunks that appeared in both semantic and centroid neighbor sets."""
    
    semantic_neighbor_assessments: RelevanceAssessmentResponse
    """Detailed assessment results for semantically similar chunks."""

    centroid_neighbor_assessments: RelevanceAssessmentResponse
    """Detailed assessment results for centroid neighbor chunks."""
    
    all_assessments: RelevanceAssessmentResponse
    """All relevance assessments for this cluster."""
    
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization, excluding embeddings."""
        
        def clean_assessment_response(response) -> dict[str, Any]:
            """Remove embeddings from RelevanceAssessmentResponse for serialization."""
            response_dict = response.model_dump()
            
            # Clean each assessment item to remove text_embedding
            for item in response_dict.get("assessment", []):
                if "text_unit" in item and item["text_unit"] is not None:
                    # Remove the text_embedding field to reduce file size
                    if "text_embedding" in item["text_unit"]:
                        item["text_unit"]["text_embedding"] = None
            
            return response_dict
        
        return {
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "num_assessments": self.num_assessments,
            "num_overlapping_chunks": self.num_overlapping_chunks,
            "semantic_neighbor_assessments": clean_assessment_response(self.semantic_neighbor_assessments),
            "centroid_neighbor_assessments": clean_assessment_response(self.centroid_neighbor_assessments),
            "all_assessments": clean_assessment_response(self.all_assessments),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClusterRelevanceResult":
        """Create from dictionary (JSON deserialization)."""
        from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentResponse
        
        # Reconstruct RelevanceAssessmentResponse objects
        semantic_assessments = RelevanceAssessmentResponse.model_validate(data["semantic_neighbor_assessments"])
        centroid_assessments = RelevanceAssessmentResponse.model_validate(data["centroid_neighbor_assessments"])
        all_assessments = RelevanceAssessmentResponse.model_validate(data["all_assessments"])
        
        return cls(
            cluster_id=data["cluster_id"],
            cluster_size=data["cluster_size"],
            num_assessments=data["num_assessments"],
            num_overlapping_chunks=data["num_overlapping_chunks"],
            semantic_neighbor_assessments=semantic_assessments,
            centroid_neighbor_assessments=centroid_assessments,
            all_assessments=all_assessments,
        )


@dataclass
class QueryClusterReferenceResult:
    """Container for query cluster reference results."""
    
    question_id: str
    question_text: str
    cluster_results: list[ClusterRelevanceResult]
    
    def to_dict(self, include_clusters: bool = False) -> dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        result_dict = {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "cluster_results": [result.to_dict() for result in self.cluster_results]
        }
        return result_dict
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryClusterReferenceResult":
        """Create from dictionary format."""
        cluster_results = [
            ClusterRelevanceResult.from_dict(result_data)
            for result_data in data["cluster_results"]
        ]
        
        return cls(
            question_id=data["question_id"],
            question_text=data["question_text"],
            cluster_results=cluster_results
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
            log.info(f"Initialized with {len(self.clusters)} pre-computed clusters")
        elif isinstance(corpus[0], TextUnit):
            log.info(f"Performing one-time clustering of {len(corpus)} text units")
            self.clusters = self._cluster_corpus(corpus, self.clusterer, self.num_clusters)  # type: ignore - we know this is list[TextUnit]
            log.info(f"Created {len(self.clusters)} clusters")
        else:
            raise ValueError(f"Data must be list of TextCluster or TextUnit objects, got {type(corpus[0])}")

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
            raise ValueError("Text units must have embeddings for clustering")
        
        if len(units_with_embeddings) < len(text_units):
            log.warning(f"Only {len(units_with_embeddings)} out of {len(text_units)} text units have embeddings")
            
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
            log.info(f"Cluster size statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
                    f"mean={sum(cluster_sizes)/len(cluster_sizes):.1f}")
            
            return clusters
        except Exception as e:
            log.error(f"Clustering failed: {e}")
            raise ValueError(f"Failed to perform clustering: {e}") from e

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

        Returns:
            ClusterRelevanceResult containing relevance assessment results and metadata.
        """
        # Get representative chunks
        representative_chunks = await self._get_representative_chunks(
            cluster=cluster,
            query=query,
        )

        log.info(
            f"Testing cluster relevance with {len(representative_chunks['semantic_neighbors'])} "
            f"semantically similar chunks and {len(representative_chunks['centroid_neighbors'])} "
            f"centroid neighbor chunks ({len(representative_chunks['overlapping_chunks'])} overlapping)"
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

        Returns:
            List of ClusterRelevanceResult objects for each cluster.
        """
        # Use pre-computed clusters if available
        log.info(f"Starting concurrent assessment of {len(self.clusters)} clusters")
        
        # Create tasks for concurrent execution
        async def assess_single_cluster_with_semaphore(cluster: TextCluster, index: int) -> ClusterRelevanceResult:
            async with self.semaphore:
                log.info(f"Assessing cluster {index + 1}/{len(self.clusters)}: cluster ID {cluster.id}")
                return await self._assess_single_cluster(query=query, cluster=cluster)
        
        tasks = [
            assess_single_cluster_with_semaphore(cluster, i) 
            for i, cluster in enumerate(self.clusters)
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        log.info(f"Completed concurrent assessment for {len(self.clusters)} clusters")
        return results

    async def assess_batch(
        self,
        queries: list[Question],
    ) -> list[QueryClusterReferenceResult]:
        """
        Assess relevance for multiple queries in parallel using pre-computed clusters.

        Args:
            queries: List of queries to assess relevance for.

        Returns:
            List of QueryClusterReferenceResult objects, each containing:
            - question_id: The ID of the question
            - question_text: The text of the question
            - cluster_results: List of ClusterRelevanceResult objects
        """        
        if not queries:
            log.warning("No queries provided for assessment")
            return []
        
        log.info(f"Starting parallel assessment of {len(queries)} queries against {len(self.clusters)} clusters")
        
        # Create tasks for each query
        async def assess_single_query(query: Question) -> QueryClusterReferenceResult:
            try:
                cluster_results = await self.assess_clusters(query=query.text)
                log.debug(f"Completed assessment for query ID: {query.id}")
                return QueryClusterReferenceResult(
                    question_id=query.id,
                    question_text=query.text,
                    cluster_results=cluster_results
                )
            except Exception as e:
                log.error(f"Failed to assess query {query.id}: {e}")
                return QueryClusterReferenceResult(
                    question_id=query.id,
                    question_text=query.text,
                    cluster_results=[]
                )
        
        # Run all queries in parallel
        tasks = [assess_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        query_results = []
        successful_assessments = 0
        
        for result in results:
            if isinstance(result, Exception):
                log.error(f"Query assessment failed with exception: {result}")
                continue
            query_results.append(result)  # type: ignore
            successful_assessments += 1
        
        log.info(f"Completed parallel assessment: {successful_assessments}/{len(queries)} queries processed successfully")
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

        Returns:
            Dictionary with 'semantic_neighbors', 'centroid_neighbors', and 'overlapping_chunks' lists.
        """
        # Filter chunks that have embeddings
        chunks_with_embeddings = [
            unit for unit in cluster.text_units if unit.text_embedding is not None
        ]

        if not chunks_with_embeddings:
            log.warning(f"Cluster {cluster.id} has no chunks with embeddings")
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
    json_data = []
    for result in reference_results:
        json_result = result.to_dict(include_clusters=False)  # Clusters stored separately
        json_data.append(json_result)
    
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
    
    with open(filepath, 'w', encoding='utf-8') as f:
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
        
    Returns:
        Tuple of (QueryClusterReferenceResult objects, TextCluster objects or None if no clusters were saved).
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert back to QueryClusterReferenceResult objects
    batch_results = []
    for result_data in data["references"]:
        batch_result = QueryClusterReferenceResult.from_dict(result_data)
        batch_results.append(batch_result)

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

    log.info(f"Loaded query cluster reference results for {len(batch_results)} questions from {filepath}")
    if clusters is not None:
        log.info(f"Loaded {len(clusters)} clusters from {filepath}")
    
    return batch_results, clusters
