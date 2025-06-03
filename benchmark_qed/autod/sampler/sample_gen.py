# Copyright (c) 2025 Microsoft Corporation.
"""Module that uses AutoD functions to create a clustered sample of source texts to be used for downstream tasks such as question generation."""

import logging
from dataclasses import dataclass

from rich.status import Status

import benchmark_qed.config.defaults as dfs
from benchmark_qed.autod.data_model.document import Document
from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.data_processor.text_splitting import TokenTextSplitter
from benchmark_qed.autod.io.document import (
    create_documents,
    save_documents,
)
from benchmark_qed.autod.io.text_unit import create_text_units, save_text_units
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster
from benchmark_qed.autod.sampler.enums import ClusterRepresentativeSelectionType
from benchmark_qed.autod.sampler.sampling.kmeans_sampler import KmeansTextSampler

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class ClusteredSample:
    """Data class for storing the clustered sample of text units."""

    documents: list[Document]
    text_units: list[TextUnit]
    sample_texts: list[TextUnit]


async def acreate_clustered_sample(
    input_path: str,
    output_path: str,
    text_embedder: TextEmbedder,
    num_clusters: int = dfs.NUM_CLUSTERS,
    num_samples_per_cluster: int = dfs.NUM_SAMPLES_PER_CLUSTER,
    input_type: str = dfs.INPUT_TYPE,
    text_tag: str = dfs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    chunk_size: int = dfs.CHUNK_SIZE,
    chunk_overlap: int = dfs.CHUNK_OVERLAP,
    file_encoding: str = dfs.FILE_ENCODING,
    token_encoding: str = dfs.ENCODING_MODEL,
    random_seed: int = dfs.RANDOM_SEED,
) -> ClusteredSample:
    """Create a clustered sample of text units from a given input folder."""
    with Status("Creating documents...") as status:
        status.update("Creating documents...")
        if metadata_tags is None:
            metadata_tags = []
        documents = create_documents(
            input_path=input_path,
            input_type=input_type,
            text_tag=text_tag,
            metadata_tags=metadata_tags,
            encoding=file_encoding,
        )
        msg = f"Document count: {len(documents)}"
        log.info(msg)
        save_documents(documents, output_path)
        status.update("Creating text units and embedding...")

        # split documents into text units and embed text units
        text_splitter = TokenTextSplitter(
            encoding_name=token_encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        text_units = await create_text_units(
            documents=documents,
            metadata_tags=metadata_tags,
            text_splitter=text_splitter,
            text_embedder=text_embedder,
            embed_text=True,
        )
        msg = f"Text unit count: {len(text_units)}"
        log.info(msg)
        save_text_units(text_units, output_path)
        status.update("Sampling text units...")
        # create a sample of clustered text units
        sampler = KmeansTextSampler(random_seed=random_seed)
        sampled_text_units = sampler.sample(
            text_units=text_units,
            sample_size=None,
            num_clusters=num_clusters,
            num_samples_per_cluster=num_samples_per_cluster,
            cluster_representative_selection_type=ClusterRepresentativeSelectionType.CENTROID,
        )

        save_text_units(sampled_text_units, output_path, dfs.SAMPLE_TEXTS_OUTPUT)
        msg = f"Sampled text unit count: {len(sampled_text_units)}"
        log.info(msg)

    return ClusteredSample(
        documents=documents, text_units=text_units, sample_texts=sampled_text_units
    )


def load_texts_to_clusters(text_units: list[TextUnit]) -> list[TextCluster]:
    """Convert a list of TextUnit objects to a list of TextCluster objects."""
    clusters: dict[str, TextCluster] = {}
    for text_unit in text_units:
        cluster_id = text_unit.cluster_id
        if cluster_id is None:
            msg = f"Text unit {text_unit.id} does not have a cluster ID."
            raise ValueError(msg)
        if cluster_id not in clusters:
            clusters[cluster_id] = TextCluster(id=cluster_id, text_units=[])
        clusters[cluster_id].text_units.append(text_unit)
    return list(clusters.values())
