# Copyright (c) 2025 Microsoft Corporation.
"""IO utilities for saving and loading text clusters."""

import json
import logging
from pathlib import Path

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.sampler.clustering.cluster import TextCluster

log: logging.Logger = logging.getLogger(__name__)


def save_clusters_to_json(clusters: list[TextCluster], filepath: str | Path) -> None:
    """Save clusters to a JSON file.

    Args:
        clusters: List of TextCluster objects to save.
        filepath: Path to the JSON file where clusters will be saved.
    """
    filepath = Path(filepath)
    
    # Convert clusters to serializable format
    serializable_clusters = []
    for cluster in clusters:
        cluster_data = {
            "id": cluster.id,
            "text_units": []
        }
        
        for unit in cluster.text_units:
            unit_data = {
                "id": unit.id,
                "short_id": unit.short_id,
                "text": unit.text,
                "document_id": unit.document_id,
                "n_tokens": unit.n_tokens,
                "text_embedding": unit.text_embedding,
                "cluster_id": unit.cluster_id,
                "attributes": unit.attributes
            }
            cluster_data["text_units"].append(unit_data)
        
        serializable_clusters.append(cluster_data)
    
    # Save to JSON file
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(serializable_clusters, f, indent=4, ensure_ascii=False)
    
    log.info(f"Saved {len(clusters)} clusters to {filepath}")


def load_clusters_from_json(filepath: str | Path) -> list[TextCluster]:
    """Load clusters from a JSON file.

    Args:
        filepath: Path to the JSON file containing saved clusters.

    Returns:
        List of TextCluster objects loaded from the file.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If the JSON structure is invalid or missing required fields.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Cluster file not found: {filepath}")
    
    # Load from JSON file
    with filepath.open("r", encoding="utf-8") as f:
        cluster_data = json.load(f)
    
    # Convert back to TextCluster objects
    clusters = []
    for cluster_dict in cluster_data:
        text_units = []
        for unit_dict in cluster_dict["text_units"]:
            text_unit = TextUnit(
                id=unit_dict["id"],
                short_id=unit_dict["short_id"],
                text=unit_dict["text"],
                document_id=unit_dict["document_id"],
                n_tokens=unit_dict["n_tokens"],
                text_embedding=unit_dict["text_embedding"],
                cluster_id=unit_dict["cluster_id"],
                attributes=unit_dict["attributes"]
            )
            text_units.append(text_unit)
        
        cluster = TextCluster(
            id=cluster_dict["id"],
            text_units=text_units
        )
        clusters.append(cluster)
    
    log.info(f"Loaded {len(clusters)} clusters from {filepath}")
    return clusters
