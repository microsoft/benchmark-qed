import numpy as np
import pandas as pd
import tiktoken
from numpy.typing import NDArray

from benchmark_qed.autod.data_model.text_unit import TextUnit


class TextCluster:
    """A cluster of text units."""

    def __init__(self, id: str, text_units: list[TextUnit]):
        self.id = id
        self.text_units = text_units

    def get_centroid(self) -> NDArray[np.float64]:
        """
        Get the centroid of the cluster.
        """
        return np.mean(
            [
                unit.text_embedding
                for unit in self.text_units
                if unit.text_embedding is not None
            ],
            axis=0,
        )

    def get_centroid_neighbors(self, n: int) -> list[TextUnit]:
        """
        Get the n closest text units to the centroid of the cluster, using euclidean of the text embeddings.
        """
        centroid = self.get_centroid()
        neighbors = sorted(
            self.text_units,
            key=lambda unit: float(
                np.linalg.norm(np.array(unit.text_embedding) - centroid)
            )
            if unit.text_embedding is not None
            else float("inf"),
        )
        return neighbors[:n]

    def get_member_neighbors(self, text_unit: TextUnit, n: int) -> list[TextUnit]:
        """
        Get the n closest text units to the given text unit, using euclidean of the text embeddings.
        """

        def distance_key(unit: TextUnit) -> float:
            if unit.text_embedding is None or text_unit.text_embedding is None:
                return float("inf")
            return float(
                np.linalg.norm(
                    np.array(unit.text_embedding) - np.array(text_unit.text_embedding)
                )
            )

        neighbors = sorted(self.text_units, key=distance_key)
        return neighbors[:n]

    def get_representatives_by_shortest_distances(self, n: int) -> list[TextUnit]:
        """
        Get the n most representative text units in the cluster by selecting the text units with the shortest distances to all other text units in the cluster.
        """
        shortest_distances = []
        for unit in self.text_units:
            distances = [
                np.linalg.norm(
                    np.array(unit.text_embedding) - np.array(other.text_embedding)
                )
                for other in self.text_units
            ]
            shortest_distances.append((unit, sum(distances)))
        shortest_distances = sorted(shortest_distances, key=lambda x: x[1])
        return [unit for unit, _ in shortest_distances[:n]]

    def get_representatives_by_attribute_ranking(
        self, n: int, ranking_attributes: list[str], ascending: bool = False
    ) -> list[TextUnit]:
        """
        Get the n most representative text units in the cluster by selecting the text units with the highest attribute ranking
        """
        return sorted(
            self.text_units,
            key=lambda x: tuple(
                x.attributes[attr]
                for attr in ranking_attributes
                if x.attributes is not None and attr in x.attributes
            ),
            reverse=not ascending,
        )[:n]

    def get_cluster_token_size(self, encoder: tiktoken.Encoding) -> int:
        """
        Get the token size of the cluster using the specified encoding.
        """
        return sum([len(encoder.encode(unit.text)) for unit in self.text_units])

    def convert_to_text(self) -> str:
        """
        Convert the cluster to a text representation.
        """
        text_units = [
            {
                "id": unit.short_id,
                "text": unit.text,
            }
            for unit in self.text_units
        ]
        df = pd.DataFrame(text_units)
        return df.to_csv(index=False, sep="|")
