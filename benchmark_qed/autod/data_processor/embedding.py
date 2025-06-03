# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to embed a collection of text using OpenAI embedding model."""

import asyncio
from collections.abc import Generator
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.config.defaults import EMBEDDING_BATCH_SIZE
from benchmark_qed.llm.type.base import EmbeddingModel


class TextEmbedder:
    """Text embedder using an embedding model.

    This class provides methods to embed text strings and TextUnit objects
    using a provided embedding model.
    """

    def __init__(self, text_embedder: EmbeddingModel) -> None:
        """Initialize a TextEmbedder instance.

        Args:
            text_embedder (EmbeddingModel): An instance of an embedding model
                that implements the `embed` method.
        """
        self.text_embedder = text_embedder

    async def embed_raw_text(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single text string using the text embedder.

        Args:
            text (str): The text string to embed.
            **kwargs (Any): Additional keyword arguments to pass to the embedder.

        Returns
        -------
            list[float]: The embedding vector for the input text.
        """
        embeddings = await self.text_embedder.embed(text_list=[text], **kwargs)
        return embeddings[0]

    async def embed_text_unit(self, text_unit: TextUnit) -> TextUnit:
        """Embed a single TextUnit object.

        Updates the text_unit's text_embedding property with the embedding vector.

        Args:
            text_unit (TextUnit): The TextUnit object to embed.

        Returns
        -------
            TextUnit: The input TextUnit with its text_embedding property updated.
        """
        text_unit.text_embedding = await self.embed_raw_text(text_unit.text)
        return text_unit

    async def embed_batch(
        self,
        text_units: list[TextUnit],
        batch_size: int | None = EMBEDDING_BATCH_SIZE,
        **kwargs: Any,
    ) -> list[TextUnit]:
        """Embed a batch of TextUnit objects.

        Processes TextUnits in batches for more efficient embedding.

        Args:
            text_units (list[TextUnit]): List of TextUnit objects to embed.
            batch_size (int | None): Number of texts to process in each batch.
                If None, all texts will be processed in a single batch.
            **kwargs (Any): Additional keyword arguments to pass to the embedder.

        Returns
        -------
            list[TextUnit]: The input list of TextUnits with their text_embedding properties updated.
        """
        if batch_size is None:
            batch_size = len(text_units)

        def get_batch(
            texts: list[TextUnit], batch_size: int = EMBEDDING_BATCH_SIZE
        ) -> Generator[list[str], None, None]:
            """Yield successive n-sized chunks from text unit lists.

            Args:
                texts (list[TextUnit]): List of TextUnit objects to batch.
                batch_size (int): Size of each batch.

            Yields
            ------
                list[str]: A batch of text strings extracted from TextUnit objects.
            """
            for i in range(0, len(texts), batch_size):
                yield [text.text for text in texts[i : i + batch_size]]

        tasks = [
            self.text_embedder.embed(text_list=batch, **kwargs)
            for batch in get_batch(text_units, batch_size)
        ]
        results = await asyncio.gather(*tasks)
        results = [item for sublist in results for item in sublist]

        for embedding, text_unit in zip(results, text_units, strict=False):
            text_unit.text_embedding = embedding

        return text_units
