# Copyright (c) 2025 Microsoft Corporation.
"""Util functions to embed a collection of text using OpenAI embedding model"""

import asyncio
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.config.defaults import EMBEDDING_BATCH_SIZE
from benchmark_qed.llm.type.base import EmbeddingModel


class TextEmbedder:
    def __init__(self, text_embedder: EmbeddingModel):
        self.text_embedder = text_embedder

    async def embed_raw_text(self, text: str, **kwargs: Any) -> list[float]:
        embeddings = await self.text_embedder.embed(text_list=[text], **kwargs)
        return embeddings[0]

    async def embed_text_unit(self, text_unit: TextUnit) -> TextUnit:
        text_unit.text_embedding = await self.embed_raw_text(text_unit.text)
        return text_unit

    async def embed_batch(
        self,
        text_units: list[TextUnit],
        batch_size: int | None = EMBEDDING_BATCH_SIZE,
        **kwargs: Any,
    ) -> list[TextUnit]:
        if batch_size is None:
            batch_size = len(text_units)

        def get_batch(texts: list[TextUnit], batch_size: int = EMBEDDING_BATCH_SIZE):
            """Yield successive n-sized chunks from text unit lists"""
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
