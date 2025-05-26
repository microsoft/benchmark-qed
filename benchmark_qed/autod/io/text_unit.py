# Copyright (c) 2025 Microsoft Corporation.
"""Functions to chunk documents into text units and (optionally) embed them."""

from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.document import Document
from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autod.data_processor.embedding import TextEmbedder
from benchmark_qed.autod.data_processor.text_splitting import TextSplitter


async def create_text_units(
    documents: list[Document],
    text_embedder: TextEmbedder,
    text_splitter: TextSplitter,
    metadata_tags: list[str] | None = None,
    embedding_batch_size: int = defs.EMBEDDING_BATCH_SIZE,
    embed_text: bool = True,
    embedding_kwargs: dict[str, Any] | None = None,
) -> list[TextUnit]:
    """Create text units from documents."""
    text_units: list[TextUnit] = []

    text_id = 0
    for document in documents:
        chunks = text_splitter.split_text(document.text)

        # append document attributes to each chunk
        if document.attributes is not None and metadata_tags is not None:
            metadata = "\n".join([
                f"{key}: {document.attributes[key]}"
                for key in metadata_tags
                if key in document.attributes
            ])
        else:
            metadata = ""

        # Create text units from the chunks.
        for chunk in chunks:
            text = f"{metadata}\n{chunk}" if metadata != "" else chunk
            text_unit = TextUnit(
                id=str(uuid4()),
                short_id=str(text_id),
                n_tokens=text_splitter.get_size(text),
                text=text,
                document_id=document.id,
                attributes=document.attributes,
            )
            text_units.append(text_unit)
            text_id += 1

    # Embed the text units if requested.
    if embed_text:
        text_units = await text_embedder.embed_batch(
            text_units=text_units,
            batch_size=embedding_batch_size,
            **(embedding_kwargs if embedding_kwargs else {}),
        )

    return text_units


def load_text_units(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str = "short_id",
    text_col: str = "text",
    tokens_col: str | None = "n_tokens",
    embedding_col: str | None = "text_embedding",
    document_id_col: str | None = "document_id",
    cluster_id_col: str | None = "cluster_id",
    attributes_cols: list[str] | None = None,
) -> list[TextUnit]:
    """Read text units from a dataframe using pre-converted records."""
    records = df.to_dict("records")

    return [
        TextUnit(
            id=row.get(id_col, str(uuid4())),
            short_id=row.get(short_id_col, str(index)),
            text=row.get(text_col, ""),
            n_tokens=row.get(tokens_col, None),
            document_id=row.get(document_id_col, None),
            text_embedding=row.get(embedding_col, None),
            cluster_id=row.get(cluster_id_col, None),
            # Use the attributes_cols to extract attributes from the row
            attributes=(
                {col: row.get("attributes", {}).get(col) for col in attributes_cols}
                if attributes_cols
                else None
            ),
        )
        for index, row in enumerate(records)
    ]


def save_text_units(
    text_units: list[TextUnit],
    output_path: str,
    output_name: str = defs.TEXT_UNIT_OUTPUT,
) -> pd.DataFrame:
    """Save a list of TextUnit objects to a parquet file in the specified directory."""
    output_path_obj = Path(output_path)
    if not output_path_obj.exists():
        output_path_obj.mkdir(parents=True, exist_ok=True)

    output_file = output_path_obj / f"{output_name}.parquet"
    df = pd.DataFrame([asdict(tu) for tu in text_units])
    df.to_parquet(output_file, index=False)
    return df
