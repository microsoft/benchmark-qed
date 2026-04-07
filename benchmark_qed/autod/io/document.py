# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Load input files into Document objects."""

import datetime
import re
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import pandas as pd
from graphrag_input import InputConfig, create_input_reader
from graphrag_input.text_document import TextDocument
from graphrag_storage.file_storage import FileStorage

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.document import Document
from benchmark_qed.autod.io.enums import InputDataType


def _clean_title(title: str) -> str:
    """Strip row-index suffix and file extension from a graphrag-input title."""
    title = re.sub(r" \(\d+\)$", "", title)
    return str(PurePosixPath(title).with_suffix(""))


def _to_document(
    text_document: TextDocument,
    input_type: InputDataType | str,
    index: int,
    metadata_tags: list[str] | None,
    max_text_length: int | None,
) -> Document:
    """Convert a graphrag-input TextDocument to a local Document."""
    text = text_document.text
    if max_text_length is not None:
        text = text[:max_text_length]

    attributes: dict[str, Any] = text_document.collect(metadata_tags or [])
    if "date_created" not in attributes:
        attributes["date_created"] = text_document.get("creation_date", text_document.creation_date)

    return Document(
        id=text_document.id,
        short_id=str(index),
        title=_clean_title(text_document.title),
        type=str(input_type),
        text=text,
        attributes=attributes,
    )


def _load_docs_from_dataframe(
    data_df: pd.DataFrame,
    input_type: InputDataType,
    title: str,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    documents: list[Document] = []

    for index, row in enumerate(data_df.itertuples()):
        text = getattr(row, text_tag, "")
        if max_text_length is not None:
            text = text[:max_text_length]

        metadata: dict[str, Any] = {}
        if metadata_tags is not None:
            for tag in metadata_tags:
                if tag in data_df.columns:
                    metadata[tag] = getattr(row, tag)

        if "date_created" not in metadata:
            metadata["date_created"] = datetime.datetime.now(tz=datetime.UTC).isoformat()

        documents.append(
            Document(
                id=str(uuid4()),
                short_id=str(index),
                title=title,
                type=str(input_type),
                text=text,
                attributes=metadata,
            )
        )
    return documents


def _load_parquet_doc(
    file_path: str,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load Documents from a parquet file."""
    return _load_docs_from_dataframe(
        data_df=pd.read_parquet(file_path),
        input_type=InputDataType.PARQUET,
        title=str(file_path.replace(".parquet", "")),
        text_tag=text_tag,
        metadata_tags=metadata_tags,
        max_text_length=max_text_length,
    )


def _load_parquet_dir(
    dir_path: str,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load a directory of parquet files and return a list of Document objects."""
    documents: list[Document] = []
    for file_path in Path(dir_path).rglob("*.parquet"):
        documents.extend(
            _load_parquet_doc(
                file_path=str(file_path),
                text_tag=text_tag,
                metadata_tags=metadata_tags,
                max_text_length=max_text_length,
            )
        )

    for index, document in enumerate(documents):
        document.short_id = str(index)

    return documents


def _load_csv_doc(
    file_path: str,
    encoding: str = defs.FILE_ENCODING,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load a CSV file and return a list of Document objects (preserves column types via pandas)."""
    return _load_docs_from_dataframe(
        data_df=pd.read_csv(file_path, encoding=encoding),
        input_type=InputDataType.CSV,
        title=str(file_path.replace(".csv", "")),
        text_tag=text_tag,
        metadata_tags=metadata_tags,
        max_text_length=max_text_length,
    )


def _load_csv_dir(
    dir_path: str,
    encoding: str = defs.FILE_ENCODING,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load a directory of CSV files and return a list of Document objects."""
    documents: list[Document] = []
    for file_path in Path(dir_path).rglob("*.csv"):
        documents.extend(
            _load_csv_doc(
                file_path=str(file_path),
                encoding=encoding,
                text_tag=text_tag,
                metadata_tags=metadata_tags,
                max_text_length=max_text_length,
            )
        )

    for index, document in enumerate(documents):
        document.short_id = str(index)

    return documents


async def create_documents(
    input_path: str,
    input_type: InputDataType | str = InputDataType.JSON,
    encoding: str = defs.FILE_ENCODING,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load documents from a specified path and return a list of Document objects."""
    input_path_obj = Path(input_path)

    if str(input_type) == InputDataType.PARQUET:
        if input_path_obj.is_dir():
            return _load_parquet_dir(str(input_path), text_tag, metadata_tags, max_text_length)
        return _load_parquet_doc(str(input_path), text_tag, metadata_tags, max_text_length)

    if str(input_type) == InputDataType.CSV:
        if input_path_obj.is_dir():
            return _load_csv_dir(str(input_path), encoding, text_tag, metadata_tags, max_text_length)
        return _load_csv_doc(str(input_path), encoding, text_tag, metadata_tags, max_text_length)

    # For JSON and TEXT: delegate to graphrag-input readers
    if input_path_obj.is_dir():
        base_dir = str(input_path_obj)
        file_pattern = None
    else:
        base_dir = str(input_path_obj.parent)
        file_pattern = re.escape(input_path_obj.name) + "$"

    storage = FileStorage(base_dir=base_dir, encoding=encoding)
    config = InputConfig(
        type=str(input_type),
        encoding=encoding,
        text_column=text_tag if str(input_type) != InputDataType.TEXT else None,
        file_pattern=file_pattern,
    )
    reader = create_input_reader(config, storage)
    text_documents = await reader.read_files()

    return [
        _to_document(td, input_type, index, metadata_tags, max_text_length)
        for index, td in enumerate(text_documents)
    ]



def load_documents(
    df: pd.DataFrame,
    id_col: str = "id",
    short_id_col: str = "short_id",
    title_col: str = "title",
    type_col: str = "type",
    text_col: str = "text",
    attributes_cols: list[str] | None = None,
) -> list[Document]:
    """Read documents from a dataframe using pre-converted records."""
    records = df.to_dict("records")

    def _get_attributes(row: dict) -> dict[str, Any]:
        attributes = row.get("attributes", {})
        selected_attributes = attributes_cols or []
        return {attr: attributes.get(attr, None) for attr in selected_attributes}

    return [
        Document(
            id=row.get(id_col, str(uuid4())),
            short_id=row.get(short_id_col, str(index)),
            title=row.get(title_col, ""),
            type=row.get(type_col, ""),
            text=row.get(text_col, ""),
            attributes=_get_attributes(row),
        )
        for index, row in enumerate(records)
    ]


def save_documents(
    documents: list[Document],
    output_path: str,
    output_name: str = defs.DOCUMENT_OUTPUT,
) -> pd.DataFrame:
    """Save a list of Document objects to a parquet file in the specified directory."""
    output_path_obj = Path(output_path)
    if not output_path_obj.exists():
        output_path_obj.mkdir(parents=True, exist_ok=True)

    output_file = output_path_obj / f"{output_name}.parquet"
    document_df = pd.DataFrame([asdict(doc) for doc in documents])
    document_df.to_parquet(output_file, index=False)
    return document_df
