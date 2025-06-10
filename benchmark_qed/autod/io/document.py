# Copyright (c) 2025 Microsoft Corporation.
"""Load input files into Document objects."""

import datetime
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

import benchmark_qed.config.defaults as defs
from benchmark_qed.autod.data_model.document import Document
from benchmark_qed.autod.io.enums import InputDataType


def load_json_doc(
    file_path: str,
    encoding: str = defs.FILE_ENCODING,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,  # max length in characters
) -> Document:
    """Load a JSON file and return a Document object."""
    data = json.loads(Path(file_path).read_text(encoding=encoding))
    text = data.get(text_tag, "")
    if max_text_length is not None:
        text = text[:max_text_length]

    metadata: dict[str, Any] = {}
    if metadata_tags is not None:
        for tag in metadata_tags:
            if tag in data:
                metadata[tag] = data[tag]

    if "date_created" not in metadata:
        metadata["date_created"] = datetime.datetime.now(tz=datetime.UTC).isoformat()

    return Document(
        id=str(uuid4()),
        short_id=None,
        title=str(file_path.replace(".json", "")),
        type="json",
        text=text,
        attributes=metadata,
    )


def load_json_dir(
    dir_path: str,
    encoding: str = defs.FILE_ENCODING,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load a directory of JSON files and return a list of Document objects."""
    documents = []
    for index, file_path in enumerate(Path(dir_path).rglob("*.json")):
        document = load_json_doc(
            file_path=str(file_path),
            encoding=encoding,
            text_tag=text_tag,
            metadata_tags=metadata_tags,
            max_text_length=max_text_length,
        )
        document.short_id = str(index)
        documents.append(document)
    return documents


def load_text_doc(
    file_path: str,
    encoding: str = defs.FILE_ENCODING,
    max_text_length: int | None = None,
) -> Document:
    """Read a text file and return its content."""
    text = Path(file_path).read_text(encoding=encoding)
    if max_text_length is not None:
        text = text[:max_text_length]
    return Document(
        id=str(uuid4()),
        short_id=None,
        title=str(file_path.replace(".txt", "")),
        type="text",
        text=text,
        attributes={"date_created": datetime.datetime.now(tz=datetime.UTC).isoformat()},
    )


def load_text_dir(
    dir_path: str,
    encoding: str = defs.FILE_ENCODING,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load a directory of text files and return a list of Document objects."""
    documents = []
    for index, file_path in enumerate(Path(dir_path).rglob("*.txt")):
        document = load_text_doc(
            file_path=str(file_path),
            encoding=encoding,
            max_text_length=max_text_length,
        )
        document.short_id = str(index)
        documents.append(document)
    return documents


def load_csv_doc(
    file_path: str,
    encoding: str = defs.FILE_ENCODING,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load a CSV file and return a Document object."""
    data_df = pd.read_csv(file_path, encoding=encoding)

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
            metadata["date_created"] = datetime.datetime.now(
                tz=datetime.UTC
            ).isoformat()

        documents.append(
            Document(
                id=str(uuid4()),
                short_id=str(index),
                title=str(file_path.replace(".csv", "")),
                type="csv",
                text=text,
                attributes=metadata,
            )
        )
    return documents


def load_csv_dir(
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
            load_csv_doc(
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


def create_documents(
    input_path: str,
    input_type: InputDataType | str = InputDataType.JSON,
    encoding: str = defs.FILE_ENCODING,
    text_tag: str = defs.TEXT_COLUMN,
    metadata_tags: list[str] | None = None,
    max_text_length: int | None = None,
) -> list[Document]:
    """Load documents from a specified path and return a list of Document objects."""
    input_path_obj = Path(input_path)
    if input_path_obj.is_dir():
        match input_type:
            case InputDataType.JSON:
                documents = load_json_dir(
                    dir_path=str(input_path),
                    encoding=encoding,
                    text_tag=text_tag,
                    metadata_tags=metadata_tags,
                    max_text_length=max_text_length,
                )
            case InputDataType.TEXT:
                documents = load_text_dir(
                    dir_path=str(input_path),
                    encoding=encoding,
                    max_text_length=max_text_length,
                )
            case InputDataType.CSV:
                documents = load_csv_dir(
                    dir_path=str(input_path),
                    encoding=encoding,
                    text_tag=text_tag,
                    metadata_tags=metadata_tags,
                    max_text_length=max_text_length,
                )
            case _:
                msg = f"Unsupported input type: {input_type}"
                raise ValueError(msg)
    else:
        match input_type:
            case InputDataType.JSON:
                documents = [
                    load_json_doc(
                        file_path=str(input_path),
                        encoding=encoding,
                        text_tag=text_tag,
                        metadata_tags=metadata_tags,
                        max_text_length=max_text_length,
                    )
                ]
            case InputDataType.TEXT:
                documents = [
                    load_text_doc(
                        file_path=str(input_path),
                        encoding=encoding,
                        max_text_length=max_text_length,
                    )
                ]
            case InputDataType.CSV:
                documents = load_csv_doc(
                    file_path=str(input_path),
                    encoding=encoding,
                    text_tag=text_tag,
                    metadata_tags=metadata_tags,
                    max_text_length=max_text_length,
                )
            case _:
                msg = f"Unsupported input type: {input_type}"
                raise ValueError(msg)
    return documents


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

    return [
        Document(
            id=row.get(id_col, str(uuid4())),
            short_id=row.get(short_id_col, str(index)),
            title=row.get(title_col, ""),
            type=row.get(type_col, ""),
            text=row.get(text_col, ""),
            attributes=(
                {col: row.get(col) for col in attributes_cols}
                if attributes_cols
                else {}
            ),
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
