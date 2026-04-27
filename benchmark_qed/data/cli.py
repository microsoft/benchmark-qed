# Copyright (c) 2025 Microsoft Corporation.
"""Data downloader CLI."""

import asyncio
import base64
import zipfile
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Annotated

import requests
import typer
from graphrag_storage import Storage
from graphrag_storage.storage_config import StorageConfig
from graphrag_storage.storage_factory import create_storage

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)


class Dataset(StrEnum):
    """Enum for the dataset type."""

    AP_NEWS = "AP_news"
    PODCAST = "podcast"
    EXAMPLE_ANSWERS = "example_answers"


def _get_dataset_url(dataset: Dataset) -> str:
    """Get the download URL for a dataset."""
    if dataset == Dataset.EXAMPLE_ANSWERS:
        return f"https://raw.githubusercontent.com/microsoft/benchmark-qed/refs/heads/main/docs/notebooks/{dataset}/raw_data.zip"
    return f"https://raw.githubusercontent.com/microsoft/benchmark-qed/refs/heads/main/datasets/{dataset}/raw_data.zip"


async def _upload_zip_to_storage(
    zip_bytes: bytes, storage: Storage, storage_type: str = "blob"
) -> None:
    """Extract a zip archive and upload its contents to storage.

    Args:
        zip_bytes: The zip file contents as bytes.
        storage: The Storage backend to upload to.
        storage_type: The storage type ('blob', 'cosmosdb', or 'file').
                     Used to determine path sanitization rules.
    """
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith("/"):
                continue
            data = zip_ref.read(name)

            # Cosmos DB requires special handling: can't have slashes in IDs
            if storage_type == "cosmosdb":
                # Replace path separators with underscores for Cosmos DB ID compatibility
                safe_key = name.replace("/", "_")
                # Try to decode as UTF-8; fall back to base64 for binary files
                try:
                    content = data.decode("utf-8-sig")
                except UnicodeDecodeError:
                    content = base64.b64encode(data).decode("ascii")
                await storage.set(safe_key, content)
            else:
                # Blob storage can handle raw bytes directly
                await storage.set(name, data)


@app.command()
def download(
    dataset: Annotated[
        Dataset,
        typer.Argument(help="The dataset to download."),
    ],
    output_dir: Annotated[
        Path, typer.Argument(help="The directory to save the downloaded dataset.")
    ],
    *,
    storage_type: Annotated[
        str | None,
        typer.Option(
            help="Storage type: 'blob' for Azure Blob Storage, 'cosmosdb' for Azure Cosmos DB. Omit for local filesystem."
        ),
    ] = None,
    container_name: Annotated[
        str | None,
        typer.Option(help="The blob container or Cosmos DB container name."),
    ] = None,
    account_url: Annotated[
        str | None,
        typer.Option(help="The storage account URL (uses managed identity)."),
    ] = None,
    connection_string: Annotated[
        str | None,
        typer.Option(
            help="The storage connection string (alternative to account_url)."
        ),
    ] = None,
    database_name: Annotated[
        str | None,
        typer.Option(help="The Cosmos DB database name (required for cosmosdb type)."),
    ] = None,
    base_dir: Annotated[
        str | None,
        typer.Option(
            help="Base prefix in cloud storage. Cloud files will be stored as: base_dir/output_dir/. If omitted, files are stored under output_dir/ only."
        ),
    ] = None,
) -> None:
    """Download the specified dataset from the GitHub repository.

    For local filesystem, the dataset is extracted to `output_dir`.

    For cloud storage (blob/Cosmos DB):
    - Files are stored under `{base_dir}/{output_dir}` if `base_dir` is provided.
    - Files are stored under `{output_dir}` if `base_dir` is omitted.

    This ensures cloud storage mirrors the local directory structure.
    """
    typer.echo(
        "By downloading this dataset, you agree to the terms of use described here: https://github.com/microsoft/benchmark-qed/blob/main/datasets/LICENSE."
    )
    typer.confirm(
        "Accept Terms?",
        abort=True,
    )

    api_url = _get_dataset_url(dataset)
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()

    # Determine base directory for cloud storage
    # Include output_dir in the path so cloud storage mirrors local directory structure
    output_path = output_dir.as_posix().strip("./")
    storage_base_dir = f"{base_dir}/{output_path}" if base_dir else output_path or None

    if storage_type:
        config = StorageConfig(
            type=storage_type,
            container_name=container_name,
            account_url=account_url,
            connection_string=connection_string,
            database_name=database_name,
            base_dir=storage_base_dir,
        )
        storage = create_storage(config)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            _upload_zip_to_storage(response.content, storage, storage_type)
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset}.zip"
        output_file.write_bytes(response.content)
        with zipfile.ZipFile(output_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        output_file.unlink()

    typer.echo(f"Dataset {dataset} downloaded to {output_dir}.")
