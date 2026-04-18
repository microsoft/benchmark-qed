# Copyright (c) 2025 Microsoft Corporation.
"""Data downloader CLI."""

import asyncio
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


async def _upload_zip_to_storage(zip_bytes: bytes, storage: Storage) -> None:
    """Extract a zip archive and upload its contents to storage."""
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith("/"):
                continue
            data = zip_ref.read(name)
            # Use utf-8-sig to handle BOM in files (required for Cosmos DB storage)
            # Replace path separators with underscores for Cosmos DB compatibility (no slashes allowed in IDs)
            cosmos_safe_key = name.replace("/", "_")
            await storage.set(cosmos_safe_key, data.decode("utf-8-sig"))


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
) -> None:
    """Download the specified dataset from the GitHub repository."""
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

    if storage_type:
        config = StorageConfig(
            type=storage_type,
            container_name=container_name,
            account_url=account_url,
            connection_string=connection_string,
            database_name=database_name,
            base_dir=output_dir.as_posix().strip("./") or None,
        )
        storage = create_storage(config)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_upload_zip_to_storage(response.content, storage))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset}.zip"
        output_file.write_bytes(response.content)
        with zipfile.ZipFile(output_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        output_file.unlink()

    typer.echo(f"Dataset {dataset} downloaded to {output_dir}.")
