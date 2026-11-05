# Copyright (c) 2025 Microsoft Corporation.
"""Data downloader CLI."""

import asyncio
import zipfile
from enum import StrEnum
from importlib.resources import files
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
    NEGATIVE_ASSERTION_EXAMPLES = "negative_assertion_examples"


def _get_dataset_url(dataset: Dataset) -> str:
    """Get the download URL for a dataset."""
    if dataset == Dataset.EXAMPLE_ANSWERS:
        return f"https://raw.githubusercontent.com/microsoft/benchmark-qed/refs/heads/main/docs/notebooks/{dataset}/raw_data.zip"
    return f"https://raw.githubusercontent.com/microsoft/benchmark-qed/refs/heads/main/datasets/{dataset}/raw_data.zip"


async def _upload_zip_to_storage(zip_bytes: bytes, storage: Storage) -> None:
    """Extract a zip archive and upload its contents to blob storage."""
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith("/"):
                continue
            data = zip_ref.read(name)
            await storage.set(name, data)


def _get_dataset_bytes(dataset: Dataset) -> bytes:
    """Read bundled synthetic examples or download an existing dataset archive."""
    if dataset == Dataset.NEGATIVE_ASSERTION_EXAMPLES:
        source = files("benchmark_qed.data").joinpath("negative_assertion_examples")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for source_name, target_name in (
                ("answers.json", "negative_assertion_answers.json"),
                ("assertions.json", "negative_assertions.json"),
                ("expected_scores.json", "negative_assertion_expected_scores.json"),
            ):
                archive.writestr(target_name, source.joinpath(source_name).read_bytes())
        return buffer.getvalue()

    response = requests.get(_get_dataset_url(dataset), timeout=60)
    response.raise_for_status()
    return response.content


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
            help="Storage type: 'blob' for Azure Blob Storage. Omit for local filesystem."
        ),
    ] = None,
    container_name: Annotated[
        str | None,
        typer.Option(help="The blob container name."),
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
    base_dir: Annotated[
        str | None,
        typer.Option(
            help="Base prefix in blob storage. Files will be stored as: base_dir/output_dir/. If omitted, files are stored under output_dir/ only."
        ),
    ] = None,
) -> None:
    """Download a dataset or copy the bundled synthetic negative assertion examples.

    For local filesystem, the dataset is extracted to `output_dir`.

    For Azure Blob Storage:
    - Files are stored under `{base_dir}/{output_dir}` if `base_dir` is provided.
    - Files are stored under `{output_dir}` if `base_dir` is omitted.

    This ensures blob storage mirrors the local directory structure.
    """
    if dataset == Dataset.NEGATIVE_ASSERTION_EXAMPLES:
        typer.echo(
            "These fictional examples are bundled with BenchmarkQED under the MIT license: https://github.com/microsoft/benchmark-qed/blob/main/LICENSE."
        )
    else:
        typer.echo(
            "By downloading this dataset, you agree to the terms of use described here: https://github.com/microsoft/benchmark-qed/blob/main/datasets/LICENSE."
        )
    typer.confirm(
        "Accept Terms?",
        abort=True,
    )

    if storage_type and storage_type != "blob":
        msg = f"Unsupported storage type: {storage_type!r}. Only 'blob' is supported."
        raise typer.BadParameter(msg)

    zip_bytes = _get_dataset_bytes(dataset)

    # Determine base directory for blob storage
    # Include output_dir in the path so blob storage mirrors local directory structure
    output_path = output_dir.as_posix().strip("./")
    storage_base_dir = f"{base_dir}/{output_path}" if base_dir else output_path or None

    if storage_type:
        config = StorageConfig(
            type=storage_type,
            container_name=container_name,
            account_url=account_url,
            connection_string=connection_string,
            base_dir=storage_base_dir,
        )
        storage = create_storage(config)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_upload_zip_to_storage(zip_bytes, storage))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset}.zip"
        output_file.write_bytes(zip_bytes)
        with zipfile.ZipFile(output_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        output_file.unlink()

    typer.echo(f"Dataset {dataset} downloaded to {output_dir}.")
