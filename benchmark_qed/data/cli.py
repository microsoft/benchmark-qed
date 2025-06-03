# Copyright (c) 2025 Microsoft Corporation.
"""Data downloader CLI."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import requests
import typer

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)


class Dataset(StrEnum):
    """Enum for the dataset type."""

    AP_NEWS = "AP_news"
    PODCAST = "podcast"
    EXAMPLE_ANSWERS = "example_answers"


@app.command()
def download(
    dataset: Annotated[
        Dataset,
        typer.Argument(help="The dataset to download."),
    ],
    output_dir: Annotated[
        Path, typer.Argument(help="The directory to save the downloaded dataset.")
    ],
) -> None:
    """Download the specified dataset from the GitHub repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(
        "By downloading this dataset, you agree to the terms of use described here: https://github.com/microsoft/benchmark-qed/blob/main/datasets/LICENSE."
    )
    typer.confirm(
        "Accept Terms?",
        abort=True,
    )

    match dataset:
        case Dataset.EXAMPLE_ANSWERS:
            api_url = f"https://raw.githubusercontent.com/microsoft/benchmark-qed/refs/heads/main/docs/notebooks/{dataset}/raw_data.zip"
            response = requests.get(api_url, timeout=60)
            output_file = output_dir / f"{dataset}.zip"
            output_file.write_bytes(response.content)

        case Dataset.AP_NEWS | Dataset.PODCAST:
            api_url = f"https://raw.githubusercontent.com/microsoft/benchmark-qed/refs/heads/main/datasets/{dataset}/raw_data.zip"
            response = requests.get(api_url, timeout=60)
            output_file = output_dir / f"{dataset}.zip"
            output_file.write_bytes(response.content)
