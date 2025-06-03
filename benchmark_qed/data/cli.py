# Copyright (c) 2025 Microsoft Corporation.
"""Data downloader CLI."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import requests
import typer
from rich.progress import Progress

app: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)


class Dataset(StrEnum):
    """Enum for the dataset type."""

    AP_NEWS = "AP_news"
    PODCAST = "podcast"
    EXAMPLE_ANSWERS = "example_answers"


def _download_folder(contents: list[dict], output_dir: Path) -> None:
    with Progress() as progress:
        task = progress.add_task("Downloading files...", total=len(contents))
        for item in contents:
            item_name = item["name"]
            download_url = item["download_url"]
            if item["type"] == "file":
                file_response = requests.get(download_url, timeout=60)
                (output_dir / item_name).write_bytes(file_response.content)
                typer.echo(f"Downloaded {item_name}")
            progress.update(task, advance=1)


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

    if dataset == Dataset.EXAMPLE_ANSWERS:
        api_url = f"https://api.github.com/repos/microsoft/benchmark-qed/contents/docs/notebooks/{dataset}"
        for subdir in ["graphrag_global", "lazygraphrag", "vector_rag"]:
            response = requests.get(f"{api_url}/{subdir}", timeout=60)
            contents = response.json()
            _download_folder(contents, output_dir / subdir)
    else:
        api_url = f"https://api.github.com/repos/microsoft/benchmark-qed/contents/datasets/{dataset}"
        response = requests.get(api_url, timeout=60)
        contents = response.json()
        _download_folder(contents, output_dir)
