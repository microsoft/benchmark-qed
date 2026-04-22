# Copyright (c) 2025 Microsoft Corporation.
"""Shared scaffolding utilities for config initialization."""

from pathlib import Path

import typer


def copy_prompts(prompts_path: Path, output_path: Path) -> None:
    """Copy prompt template files from a source directory to an output directory."""
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    for prompt_file in prompts_path.iterdir():
        if prompt_file.is_file() and prompt_file.suffix == ".txt":
            target_file = output_path / prompt_file.name
            target_file.write_text(
                prompt_file.read_text(encoding="utf-8"), encoding="utf-8"
            )


def write_env_file(root: Path) -> None:
    """Create a .env file with placeholder API key if it doesn't exist."""
    env_file = root / ".env"
    if not env_file.exists():
        env_file.write_text("OPENAI_API_KEY=<API_KEY>", encoding="utf-8")
    typer.echo(
        f"Change the OPENAI_API_KEY placeholder at {env_file} with your actual OPENAI_API_KEY."
    )


def ensure_input_folder(root: Path) -> None:
    """Create the input folder if it doesn't exist."""
    input_folder = root / "input"
    if not input_folder.exists():
        input_folder.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Input folder created at {input_folder}")
        typer.echo(
            "Please place your input files in the 'input' folder before running, "
            "or modify the settings.yaml to point to your input files."
        )
