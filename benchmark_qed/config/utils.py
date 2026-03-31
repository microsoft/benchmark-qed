# Copyright (c) 2025 Microsoft Corporation.
"""Utility functions for loading and parsing configuration files."""

from pathlib import Path
from string import Template


def load_template_file(file_path: Path) -> Template:
    """
    Load a template file and return its contents.

    Args
    ----
        file_path: The path to the template file.

    Returns
    -------
        The contents of the template file as a string.
    """
    if not file_path.exists():
        msg = f"Template file {file_path} does not exist."
        raise FileNotFoundError(msg)
    return Template(file_path.read_text(encoding="utf-8"))
