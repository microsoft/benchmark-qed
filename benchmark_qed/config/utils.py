# Copyright (c) 2025 Microsoft Corporation.
"""Utility functions for loading and parsing configuration files."""

import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from dotenv import load_dotenv

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar("T", bound="BaseModel")

logger: logging.Logger = logging.getLogger(__name__)


def __load_env(env_path: Path) -> None:
    """Load environment variables from a .env file."""
    dotenv_path = env_path / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        msg = f".env file not found at {dotenv_path}"
        logger.warning(msg)


def __parse_env_vars(text: str) -> str:
    # remove comments from the file
    text = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
    return Template(text).substitute(os.environ)


def _parse(file_extension: str, contents: str) -> dict[str, Any]:
    """Parse configuration."""
    match file_extension:
        case ".yaml" | ".yml":
            return yaml.safe_load(contents)
        case ".json":
            return json.loads(contents)
        case _:
            msg = (
                f"Unable to parse config. Unsupported file extension: {file_extension}"
            )
            raise ValueError(msg)


def load_config(config_path: Path | str, factory: Callable[..., T]) -> T:
    """
    Load the configuration from a JSON file.

    Args
    ----
        config_path: The path to the JSON file containing the configuration.

    Returns
    -------
        A ScoresConfig instance.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Configuration file {config_path} does not exist."
        raise FileNotFoundError(msg)
    if config_path.suffix not in [".json", ".yaml"]:
        msg = f"Configuration file {config_path} must be a JSON or YAML file."
        raise ValueError(msg)

    __load_env(config_path.parent)
    contents = config_path.read_text()
    contents = __parse_env_vars(contents)
    contents = _parse(config_path.suffix, contents)
    return factory(**contents)


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
    return Template(file_path.read_text())
