# Copyright (c) 2025 Microsoft Corporation.
"""Synchronous helpers for reading and writing artifacts via a graphrag Storage.

These helpers exist so that pipeline functions (which are synchronous) can route
all of their file I/O through the same `Storage` abstraction used by the CLI,
allowing artifacts to be persisted to local filesystem or remote backends
(e.g. Azure Blob Storage) without changing the public function signatures.
"""

import asyncio
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from graphrag_storage import Storage
from graphrag_storage.file_storage import FileStorage
from graphrag_storage.storage_config import StorageConfig
from graphrag_storage.storage_factory import create_storage


def resolve_storage(output_storage: Storage | None, output_dir: Path | None) -> Storage:
    """Return ``output_storage`` when provided, otherwise build a FileStorage.

    When falling back to FileStorage, ``output_dir`` is created on disk so that
    subsequent writes succeed. ``output_dir`` must be provided in that case.
    """
    if output_storage is not None:
        return output_storage
    if output_dir is None:
        msg = "output_dir is required when output_storage is not provided"
        raise ValueError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    return FileStorage(base_dir=str(output_dir))


def _run(coro: object) -> object:
    """Run an async coroutine to completion from synchronous code."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def write_csv(
    storage: Storage, key: str, df: pd.DataFrame, *, index: bool = False
) -> None:
    """Write a DataFrame as CSV to storage."""
    _run(storage.set(key, df.to_csv(index=index)))


def write_json(
    storage: Storage, key: str, data: object, *, indent: int | None = None
) -> None:
    """Write JSON-serializable data to storage."""
    _run(storage.set(key, json.dumps(data, indent=indent)))


def write_text(storage: Storage, key: str, text: str) -> None:
    """Write raw text to storage."""
    _run(storage.set(key, text))


def read_csv(storage: Storage, key: str) -> pd.DataFrame:
    """Read a CSV file from storage and return as a DataFrame."""
    data = _run(storage.get(key))
    if data is None:
        msg = f"File not found in storage: {key}"
        raise FileNotFoundError(msg)
    return pd.read_csv(StringIO(data))  # type: ignore[arg-type]


def storage_has(storage: Storage, key: str) -> bool:
    """Check whether a key exists in storage."""
    return bool(_run(storage.has(key)))


def read_json_df(storage: Storage, key: str) -> pd.DataFrame:
    """Read a JSON file from storage and return as a DataFrame."""
    data = _run(storage.get(key))
    if data is None:
        msg = f"File not found in storage: {key}"
        raise FileNotFoundError(msg)
    return pd.read_json(StringIO(data))  # type: ignore[arg-type]


def read_json(storage: Storage, key: str) -> Any:
    """Read a JSON file from storage and return parsed data."""
    data = _run(storage.get(key))
    if data is None:
        msg = f"File not found in storage: {key}"
        raise FileNotFoundError(msg)
    return json.loads(data)  # type: ignore[arg-type]


def read_bytes(storage: Storage, key: str) -> bytes:
    """Read raw bytes from storage (for binary formats like parquet)."""
    data = _run(storage.get(key, as_bytes=True))
    if data is None:
        msg = f"File not found in storage: {key}"
        raise FileNotFoundError(msg)
    return data  # type: ignore[return-value]


def build_input_storage(
    storage_config: StorageConfig | None, default_root: Path | str | None = None
) -> Storage:
    """Build an input Storage instance.

    When ``storage_config`` is provided, returns a Storage built from it.
    Otherwise, returns a FileStorage rooted at ``default_root`` (or the current
    working directory when ``default_root`` is None).
    """
    if storage_config is not None:
        return create_storage(storage_config)
    base = str(default_root) if default_root is not None else "."
    return FileStorage(base_dir=base)
