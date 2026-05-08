# Copyright (c) 2025 Microsoft Corporation.
"""Resolve CLI config-file arguments that may live in Azure Blob Storage.

CLI commands such as ``benchmark-qed autoq <settings.yaml> <output>`` accept a
local filesystem path for the configuration file. To allow the configuration to
also live in Azure Blob Storage, this module exposes :func:`resolve_config_path`
which detects ``blob://`` URIs, downloads the configuration *and every sibling
file under the same prefix* (so prompt templates resolve correctly) into a
temporary directory, and returns the local path to the downloaded settings
file.

Authentication may be provided either as direct arguments to
:func:`resolve_config_path` (the CLI surfaces ``--account-url`` /
``--connection-string`` options) or via the environment variables
``AZURE_STORAGE_CONNECTION_STRING`` / ``AZURE_STORAGE_ACCOUNT_URL`` as a
fallback.

URI format: ``blob://<container>/<key>``. ``Pathlib`` collapses ``//`` to ``/``
when Typer coerces the argument, so ``blob:/<container>/<key>`` is also
accepted.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from graphrag_storage.storage_config import StorageConfig
from graphrag_storage.storage_factory import create_storage

if TYPE_CHECKING:
    from graphrag_storage import Storage

_BLOB_SCHEMES = ("blob://", "blob:/")


# Reusable Annotated type aliases for CLI commands that accept a config path
# which may be a blob:// URI. Adding these as keyword options on a command
# lets users pass credentials inline (no env vars required).
AccountUrlOption = Annotated[
    str | None,
    typer.Option(
        "--account-url",
        help=(
            "Azure Blob Storage account URL for managed-identity auth, used "
            "when the config path is a blob:// URI. Falls back to "
            "$AZURE_STORAGE_ACCOUNT_URL."
        ),
    ),
]

ConnectionStringOption = Annotated[
    str | None,
    typer.Option(
        "--connection-string",
        help=(
            "Azure Blob Storage connection string, used when the config path "
            "is a blob:// URI. Falls back to "
            "$AZURE_STORAGE_CONNECTION_STRING."
        ),
    ),
]


def is_blob_uri(path: Path | str) -> bool:
    """Return True when ``path`` looks like a ``blob://`` URI."""
    return str(path).startswith(_BLOB_SCHEMES)


def parse_blob_uri(uri: str) -> tuple[str, str]:
    """Parse ``blob://<container>/<key>`` into ``(container, key)``.

    Also accepts the single-slash form produced when the URI is round-tripped
    through :class:`pathlib.Path` (e.g. ``blob:/<container>/<key>``).
    """
    parts = Path(uri).parts
    if len(parts) < 3 or parts[0] != "blob:":
        msg = f"Invalid blob URI {uri!r}: expected blob://<container>/<key>"
        raise typer.BadParameter(msg)
    container = parts[1]
    key = "/".join(parts[2:])
    return container, key


def _create_blob_storage(
    container: str,
    *,
    base_dir: str | None = None,
    account_url: str | None = None,
    connection_string: str | None = None,
) -> Storage:
    """Create a blob :class:`Storage` from explicit args or environment vars."""
    connection_string = connection_string or os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    account_url = account_url or os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    if not connection_string and not account_url:
        msg = (
            "Cannot read blob:// config: pass --account-url or "
            "--connection-string, or set AZURE_STORAGE_ACCOUNT_URL / "
            "AZURE_STORAGE_CONNECTION_STRING in the environment."
        )
        raise typer.BadParameter(msg)
    config = StorageConfig(
        type="blob",
        container_name=container,
        connection_string=connection_string,
        account_url=account_url,
        base_dir=base_dir,
    )
    return create_storage(config)


async def _download_tree(storage: Storage, dest: Path) -> int:
    """Download every blob in ``storage`` (already scoped to a base_dir) into ``dest``.

    Returns the number of files downloaded.
    """
    keys = list(storage.find(re.compile(r".*")))
    count = 0
    for key in keys:
        if not key:
            continue
        local_path = dest / key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        data = await storage.get(key, as_bytes=True)
        if data is None:
            continue
        local_path.write_bytes(data if isinstance(data, bytes) else str(data).encode())
        count += 1
    return count


def resolve_config_path(
    path: Path | str,
    *,
    account_url: str | None = None,
    connection_string: str | None = None,
) -> Path:
    """Resolve a CLI config-file argument that may be a ``blob://`` URI.

    For local paths this returns ``Path(path)`` unchanged. For ``blob://`` URIs
    it downloads the configuration file *and every sibling file under the same
    prefix* (so referenced prompt templates resolve via relative paths) into a
    fresh temporary directory and returns the local path to the downloaded
    config file.

    ``account_url`` / ``connection_string`` take precedence over the
    corresponding ``AZURE_STORAGE_*`` environment variables.
    """
    if not is_blob_uri(path):
        return Path(path)

    container, key = parse_blob_uri(str(path))
    parent_key, _, filename = key.rpartition("/")
    if not filename:
        msg = f"Invalid blob URI {path!r}: missing config filename."
        raise typer.BadParameter(msg)

    storage = _create_blob_storage(
        container,
        base_dir=parent_key or None,
        account_url=account_url,
        connection_string=connection_string,
    )

    tmp_root = Path(tempfile.mkdtemp(prefix="benchmark-qed-blob-"))
    typer.echo(
        f"Downloading config from blob://{container}/"
        f"{parent_key + '/' if parent_key else ''}* to {tmp_root}"
    )
    downloaded = asyncio.get_event_loop().run_until_complete(
        _download_tree(storage, tmp_root)
    )

    local_settings = tmp_root / filename
    if not local_settings.exists():
        msg = (
            f"Config file not found in blob storage: {path} "
            f"(downloaded {downloaded} file(s) under prefix "
            f"{parent_key!r}; check the URI and credentials)."
        )
        raise typer.BadParameter(msg)
    return local_settings
