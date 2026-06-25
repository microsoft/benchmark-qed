# Copyright (c) 2025 Microsoft Corporation.
"""Cross-platform compatibility shims for ``graphrag_storage``.

``graphrag_storage``'s ``AzureBlobStorage`` builds blob keys with
``pathlib.Path``, which emits OS-specific separators. On Windows this produces
backslash blob names (``a\\b\\c``) that do not match the forward-slash names
Azure Blob Storage actually uses, breaking every read and write (config
download, input read, and output write all fail).

These patches force POSIX (``/``) separators for blob key generation so blob
storage works identically on Windows and POSIX systems. They are idempotent and
safe to apply more than once.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphrag_storage.storage import Storage

_PATCHED = False


def _posix_join(base_dir: str | None, name: str) -> str:
    """Join ``base_dir`` and ``name`` using POSIX separators.

    Any backslashes already present (e.g. from a previously created child
    storage) are normalized to forward slashes so keys stay consistent.
    """
    normalized_name = str(name).replace("\\", "/")
    if not base_dir:
        return normalized_name
    normalized_base = str(base_dir).replace("\\", "/")
    return str(PurePosixPath(normalized_base) / normalized_name)


def apply_blob_storage_posix_keys() -> None:
    """Patch ``AzureBlobStorage`` to use POSIX separators for blob keys."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from graphrag_storage.azure_blob_storage import AzureBlobStorage
    except Exception:  # noqa: BLE001 - azure deps optional / library may change
        return

    if not hasattr(AzureBlobStorage, "_keyname") or not hasattr(
        AzureBlobStorage, "child"
    ):
        return

    def _keyname(self: AzureBlobStorage, key: str) -> str:
        return _posix_join(self._base_dir, key) if self._base_dir else key

    def child(self: AzureBlobStorage, name: str | None) -> "Storage":
        if name is None:
            return self
        path = _posix_join(self._base_dir, name)
        return AzureBlobStorage(
            connection_string=self._connection_string,
            container_name=self._container_name,
            encoding=self._encoding,
            base_dir=path,
            account_url=self._account_url,
        )

    AzureBlobStorage._keyname = _keyname  # type: ignore[method-assign]
    AzureBlobStorage.child = child  # type: ignore[method-assign]
    _PATCHED = True
