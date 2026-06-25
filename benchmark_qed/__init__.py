# Copyright (c) 2025 Microsoft Corporation.
"""Answer eval module."""

from benchmark_qed.storage_compat import apply_blob_storage_posix_keys

# Normalize Azure Blob Storage keys to POSIX separators so blob I/O works on
# Windows (graphrag_storage builds keys with pathlib.Path, yielding backslashes).
apply_blob_storage_posix_keys()
