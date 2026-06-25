# Copyright (c) 2025 Microsoft Corporation.
"""Tests for the graphrag_storage POSIX-key compatibility shim."""

from benchmark_qed.storage_compat import _posix_join, apply_blob_storage_posix_keys


def test_posix_join_uses_forward_slashes() -> None:
    assert _posix_join("ui_test/autoq", "_local/.env") == "ui_test/autoq/_local/.env"


def test_posix_join_normalizes_backslashes() -> None:
    assert _posix_join("ui_test\\autoq", "_local\\.env") == "ui_test/autoq/_local/.env"


def test_posix_join_without_base_dir() -> None:
    assert _posix_join("", "settings.yaml") == "settings.yaml"
    assert _posix_join(None, "settings.yaml") == "settings.yaml"


def test_keyname_patch_produces_forward_slashes() -> None:
    apply_blob_storage_posix_keys()

    from graphrag_storage.azure_blob_storage import AzureBlobStorage

    # Build a bare instance without triggering Azure connectivity.
    storage = object.__new__(AzureBlobStorage)
    storage._base_dir = "ui_test/autoq"  # type: ignore[attr-defined]

    assert storage._keyname("_local/.env") == "ui_test/autoq/_local/.env"

    storage._base_dir = "ui_test\\autoq"  # type: ignore[attr-defined]
    assert storage._keyname("sample_texts.parquet") == "ui_test/autoq/sample_texts.parquet"
