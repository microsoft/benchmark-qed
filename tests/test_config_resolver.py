# Copyright (c) 2025 Microsoft Corporation.
"""Tests for the blob:// config-path resolver."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from benchmark_qed.cli.config_resolver import (
    is_blob_uri,
    parse_blob_uri,
    resolve_config_path,
)


def test_is_blob_uri_detects_double_slash() -> None:
    assert is_blob_uri("blob://my-container/path/to/settings.yaml")


def test_is_blob_uri_detects_single_slash_path_form() -> None:
    # pathlib.Path("blob://x/y") collapses to "blob:/x/y" on POSIX.
    assert is_blob_uri("blob:/my-container/path/to/settings.yaml")


def test_is_blob_uri_rejects_local_path() -> None:
    assert not is_blob_uri("/tmp/settings.yaml")
    assert not is_blob_uri("./settings.yaml")
    assert not is_blob_uri(Path("settings.yaml"))


def test_parse_blob_uri_double_slash() -> None:
    container, key = parse_blob_uri("blob://my-container/dir/settings.yaml")
    assert container == "my-container"
    assert key == "dir/settings.yaml"


def test_parse_blob_uri_single_slash() -> None:
    container, key = parse_blob_uri("blob:/my-container/dir/settings.yaml")
    assert container == "my-container"
    assert key == "dir/settings.yaml"


def test_parse_blob_uri_invalid_raises() -> None:
    with pytest.raises(typer.BadParameter):
        parse_blob_uri("blob://only-container")
    with pytest.raises(typer.BadParameter):
        parse_blob_uri("not-a-blob-uri")


def test_resolve_config_path_local_passthrough(tmp_path: Path) -> None:
    local = tmp_path / "settings.yaml"
    local.write_text("input: {}", encoding="utf-8")
    result = resolve_config_path(local)
    assert result == local


def test_resolve_config_path_blob_downloads_tree(monkeypatch) -> None:
    """resolve_config_path downloads the settings file and sibling files."""
    monkeypatch.setenv(
        "AZURE_STORAGE_ACCOUNT_URL", "https://acct.blob.core.windows.net"
    )
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)

    fake_storage = MagicMock()
    # Storage is created with base_dir=parent_key, so find() returns paths
    # already relative to that base_dir (matching the real AzureBlobStorage
    # behavior of stripping base_dir from blob names).
    fake_storage.find.return_value = iter([
        "settings.yaml",
        ".env",
        "prompts/foo.txt",
        "prompts/sub/bar.txt",
    ])

    def fake_get(key: str, as_bytes: bool | None = False) -> bytes:
        return f"content of {key}".encode()

    fake_storage.get = AsyncMock(side_effect=fake_get)

    with patch(
        "benchmark_qed.cli.config_resolver.create_storage",
        return_value=fake_storage,
    ):
        result = resolve_config_path("blob://my-container/autoq_test/settings.yaml")

    assert result.name == "settings.yaml"
    assert result.exists()
    assert result.read_text(encoding="utf-8") == "content of settings.yaml"

    root = result.parent
    assert (root / ".env").read_text(encoding="utf-8") == "content of .env"
    assert (root / "prompts/foo.txt").read_text(encoding="utf-8") == (
        "content of prompts/foo.txt"
    )
    assert (root / "prompts/sub/bar.txt").read_text(encoding="utf-8") == (
        "content of prompts/sub/bar.txt"
    )


def test_resolve_config_path_blob_missing_credentials(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_URL", raising=False)
    with pytest.raises(typer.BadParameter, match="AZURE_STORAGE"):
        resolve_config_path("blob://my-container/autoq/settings.yaml")


def test_resolve_config_path_blob_missing_settings_file(monkeypatch) -> None:
    monkeypatch.setenv(
        "AZURE_STORAGE_ACCOUNT_URL", "https://acct.blob.core.windows.net"
    )
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)

    fake_storage = MagicMock()
    fake_storage.find.return_value = iter(["prompts/foo.txt"])
    fake_storage.get = AsyncMock(return_value=b"x")

    with (
        patch(
            "benchmark_qed.cli.config_resolver.create_storage",
            return_value=fake_storage,
        ),
        pytest.raises(typer.BadParameter, match="not found"),
    ):
        resolve_config_path("blob://my-container/autoq/settings.yaml")
