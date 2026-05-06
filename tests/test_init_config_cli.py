# Copyright (c) 2025 Microsoft Corporation.
"""Tests for config init scaffolding behavior."""

from pathlib import Path
from unittest.mock import patch

from benchmark_qed.cli.init_config import ConfigType, init


def test_init_autoq_default_uses_local_storage_template(tmp_path: Path) -> None:
    """Default init keeps blob storage examples commented out."""
    init(ConfigType.autoq, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "  # storage:\n" in settings
    assert "# output_storage:\n" in settings


def test_init_autoq_blob_scaffolds_active_storage_sections(tmp_path: Path) -> None:
    """Blob mode scaffolds active input/output storage sections for AutoQ."""
    init(ConfigType.autoq, tmp_path, storage_type="blob")

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "  storage:\n    type: blob\n    container_name: my-datasets" in settings
    assert "output_storage:\n  type: blob\n  container_name: my-output" in settings


def test_init_autoe_blob_scaffolds_active_storage_sections(tmp_path: Path) -> None:
    """Blob mode scaffolds active input/output storage sections for AutoE configs."""
    init(ConfigType.autoe_reference, tmp_path, storage_type="blob")

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "input_storage:\n  type: blob\n  container_name: my-datasets" in settings
    assert "output_storage:\n  type: blob\n  container_name: my-output" in settings


def test_init_autoq_blob_with_custom_values(tmp_path: Path) -> None:
    """Blob mode with custom values uploads settings with pre-filled values to blob storage."""
    with patch("benchmark_qed.cli.init_config._write_to_blob") as mock_write_blob:
        init(
            ConfigType.autoq,
            tmp_path,
            storage_type="blob",
            container_name="my-container",
            account_url="https://myaccount.blob.core.windows.net",
            base_dir="data/project1",
        )

    mock_write_blob.assert_called_once()
    kwargs = mock_write_blob.call_args.kwargs
    settings = kwargs["settings_content"]

    assert kwargs["container_name"] == "my-container"
    assert kwargs["account_url"] == "https://myaccount.blob.core.windows.net"
    assert kwargs["base_dir"] == "data/project1"
    assert "container_name: my-container" in settings
    assert "account_url: https://myaccount.blob.core.windows.net" in settings
    assert "base_dir: data/project1" in settings
    # No local files should be created
    assert not (tmp_path / "settings.yaml").exists()
    assert not (tmp_path / "input").exists()


def test_init_autoe_blob_with_connection_string(tmp_path: Path) -> None:
    """Blob mode with connection string uploads settings with the value to blob storage."""
    with patch("benchmark_qed.cli.init_config._write_to_blob") as mock_write_blob:
        init(
            ConfigType.autoe_pairwise,
            tmp_path,
            storage_type="blob",
            container_name="scoring-data",
            connection_string="DefaultEndpointsProtocol=https;AccountName=test",
        )

    mock_write_blob.assert_called_once()
    kwargs = mock_write_blob.call_args.kwargs
    settings = kwargs["settings_content"]

    assert kwargs["container_name"] == "scoring-data"
    assert (
        kwargs["connection_string"] == "DefaultEndpointsProtocol=https;AccountName=test"
    )
    assert "container_name: scoring-data" in settings
    assert (
        "connection_string: DefaultEndpointsProtocol=https;AccountName=test" in settings
    )
    # No local files should be created
    assert not (tmp_path / "settings.yaml").exists()
