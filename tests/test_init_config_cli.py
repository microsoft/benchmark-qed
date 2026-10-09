# Copyright (c) 2025 Microsoft Corporation.
"""Tests for config init scaffolding behavior."""

from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pytest
import yaml

from benchmark_qed.autoe.cli import _print_calibration_output_guide
from benchmark_qed.autoe.config import (
    CalibratedAbsoluteCalibrationConfig,
    CalibratedAbsoluteScoringConfig,
)
from benchmark_qed.cli.init_config import ConfigType, init


def test_init_autoq_default_uses_local_storage_template(tmp_path: Path) -> None:
    """Default init keeps blob storage examples commented out."""
    init(ConfigType.autoq, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "  # storage:\n" in settings
    assert "# output_storage:\n" in settings


def test_calibration_output_guide_lists_detailed_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_calibration_output_guide(Path("output"))

    captured = capsys.readouterr()
    assert "Detailed calibration results" in captured.out
    assert "output/calibration.json" in captured.out
    assert "Elo ratings, level assignments" in captured.out
    assert "output/calibration_summary.csv" in captured.out
    assert "output/model_usage.json" in captured.out
    assert "Keep calibration.json unchanged" in captured.out


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


def test_init_chunk_assertion_scaffolds_cache_config(tmp_path: Path) -> None:
    init(ConfigType.autoe_chunk_assertion, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert (
        "retrieval_path: "
        "input/vector_rag_short_context/data_local_retrieval_results.json" in settings
    )
    assert "assertions_path: input/data_local_assertions.json" in settings
    assert "cache_config:\n  type: sqlite" in settings
    assert "base_dir: .benchmark_qed_cache/chunk_assertions" in settings
    assert "database_name: chunk_assertions.sqlite3" in settings
    assert "cache_dir:" not in settings


def test_init_pairwise_scaffolds_cache_config(tmp_path: Path) -> None:
    init(ConfigType.autoe_pairwise, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "cache_config:\n  type: sqlite" in settings
    assert "base_dir: .benchmark_qed_cache/pairwise" in settings
    assert "database_name: pairwise.sqlite3" in settings
    assert "Set type: none and storage: null" in settings


def test_init_differential_pairwise_scaffolds_stage_cache(tmp_path: Path) -> None:
    init(ConfigType.autoe_differential_pairwise, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "cache_config:\n  type: sqlite" in settings
    assert "base_dir: .benchmark_qed_cache/differential_pairwise" in settings
    assert "database_name: differential_pairwise.sqlite3" in settings
    assert "Caches extraction and verdict stages independently" in settings


def test_init_calibrated_absolute_calibration_settings(tmp_path: Path) -> None:
    init(ConfigType.autoe_absolute_calibrate, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "calibration:\n  - name: vector_rag" in settings
    assert "answer_base_path: input/vector_rag/data_local.json" in settings
    assert "answer_base_path: input/lazygraphrag/data_local.json" in settings
    assert "appearances: 3" in settings
    assert "max_batch_size: 5" in settings
    assert "base_dir: .benchmark_qed_cache/absolute_calibrate" in settings
    assert "database_name: absolute_calibrate.sqlite3" in settings
    assert "llm_config:" in settings
    config = CalibratedAbsoluteCalibrationConfig.model_validate(
        yaml.safe_load(settings)
    )
    assert len(config.calibration) == 2


def test_init_calibrated_absolute_scoring_settings(tmp_path: Path) -> None:
    init(ConfigType.autoe_absolute_score, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "calibration_path: ../calibration/output/calibration.json" in settings
    assert "use an absolute path or a path relative" in settings
    assert "With input_storage enabled" in settings
    assert "generated:\n  - name: graphrag_global" in settings
    assert "answer_base_path: input/graphrag_global/data_local.json" in settings
    assert "passes: 3" in settings
    assert "base_dir: .benchmark_qed_cache/absolute_score" in settings
    assert "database_name: absolute_score.sqlite3" in settings
    assert "llm_config:" in settings
    config = CalibratedAbsoluteScoringConfig.model_validate(yaml.safe_load(settings))
    assert config.generated[0].name == "graphrag_global"


def test_init_calibrated_absolute_scoring_blob_uses_storage_key(
    tmp_path: Path,
) -> None:
    init(ConfigType.autoe_absolute_score, tmp_path, storage_type="blob")

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "calibration_path: output/calibration.json" in settings
    assert "../calibration" not in settings


def test_calibrated_absolute_example_paths_exist_in_download_archive(
    tmp_path: Path,
) -> None:
    init(ConfigType.autoe_absolute_calibrate, tmp_path / "calibration")
    init(ConfigType.autoe_absolute_score, tmp_path / "scoring")
    calibration = yaml.safe_load(
        (tmp_path / "calibration/settings.yaml").read_text(encoding="utf-8")
    )
    scoring = yaml.safe_load(
        (tmp_path / "scoring/settings.yaml").read_text(encoding="utf-8")
    )
    archive = Path(__file__).parents[1] / "docs/notebooks/example_answers/raw_data.zip"

    with ZipFile(archive) as example_answers:
        names = set(example_answers.namelist())

    configured_paths = [
        condition["answer_base_path"].removeprefix("input/")
        for condition in calibration["calibration"] + scoring["generated"]
    ]
    assert set(configured_paths) <= names


def test_init_reference_scaffolds_judgment_cache(tmp_path: Path) -> None:
    init(ConfigType.autoe_reference, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "cache_config:\n  type: sqlite" in settings
    assert "base_dir: .benchmark_qed_cache/reference" in settings
    assert "database_name: reference.sqlite3" in settings
    assert "Reuses completed question/criterion/trial judgments" in settings


def test_init_assertion_scaffolds_judgment_cache(tmp_path: Path) -> None:
    init(ConfigType.autoe_assertion, tmp_path)

    settings = (tmp_path / "settings.yaml").read_text(encoding="utf-8")

    assert "cache_config:\n  type: sqlite" in settings
    assert "base_dir: .benchmark_qed_cache/assertion" in settings
    assert "database_name: assertion.sqlite3" in settings
    assert "Reuses completed question/assertion/trial judgments" in settings


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
