# Copyright (c) Microsoft Corporation.
"""Tests for bundled negative assertion sample data and its CLI workflow."""

import json
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import yaml
from graphrag_llm.completion import LLMCompletion
from graphrag_storage import Storage
from typer.testing import CliRunner

from benchmark_qed.__main__ import app
from benchmark_qed.autoe import cli as autoe_cli
from benchmark_qed.autoe.assertion import standard
from benchmark_qed.autoe.data_model.assertion import AssertionLLMResponse
from benchmark_qed.data import cli as data_cli

SAMPLE_FILES = {
    "negative_assertion_answers.json",
    "negative_assertions.json",
    "negative_assertion_expected_scores.json",
}


@pytest.fixture
def sample_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    requests_get = MagicMock(side_effect=AssertionError("Samples must work offline."))
    monkeypatch.setattr(data_cli.requests, "get", requests_get)
    path = tmp_path / "input"
    result = CliRunner().invoke(
        app, ["data", "download", "negative_assertion_examples", str(path)], input="y\n"
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    requests_get.assert_not_called()
    assert {file.name for file in path.iterdir()} == SAMPLE_FILES
    return path


def test_sample_inputs_and_expected_scores_are_complete(sample_dir: Path) -> None:
    answers = json.loads(
        (sample_dir / "negative_assertion_answers.json").read_text(encoding="utf-8")
    )
    assertions = json.loads(
        (sample_dir / "negative_assertions.json").read_text(encoding="utf-8")
    )
    expected = json.loads(
        (sample_dir / "negative_assertion_expected_scores.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(answers) == len(assertions) == 6
    answers_by_id = {answer["question_id"]: answer for answer in answers}
    assert len(answers_by_id) == len(answers)
    for question in assertions:
        assert (
            question["question_text"]
            == answers_by_id[question["question_id"]]["question_text"]
        )
    assertion_keys = {
        (question["question_id"], assertion)
        for question in assertions
        for assertion in question["assertions"]
    }
    expected_keys = {(row["question_id"], row["assertion"]) for row in expected}
    assert assertion_keys == expected_keys
    assert len(assertion_keys) == len(expected) == 10
    assert [row["score"] for row in expected].count(1) == 6
    assert [row["score"] for row in expected].count(0) == 4


def test_bundled_samples_run_with_generated_config(
    sample_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the documented workflow; mocked scores do not establish judge accuracy."""
    root = sample_dir.parent
    runner = CliRunner()
    result = runner.invoke(
        app, ["config", "init", "autoe_negative_assertion", str(root)]
    )
    assert result.exit_code == 0, result.output
    config_path = root / "settings.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert root / config["generated"]["answer_base_path"] == (
        sample_dir / "negative_assertion_answers.json"
    )
    assert root / config["assertions"]["assertions_path"] == (
        sample_dir / "negative_assertions.json"
    )
    config["trials"] = 1
    config["cache_config"] = {"type": "none", "storage": None}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    expected = json.loads(
        (sample_dir / "negative_assertion_expected_scores.json").read_text(
            encoding="utf-8"
        )
    )
    chat = AsyncMock(
        side_effect=[
            SimpleNamespace(
                formatted_response=AssertionLLMResponse(
                    score=row["score"], reasoning="Mock sample judgment."
                )
            )
            for row in expected
        ]
    )
    monkeypatch.setattr(standard, "chat", chat)
    client = MagicMock(spec=LLMCompletion)
    client.metrics_store.get_metrics.return_value = {}
    monkeypatch.setattr(autoe_cli.ModelFactory, "create_chat_model", lambda _: client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    output = root / "output"

    result = runner.invoke(
        app, ["autoe", "negative-assertion-scores", str(config_path), str(output)]
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    scores = pd.read_csv(output / "assertion_scores.csv")
    assert scores["assertion"].tolist() == [row["assertion"] for row in expected]
    assert scores["score"].tolist() == [row["score"] for row in expected]
    assert chat.await_count == 10


def test_bundled_samples_support_blob_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = MagicMock(spec=Storage)
    storage.set = AsyncMock()
    create_storage = MagicMock(return_value=storage)
    monkeypatch.setattr(data_cli, "create_storage", create_storage)
    requests_get = MagicMock(side_effect=AssertionError("Samples must work offline."))
    monkeypatch.setattr(data_cli.requests, "get", requests_get)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "download",
            "negative_assertion_examples",
            "input",
            "--storage-type",
            "blob",
            "--container-name",
            "samples",
            "--account-url",
            "https://example.blob.core.windows.net",
            "--base-dir",
            "negative-test",
        ],
        input="y\n",
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    requests_get.assert_not_called()
    assert create_storage.call_args.args[0].base_dir == "negative-test/input"
    assert {call.args[0] for call in storage.set.await_args_list} == SAMPLE_FILES
    for call in storage.set.await_args_list:
        assert isinstance(json.loads(call.args[1]), list)


def test_declining_sample_terms_does_not_create_files(tmp_path: Path) -> None:
    output = tmp_path / "input"
    result = CliRunner().invoke(
        app,
        ["data", "download", "negative_assertion_examples", str(output)],
        input="n\n",
    )
    assert result.exit_code != 0
    assert not output.exists()


@pytest.mark.parametrize(
    "dataset",
    [
        data_cli.Dataset.AP_NEWS,
        data_cli.Dataset.PODCAST,
        data_cli.Dataset.EXAMPLE_ANSWERS,
    ],
)
def test_existing_datasets_still_download_and_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: data_cli.Dataset
) -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("example.json", "[]")
    response = MagicMock()
    response.content = buffer.getvalue()
    requests_get = MagicMock(return_value=response)
    monkeypatch.setattr(data_cli.requests, "get", requests_get)
    output = tmp_path / "input"
    result = CliRunner().invoke(
        app, ["data", "download", dataset.value, str(output)], input="y\n"
    )
    assert result.exit_code == 0, result.output
    requests_get.assert_called_once_with(data_cli._get_dataset_url(dataset), timeout=60)
    response.raise_for_status.assert_called_once()
    assert (output / "example.json").read_text(encoding="utf-8") == "[]"
    assert not (output / f"{dataset}.zip").exists()
