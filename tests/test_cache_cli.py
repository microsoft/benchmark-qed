# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for cache inspection commands."""

import json
from pathlib import Path

from typer.testing import CliRunner

from benchmark_qed.__main__ import app
from benchmark_qed.cache import SQLiteCache


def test_cache_inspect_json(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.sqlite3"
    store = SQLiteCache(cache_path, "test")
    store.put_many(
        [("key", {"score": 3}, {"model": "test-model"})],
        identities={"key": ("logical", "config")},
    )

    result = CliRunner().invoke(app, ["cache", "inspect", str(cache_path), "--json"])

    assert result.exit_code == 0
    details = json.loads(result.stdout)
    assert details["schema_version"] == 2
    assert details["active_leases"] == 0
    assert details["namespaces"] == [
        {"namespace": "test", "entries": 1, "configurations": 1}
    ]
    assert details["recent_provenance"][0]["metadata"]["model"] == "test-model"
