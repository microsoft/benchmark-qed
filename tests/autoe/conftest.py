# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared cache backend fixtures for evaluator integration tests."""

from pathlib import Path

import pytest
from graphrag_cache import CacheConfig, CacheType
from graphrag_storage import StorageConfig, StorageType


@pytest.fixture(
    params=[CacheType.Json, CacheType.Sqlite, CacheType.Memory, CacheType.Noop],
    ids=["json", "sqlite", "memory", "no-cache"],
)
def evaluation_cache_config(
    request: pytest.FixtureRequest, tmp_path: Path
) -> CacheConfig:
    """Configure each supported cache backend in an isolated test directory."""
    cache_type = request.param
    assert isinstance(cache_type, CacheType)
    return CacheConfig(
        type=cache_type,
        storage=(
            StorageConfig(type=StorageType.File, base_dir=str(tmp_path / "cache"))
            if cache_type in {CacheType.Json, CacheType.Sqlite}
            else None
        ),
    )
