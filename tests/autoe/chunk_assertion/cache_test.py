# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the content-addressed (assertion, chunk) cache."""

import asyncio
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from graphrag_cache import CacheConfig, CacheType
from graphrag_storage import StorageConfig, StorageType

import benchmark_qed.cache as cache_module
from benchmark_qed.autoe.chunk_assertion.cache import (
    ContentAddressedCache,
    build_cache_metadata,
    compute_cache_key,
    compute_config_fingerprint,
    compute_logical_key,
)
from benchmark_qed.cache import CacheStore, create_default_cache_config, inspect_cache


def _cache_config(cache_path: Path | str) -> CacheConfig:
    path = Path(cache_path)
    return create_default_cache_config(path.parent, database_name=path.name)


def test_retries_locked_sqlite_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_create_cache = cache_module.create_cache
    call_count = 0

    def flaky_create_cache(config: CacheConfig) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            message = "database is locked"
            raise sqlite3.OperationalError(message)
        return real_create_cache(config)

    monkeypatch.setattr(cache_module, "create_cache", flaky_create_cache)

    cache_module.create_configured_cache(
        _cache_config(tmp_path / "cache.sqlite3"), "test"
    )

    assert call_count == 2


@pytest.mark.parametrize(
    "cache_type", [CacheType.Sqlite, CacheType.Json, CacheType.Memory]
)
async def test_supported_cache_backends_roundtrip(
    tmp_path: Path, cache_type: CacheType
) -> None:
    if cache_type == CacheType.Sqlite:
        config = _cache_config(tmp_path / "cache.sqlite3")
    elif cache_type == CacheType.Json:
        config = CacheConfig(
            type=cache_type,
            storage=StorageConfig(
                type=StorageType.File, base_dir=str(tmp_path / "json")
            ),
        )
    else:
        config = CacheConfig(type=cache_type, storage=None)

    store = CacheStore(config, "test")
    assert await store.put_many([("key", "value", {"backend": cache_type})]) == 1
    assert await store.get("key") == ("value", {"backend": cache_type})


async def test_noop_cache_does_not_persist(tmp_path: Path) -> None:
    store = CacheStore(CacheConfig(type=CacheType.Noop, storage=None), "test")

    await store.put_many([("key", "value", {})])

    assert await store.get("key") is None


def _write_cache_entry(cache_path: str, cache_key: str, grade: str) -> None:
    async def write() -> None:
        cache = ContentAddressedCache(_cache_config(cache_path))
        await cache.put(cache_key, grade, {"writer": cache_key})
        await cache.flush()

    asyncio.run(write())


def _try_claim(cache_path: str, owner_id: str) -> bool:
    store = CacheStore(_cache_config(cache_path), "lease-test")
    return store.try_acquire("shared", owner_id, ttl_seconds=30)


def _compute_once(cache_path: str, owner_id: str) -> bool:
    async def compute() -> bool:
        store = CacheStore(_cache_config(cache_path), "coalescing-test")
        while True:
            if await store.get("shared") is not None:
                return False
            if store.try_acquire("shared", owner_id, ttl_seconds=5):
                await asyncio.sleep(0.1)
                await store.publish(
                    "shared",
                    "result",
                    {"owner": owner_id},
                    owner_id=owner_id,
                    logical_key="logical",
                    config_fingerprint="config",
                )
                return True
            await asyncio.sleep(0.02)

    return asyncio.run(compute())


def _acquire_then_exit(cache_path: str) -> bool:
    store = CacheStore(_cache_config(cache_path), "interruption-test")
    return store.try_acquire("abandoned", "crashed", ttl_seconds=0.2)


class TestComputeCacheKey:
    """Tests for compute_cache_key."""

    def test_stable_across_calls(self) -> None:
        """The same inputs always produce the same key."""
        assert compute_cache_key("a", "b") == compute_cache_key("a", "b")

    def test_distinguishes_inputs(self) -> None:
        """Different assertion/chunk combinations produce different keys."""
        assert compute_cache_key("a", "b") != compute_cache_key("b", "a")
        assert compute_cache_key("a", "b") != compute_cache_key("a", "c")

    def test_distinguishes_model(self) -> None:
        """Different judge models produce different keys."""
        assert compute_cache_key("a", "b", model="gpt-4.1") != compute_cache_key(
            "a", "b", model="gpt-4o"
        )

    def test_distinguishes_call_args(self) -> None:
        """Different call arguments produce different keys."""
        assert compute_cache_key(
            "a", "b", call_args={"temperature": 0.0}
        ) != compute_cache_key("a", "b", call_args={"temperature": 1.0})

    def test_distinguishes_prompts(self) -> None:
        """Different prompt templates produce different keys."""
        assert compute_cache_key("a", "b", system_prompt="s1") != compute_cache_key(
            "a", "b", system_prompt="s2"
        )
        assert compute_cache_key("a", "b", user_prompt="u1") != compute_cache_key(
            "a", "b", user_prompt="u2"
        )

    def test_credentials_do_not_affect_key(self) -> None:
        """Credentials are neither persisted nor included in cache identity."""
        first = compute_cache_key("a", "b", call_args={"api_key": "first"})
        second = compute_cache_key("a", "b", call_args={"api_key": "second"})
        assert first == second


def test_build_cache_metadata_redacts_credentials() -> None:
    metadata = build_cache_metadata(
        model="test-model",
        call_args={
            "temperature": 0,
            "max_tokens": 100,
            "api_key": "secret",
            "headers": {"Authorization": "Bearer secret"},
        },
        system_prompt="system",
        user_prompt="user",
    )

    assert metadata["model"] == "test-model"
    assert metadata["call_args"] == {
        "temperature": 0,
        "max_tokens": 100,
        "api_key": "<redacted>",
        "headers": {"Authorization": "<redacted>"},
    }


class TestContentAddressedCache:
    """Tests for ContentAddressedCache persistence semantics."""

    async def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        """Stored grades are retrievable and missing keys return None."""
        cache = ContentAddressedCache(_cache_config(tmp_path / "cache.sqlite3"))
        await cache.put("k1", "full_support")
        assert await cache.get("k1") == "full_support"
        assert await cache.get("missing") is None

    async def test_put_is_idempotent_for_new_count(self, tmp_path: Path) -> None:
        """Re-putting an existing key does not increment the new-entry count."""
        cache = ContentAddressedCache(_cache_config(tmp_path / "cache.sqlite3"))
        await cache.put("k1", "full_support")
        await cache.put("k1", "no_support")
        assert cache.new_count == 1
        assert await cache.get("k1") == "full_support"

    async def test_flush_persists_and_reloads(self, tmp_path: Path) -> None:
        """Flushed entries survive a reload from disk."""
        cache_path = tmp_path / "cache.sqlite3"
        cache = ContentAddressedCache(_cache_config(cache_path))
        await cache.put("k1", "full_support")
        await cache.put("k2", "partial_support")
        await cache.flush()

        reloaded = ContentAddressedCache(_cache_config(cache_path))
        assert await reloaded.get("k1") == "full_support"
        assert await reloaded.get("k2") == "partial_support"

    async def test_repeated_flush_does_not_duplicate(self, tmp_path: Path) -> None:
        """Incremental flushes preserve one row per cache key."""
        cache_path = tmp_path / "cache.sqlite3"
        cache = ContentAddressedCache(_cache_config(cache_path))

        await cache.put("k1", "full_support")
        await cache.flush()
        await cache.put("k2", "no_support")
        await cache.flush()

        with sqlite3.connect(cache_path) as connection:
            row_count = connection.execute(
                "SELECT COUNT(*) FROM cache_entries"
            ).fetchone()
        assert row_count == (2,)

        reloaded = ContentAddressedCache(_cache_config(cache_path))
        assert await reloaded.get("k1") == "full_support"
        assert await reloaded.get("k2") == "no_support"

    async def test_flush_noop_when_no_new_entries(self, tmp_path: Path) -> None:
        """Flushing with no new entries leaves the database empty."""
        cache_path = tmp_path / "cache.sqlite3"
        cache = ContentAddressedCache(_cache_config(cache_path))
        assert await cache.flush() == 0
        with sqlite3.connect(cache_path) as connection:
            row_count = connection.execute(
                "SELECT COUNT(*) FROM cache_entries"
            ).fetchone()
        assert row_count == (0,)

    async def test_metadata_persists(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        metadata = build_cache_metadata(model="gpt-test", call_args={"temperature": 0})
        cache = ContentAddressedCache(_cache_config(cache_path))
        await cache.put("k1", "full_support", metadata)
        await cache.flush()

        assert (
            await ContentAddressedCache(_cache_config(cache_path)).get_metadata("k1")
            == metadata
        )

    async def test_imports_legacy_jsonl_cache(self, tmp_path: Path) -> None:
        legacy_path = tmp_path / "cache.jsonl"
        legacy_path.write_text(
            json.dumps({"key": "legacy", "grade": "partial_support"}) + "\n",
            encoding="utf-8",
        )

        cache = ContentAddressedCache(
            create_default_cache_config(
                tmp_path,
                database_name="cache.sqlite3",
            )
        )

        assert cache.cache_path == tmp_path / "cache.sqlite3"
        assert await cache.get("legacy") == "partial_support"
        assert await cache.get_metadata("legacy") == {
            "schema_version": 1,
            "source": "legacy_jsonl",
        }

    async def test_concurrent_processes_preserve_all_entries(
        self, tmp_path: Path
    ) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        entries = [(f"k{index}", f"grade-{index}") for index in range(12)]

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_write_cache_entry, str(cache_path), key, grade)
                for key, grade in entries
            ]
            for future in futures:
                future.result(timeout=30)

        cache = ContentAddressedCache(_cache_config(cache_path))
        assert {key: await cache.get(key) for key, _grade in entries} == dict(entries)

    async def test_concurrent_same_key_uses_first_writer(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        grades = [f"grade-{index}" for index in range(8)]

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_write_cache_entry, str(cache_path), "shared", grade)
                for grade in grades
            ]
            for future in futures:
                future.result(timeout=30)

        assert (
            await ContentAddressedCache(_cache_config(cache_path)).get("shared")
            in grades
        )
        with sqlite3.connect(cache_path) as connection:
            row_count = connection.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE key = 'shared'"
            ).fetchone()
        assert row_count == (1,)

    def test_only_one_process_acquires_same_work(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"

        with ProcessPoolExecutor(max_workers=4) as executor:
            acquired = list(
                executor.map(
                    _try_claim,
                    [str(cache_path)] * 8,
                    [f"owner-{index}" for index in range(8)],
                )
            )

        assert acquired.count(True) == 1
        assert inspect_cache(cache_path)["active_leases"] == 1

    async def test_concurrent_processes_compute_same_work_once(
        self, tmp_path: Path
    ) -> None:
        cache_path = tmp_path / "cache.sqlite3"

        with ProcessPoolExecutor(max_workers=4) as executor:
            computed = list(
                executor.map(
                    _compute_once,
                    [str(cache_path)] * 4,
                    [f"owner-{index}" for index in range(4)],
                )
            )

        assert computed.count(True) == 1
        store = CacheStore(_cache_config(cache_path), "coalescing-test")
        assert await store.get("shared") is not None
        assert inspect_cache(cache_path)["active_leases"] == 0

    def test_expired_lease_is_recovered_after_writer_interruption(
        self, tmp_path: Path
    ) -> None:
        store = CacheStore(_cache_config(tmp_path / "cache.sqlite3"), "lease-test")

        assert store.try_acquire("k1", "crashed", ttl_seconds=1, now=100)
        assert not store.try_acquire("k1", "waiting", ttl_seconds=1, now=100.5)
        assert store.try_acquire("k1", "recovered", ttl_seconds=1, now=101)

    def test_recovers_lease_abandoned_by_exited_process(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        with ProcessPoolExecutor(max_workers=1) as executor:
            assert executor.submit(_acquire_then_exit, str(cache_path)).result(
                timeout=30
            )

        time.sleep(0.25)
        store = CacheStore(_cache_config(cache_path), "interruption-test")
        assert store.try_acquire("abandoned", "recovered", ttl_seconds=1)

    async def test_publish_releases_lease_atomically(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        store = CacheStore(_cache_config(cache_path), "lease-test")
        assert store.try_acquire("k1", "owner", ttl_seconds=30)

        inserted = await store.publish(
            "k1",
            "value",
            {"model": "test"},
            owner_id="owner",
            logical_key="logical",
            config_fingerprint="config",
        )

        assert inserted
        assert await store.get("k1") == ("value", {"model": "test"})
        assert inspect_cache(cache_path)["active_leases"] == 0

    async def test_migrates_local_schema_to_graphrag_cache(
        self, tmp_path: Path
    ) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        with sqlite3.connect(cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE cache_entries (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            connection.execute(
                """
                INSERT INTO cache_entries(namespace, key, value_json, metadata_json)
                VALUES ('chunk_assertion', 'old', '"full_support"', '{}')
                """
            )

        store = CacheStore(_cache_config(cache_path), "chunk_assertion")

        assert await store.get("old") == ("full_support", {})
        with sqlite3.connect(cache_path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(cache_entries)")
            }
        assert columns == {"namespace", "key", "value_json"}

    async def test_reports_alternate_configuration(self, tmp_path: Path) -> None:
        cache = ContentAddressedCache(_cache_config(tmp_path / "cache.sqlite3"))
        assertion = "assertion"
        chunk = "chunk"
        first_metadata = build_cache_metadata(model="first")
        first_key = compute_cache_key(assertion, chunk, model="first")
        logical_key = compute_logical_key(assertion, chunk)
        first_fingerprint = compute_config_fingerprint(model="first")
        assert await cache.claim(first_key, "owner")
        await cache.publish(
            first_key,
            "full_support",
            first_metadata,
            owner_id="owner",
            logical_key=logical_key,
            config_fingerprint=first_fingerprint,
        )

        mismatch_count, fields = await cache.find_configuration_mismatches(
            [(logical_key, compute_config_fingerprint(model="second"))],
            build_cache_metadata(model="second"),
        )

        assert mismatch_count == 1
        assert fields == ["model"]
