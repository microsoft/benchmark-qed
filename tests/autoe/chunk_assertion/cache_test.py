# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the content-addressed (assertion, chunk) cache."""

import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark_qed.autoe.chunk_assertion.cache import (
    ContentAddressedCache,
    build_cache_metadata,
    compute_cache_key,
    compute_config_fingerprint,
    compute_logical_key,
)
from benchmark_qed.cache import SQLiteCache, inspect_cache


def _write_cache_entry(cache_path: str, cache_key: str, grade: str) -> None:
    cache = ContentAddressedCache(cache_path)
    cache.put(cache_key, grade, {"writer": cache_key})
    cache.flush()


def _try_claim(cache_path: str, owner_id: str) -> bool:
    store = SQLiteCache(cache_path, "lease-test")
    return store.try_acquire("shared", owner_id, ttl_seconds=30)


def _compute_once(cache_path: str, owner_id: str) -> bool:
    store = SQLiteCache(cache_path, "coalescing-test")
    while True:
        if store.get("shared") is not None:
            return False
        if store.try_acquire("shared", owner_id, ttl_seconds=5):
            time.sleep(0.1)
            store.publish(
                "shared",
                "result",
                {"owner": owner_id},
                owner_id=owner_id,
                logical_key="logical",
                config_fingerprint="config",
            )
            return True
        time.sleep(0.02)


def _acquire_then_exit(cache_path: str) -> bool:
    store = SQLiteCache(cache_path, "interruption-test")
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

    def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        """Stored grades are retrievable and missing keys return None."""
        cache = ContentAddressedCache(tmp_path / "cache.sqlite3")
        cache.put("k1", "full_support")
        assert cache.get("k1") == "full_support"
        assert cache.get("missing") is None

    def test_put_is_idempotent_for_new_count(self, tmp_path: Path) -> None:
        """Re-putting an existing key does not increment the new-entry count."""
        cache = ContentAddressedCache(tmp_path / "cache.sqlite3")
        cache.put("k1", "full_support")
        cache.put("k1", "no_support")
        assert cache.new_count == 1
        assert cache.get("k1") == "full_support"

    def test_flush_persists_and_reloads(self, tmp_path: Path) -> None:
        """Flushed entries survive a reload from disk."""
        cache_path = tmp_path / "cache.sqlite3"
        cache = ContentAddressedCache(cache_path)
        cache.put("k1", "full_support")
        cache.put("k2", "partial_support")
        cache.flush()

        reloaded = ContentAddressedCache(cache_path)
        assert reloaded.get("k1") == "full_support"
        assert reloaded.get("k2") == "partial_support"

    def test_repeated_flush_does_not_duplicate(self, tmp_path: Path) -> None:
        """Incremental flushes preserve one row per cache key."""
        cache_path = tmp_path / "cache.sqlite3"
        cache = ContentAddressedCache(cache_path)

        cache.put("k1", "full_support")
        cache.flush()
        cache.put("k2", "no_support")
        cache.flush()

        with sqlite3.connect(cache_path) as connection:
            row_count = connection.execute(
                "SELECT COUNT(*) FROM cache_entries"
            ).fetchone()
        assert row_count == (2,)

        reloaded = ContentAddressedCache(cache_path)
        assert reloaded.get("k1") == "full_support"
        assert reloaded.get("k2") == "no_support"

    def test_flush_noop_when_no_new_entries(self, tmp_path: Path) -> None:
        """Flushing with no new entries leaves the database empty."""
        cache_path = tmp_path / "cache.sqlite3"
        cache = ContentAddressedCache(cache_path)
        assert cache.flush() == 0
        with sqlite3.connect(cache_path) as connection:
            row_count = connection.execute(
                "SELECT COUNT(*) FROM cache_entries"
            ).fetchone()
        assert row_count == (0,)

    def test_metadata_persists(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        metadata = build_cache_metadata(model="gpt-test", call_args={"temperature": 0})
        cache = ContentAddressedCache(cache_path)
        cache.put("k1", "full_support", metadata)
        cache.flush()

        assert ContentAddressedCache(cache_path).get_metadata("k1") == metadata

    def test_imports_legacy_jsonl_cache(self, tmp_path: Path) -> None:
        legacy_path = tmp_path / "cache.jsonl"
        legacy_path.write_text(
            json.dumps({"key": "legacy", "grade": "partial_support"}) + "\n",
            encoding="utf-8",
        )

        cache = ContentAddressedCache(legacy_path)

        assert cache.cache_path == tmp_path / "cache.sqlite3"
        assert cache.get("legacy") == "partial_support"
        assert cache.get_metadata("legacy") == {
            "schema_version": 1,
            "source": "legacy_jsonl",
        }

    def test_concurrent_processes_preserve_all_entries(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        entries = [(f"k{index}", f"grade-{index}") for index in range(12)]

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_write_cache_entry, str(cache_path), key, grade)
                for key, grade in entries
            ]
            for future in futures:
                future.result(timeout=30)

        cache = ContentAddressedCache(cache_path)
        assert {key: cache.get(key) for key, _grade in entries} == dict(entries)

    def test_concurrent_same_key_uses_first_writer(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        grades = [f"grade-{index}" for index in range(8)]

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_write_cache_entry, str(cache_path), "shared", grade)
                for grade in grades
            ]
            for future in futures:
                future.result(timeout=30)

        assert ContentAddressedCache(cache_path).get("shared") in grades
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

    def test_concurrent_processes_compute_same_work_once(self, tmp_path: Path) -> None:
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
        store = SQLiteCache(cache_path, "coalescing-test")
        assert store.get("shared") is not None
        assert inspect_cache(cache_path)["active_leases"] == 0

    def test_expired_lease_is_recovered_after_writer_interruption(
        self, tmp_path: Path
    ) -> None:
        store = SQLiteCache(tmp_path / "cache.sqlite3", "lease-test")

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
        store = SQLiteCache(cache_path, "interruption-test")
        assert store.try_acquire("abandoned", "recovered", ttl_seconds=1)

    def test_publish_releases_lease_atomically(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        store = SQLiteCache(cache_path, "lease-test")
        assert store.try_acquire("k1", "owner", ttl_seconds=30)

        inserted = store.publish(
            "k1",
            "value",
            {"model": "test"},
            owner_id="owner",
            logical_key="logical",
            config_fingerprint="config",
        )

        assert inserted
        assert store.get("k1") == ("value", {"model": "test"})
        assert inspect_cache(cache_path)["active_leases"] == 0

    def test_migrates_v1_schema(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.sqlite3"
        with sqlite3.connect(cache_path) as connection:
            connection.execute(
                "CREATE TABLE cache_properties (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO cache_properties VALUES ('schema_version', '1')"
            )
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

        store = SQLiteCache(cache_path, "chunk_assertion")

        assert store.get("old") == ("full_support", {})
        assert inspect_cache(cache_path)["schema_version"] == 2
        with sqlite3.connect(cache_path) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(cache_entries)")
            }
        assert {"logical_key", "config_fingerprint"} <= columns

    def test_reports_alternate_configuration(self, tmp_path: Path) -> None:
        cache = ContentAddressedCache(tmp_path / "cache.sqlite3")
        assertion = "assertion"
        chunk = "chunk"
        first_metadata = build_cache_metadata(model="first")
        first_key = compute_cache_key(assertion, chunk, model="first")
        logical_key = compute_logical_key(assertion, chunk)
        first_fingerprint = compute_config_fingerprint(model="first")
        assert cache.claim(first_key, "owner")
        cache.publish(
            first_key,
            "full_support",
            first_metadata,
            owner_id="owner",
            logical_key=logical_key,
            config_fingerprint=first_fingerprint,
        )

        mismatch_count, fields = cache.find_configuration_mismatches(
            [(logical_key, compute_config_fingerprint(model="second"))],
            build_cache_metadata(model="second"),
        )

        assert mismatch_count == 1
        assert fields == ["model"]
