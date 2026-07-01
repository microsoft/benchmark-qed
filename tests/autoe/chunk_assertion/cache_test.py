# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the content-addressed (assertion, chunk) cache."""

from pathlib import Path

from benchmark_qed.autoe.chunk_assertion.cache import (
    ContentAddressedCache,
    compute_cache_key,
)


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


class TestContentAddressedCache:
    """Tests for ContentAddressedCache persistence semantics."""

    def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        """Stored grades are retrievable and missing keys return None."""
        cache = ContentAddressedCache(tmp_path / "cache.jsonl")
        cache.put("k1", "full_support")
        assert cache.get("k1") == "full_support"
        assert cache.get("missing") is None

    def test_put_is_idempotent_for_new_count(self, tmp_path: Path) -> None:
        """Re-putting an existing key does not increment the new-entry count."""
        cache = ContentAddressedCache(tmp_path / "cache.jsonl")
        cache.put("k1", "full_support")
        cache.put("k1", "no_support")
        assert cache.new_count == 1
        assert cache.get("k1") == "full_support"

    def test_flush_persists_and_reloads(self, tmp_path: Path) -> None:
        """Flushed entries survive a reload from disk."""
        cache_path = tmp_path / "cache.jsonl"
        cache = ContentAddressedCache(cache_path)
        cache.put("k1", "full_support")
        cache.put("k2", "partial_support")
        cache.flush()

        reloaded = ContentAddressedCache(cache_path)
        assert reloaded.get("k1") == "full_support"
        assert reloaded.get("k2") == "partial_support"

    def test_repeated_flush_does_not_duplicate(self, tmp_path: Path) -> None:
        """Incremental flushes rewrite the file without duplicating entries."""
        cache_path = tmp_path / "cache.jsonl"
        cache = ContentAddressedCache(cache_path)

        cache.put("k1", "full_support")
        cache.flush()
        cache.put("k2", "no_support")
        cache.flush()

        line_count = sum(
            1 for line in cache_path.read_text(encoding="utf-8").splitlines() if line
        )
        assert line_count == 2

        reloaded = ContentAddressedCache(cache_path)
        assert reloaded.get("k1") == "full_support"
        assert reloaded.get("k2") == "no_support"

    def test_flush_noop_when_no_new_entries(self, tmp_path: Path) -> None:
        """Flushing with no new entries leaves the file untouched."""
        cache_path = tmp_path / "cache.jsonl"
        cache = ContentAddressedCache(cache_path)
        cache.flush()
        assert not cache_path.exists()
