# Copyright (c) 2025 Microsoft Corporation.
"""Content-addressed cache for (assertion, chunk) pairs."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log: logging.Logger = logging.getLogger(__name__)


class ContentAddressedCache:
    r"""Persistent SHA256-based cache for (assertion, chunk) -> grade.

    Key is computed as SHA256(assertion_text + '\x00' + chunk_content),
    ensuring stable cache hits across runs with identical assertions and chunks.
    """

    def __init__(self, cache_path: Path | str) -> None:
        """Initialize cache at given path.

        Args:
            cache_path: Path to JSONL cache file. Parent directory is created if needed.
        """
        self.cache_path: Path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] = {}
        self.new_count: int = 0
        self._load()

    def _load(self) -> None:
        """Load existing cache from disk."""
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open(encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
                        cache_key = record.get("key")
                        grade = record.get("grade")
                        if cache_key and grade:
                            self._data[cache_key] = grade
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            log.warning("Failed to load cache from %s: %s", self.cache_path, exc)

    def get(self, cache_key: str) -> str | None:
        """Retrieve grade for cache key.

        Args:
            cache_key: SHA256 hash of (assertion + chunk)

        Returns
        -------
            Grade string ('full_support', 'partial_support', 'no_support') or None
        """
        return self._data.get(cache_key)

    def put(self, cache_key: str, grade: str) -> None:
        """Store grade for cache key.

        Args:
            cache_key: SHA256 hash of (assertion + chunk)
            grade: Grade string ('full_support', 'partial_support', 'no_support')
        """
        if cache_key not in self._data:
            self._data[cache_key] = grade
            self.new_count += 1

    def flush(self) -> None:
        """Write cache contents to disk.

        Rewrites the full JSONL file atomically to avoid duplicating existing
        entries when flushing incremental updates.
        """
        if self.new_count == 0:
            return
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            f.writelines(
                json.dumps({"key": cache_key, "grade": grade}) + "\n"
                for cache_key, grade in self._data.items()
            )
        tmp_path.replace(self.cache_path)
        self.new_count = 0


def compute_cache_key(assertion_text: str, chunk_content: str) -> str:
    """Compute stable SHA256 cache key for (assertion, chunk) pair.

    Args:
        assertion_text: Assertion statement
        chunk_content: Chunk/passage text

    Returns
    -------
        SHA256 hexdigest
    """
    payload = f"{assertion_text}\x00{chunk_content}"
    return hashlib.sha256(payload.encode()).hexdigest()
