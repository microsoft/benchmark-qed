# Copyright (c) 2025 Microsoft Corporation.
"""Content-addressed cache for (assertion, chunk) pairs."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log: logging.Logger = logging.getLogger(__name__)


class ContentAddressedCache:
    r"""Persistent SHA256-based cache for (assertion, chunk) -> grade.

    Key is computed by :func:`compute_cache_key`, which hashes the assertion,
    chunk, judge model configuration, and prompt templates so that changes to
    the model or prompts invalidate stale entries while identical configurations
    reuse cached grades across runs.
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


def compute_cache_key(
    assertion_text: str,
    chunk_content: str,
    *,
    model: str = "",
    call_args: dict[str, Any] | None = None,
    system_prompt: str = "",
    user_prompt: str = "",
) -> str:
    """Compute stable SHA256 cache key for an (assertion, chunk) judgement.

    The judge model identifier, call arguments (for example temperature and
    seed), and the prompt templates are folded into the key so that changing
    the model or prompts invalidates stale cache entries instead of returning
    grades produced under a different configuration.

    Args:
        assertion_text: Assertion statement
        chunk_content: Chunk/passage text
        model: Judge model identifier (for example ``gpt-4.1``)
        call_args: Additional LLM call arguments (for example temperature, seed)
        system_prompt: System prompt template used for judging
        user_prompt: User prompt template used for judging

    Returns
    -------
        SHA256 hexdigest
    """
    payload = {
        "assertion": assertion_text,
        "chunk": chunk_content,
        "model": model,
        "call_args": call_args or {},
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    content_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(content_str.encode()).hexdigest()
