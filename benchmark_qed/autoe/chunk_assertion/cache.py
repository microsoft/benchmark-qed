# Copyright (c) 2025 Microsoft Corporation.
"""Content-addressed cache for (assertion, chunk) pairs."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from benchmark_qed.cache import (
    SQLiteCache,
    changed_configuration_fields,
    redact_sensitive_values,
    stable_fingerprint,
)

log: logging.Logger = logging.getLogger(__name__)

_CACHE_NAMESPACE = "chunk_assertion"
_CACHE_SCHEMA_VERSION = 1
_DEFAULT_LEASE_TTL_SECONDS = 120.0


def build_cache_metadata(
    *,
    model: str = "",
    call_args: dict[str, Any] | None = None,
    system_prompt: str = "",
    user_prompt: str = "",
) -> dict[str, Any]:
    """Build inspectable, credential-safe metadata for a cached judgement."""
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "evaluator": _CACHE_NAMESPACE,
        "model": model,
        "call_args": redact_sensitive_values(call_args or {}),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def compute_logical_key(assertion_text: str, chunk_content: str) -> str:
    """Fingerprint the inputs independently from judge configuration."""
    return stable_fingerprint({
        "assertion": assertion_text,
        "chunk": chunk_content,
    })


def compute_config_fingerprint(
    *,
    model: str = "",
    call_args: dict[str, Any] | None = None,
    system_prompt: str = "",
    user_prompt: str = "",
) -> str:
    """Fingerprint the judge configuration independently from inputs."""
    return stable_fingerprint(
        build_cache_metadata(
            model=model,
            call_args=call_args,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    )


class ContentAddressedCache:
    r"""Persistent SQLite cache for (assertion, chunk) -> grade.

    SQLite WAL mode allows readers and writers in separate processes to share
    the cache safely. Writes use first-writer-wins semantics for a cache key.
    Existing JSONL caches are imported once when the database is initialized.
    """

    def __init__(
        self,
        cache_path: Path | str,
        *,
        lease_ttl_seconds: float = _DEFAULT_LEASE_TTL_SECONDS,
    ) -> None:
        """Initialize the cache and import a legacy JSONL cache when present."""
        requested_path = Path(cache_path)
        self.legacy_cache_path: Path | None = None
        if requested_path.suffix.lower() == ".jsonl":
            self.legacy_cache_path = requested_path
            self.cache_path = requested_path.with_suffix(".sqlite3")
        else:
            self.cache_path = requested_path
            legacy_path = requested_path.with_suffix(".jsonl")
            if legacy_path.exists():
                self.legacy_cache_path = legacy_path

        self._store = SQLiteCache(self.cache_path, _CACHE_NAMESPACE)
        self.lease_ttl_seconds = lease_ttl_seconds
        self._pending: dict[str, tuple[str, dict[str, Any]]] = {}
        self.new_count: int = 0
        self._migrate_legacy_cache()

    def _migrate_legacy_cache(self) -> None:
        """Import an existing JSONL cache once without modifying the source."""
        if self.legacy_cache_path is None or not self.legacy_cache_path.exists():
            return

        migration_name = f"legacy_import:{self.legacy_cache_path.resolve()}"
        if self._store.get_property(migration_name) is not None:
            return

        records: list[tuple[str, str, dict[str, Any]]] = []
        try:
            with self.legacy_cache_path.open(encoding="utf-8") as legacy_file:
                for line in legacy_file:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cache_key = record.get("key")
                    grade = record.get("grade")
                    if cache_key and grade:
                        records.append((
                            str(cache_key),
                            str(grade),
                            {
                                "schema_version": _CACHE_SCHEMA_VERSION,
                                "source": "legacy_jsonl",
                            },
                        ))
        except OSError as exc:
            log.warning(
                "Failed to import legacy cache from %s: %s", self.legacy_cache_path, exc
            )
            return

        self._store.put_many(records)
        self._store.set_property_if_absent(migration_name, "complete")

    def get(self, cache_key: str) -> str | None:
        """Retrieve a grade for a cache key."""
        pending = self._pending.get(cache_key)
        if pending is not None:
            return pending[0]
        entry = self._store.get(cache_key)
        return str(entry[0]) if entry is not None else None

    def get_metadata(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieve the inspectable metadata stored with a cache entry."""
        pending = self._pending.get(cache_key)
        if pending is not None:
            return pending[1]
        entry = self._store.get(cache_key)
        return entry[1] if entry is not None else None

    def put(
        self,
        cache_key: str,
        grade: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stage a grade for insertion on the next flush."""
        if cache_key in self._pending or self.get(cache_key) is not None:
            return
        self._pending[cache_key] = (grade, metadata or {})
        self.new_count = len(self._pending)

    def claim(self, cache_key: str, owner_id: str) -> bool:
        """Claim an uncached judgement for this worker."""
        return self._store.try_acquire(
            cache_key,
            owner_id,
            ttl_seconds=self.lease_ttl_seconds,
        )

    def renew(self, cache_keys: list[str], owner_id: str) -> int:
        """Renew active judgement leases."""
        return self._store.renew_leases(
            cache_keys,
            owner_id,
            ttl_seconds=self.lease_ttl_seconds,
        )

    def release(self, cache_keys: list[str], owner_id: str) -> None:
        """Release judgement leases without publishing."""
        self._store.release_leases(cache_keys, owner_id)

    def publish(
        self,
        cache_key: str,
        grade: str,
        metadata: dict[str, Any],
        *,
        owner_id: str,
        logical_key: str,
        config_fingerprint: str,
    ) -> bool:
        """Publish a grade and release its lease atomically."""
        return self._store.publish(
            cache_key,
            grade,
            metadata,
            owner_id=owner_id,
            logical_key=logical_key,
            config_fingerprint=config_fingerprint,
        )

    def find_configuration_mismatches(
        self,
        identities: list[tuple[str, str]],
        current_metadata: dict[str, Any],
    ) -> tuple[int, list[str]]:
        """Return the mismatch count and changed configuration fields."""
        alternatives = self._store.find_alternate_configurations(identities)
        changed_fields = {
            field
            for metadata in alternatives
            for field in changed_configuration_fields(current_metadata, metadata)
        }
        return len(alternatives), sorted(changed_fields)

    def flush(self) -> int:
        """Atomically insert staged entries and return the number persisted."""
        inserted_count = self._store.put_many([
            (cache_key, grade, metadata)
            for cache_key, (grade, metadata) in self._pending.items()
        ])
        self._pending.clear()
        self.new_count = 0
        return inserted_count


def compute_cache_key(
    assertion_text: str,
    chunk_content: str,
    *,
    model: str = "",
    call_args: dict[str, Any] | None = None,
    system_prompt: str = "",
    user_prompt: str = "",
) -> str:
    """Compute a stable SHA256 key for an (assertion, chunk) judgement."""
    payload = {
        "assertion": assertion_text,
        "chunk": chunk_content,
        "model": model,
        "call_args": redact_sensitive_values(call_args or {}),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    content_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(content_str.encode()).hexdigest()
