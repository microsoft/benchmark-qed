# Copyright (c) 2025 Microsoft Corporation.
"""Content-addressed cache for (assertion, chunk) pairs."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from benchmark_qed.cache import (
    CacheStore,
    changed_configuration_fields,
    get_cache_base_dir,
    get_sqlite_cache_path,
    redact_sensitive_values,
    stable_fingerprint,
)

log: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from graphrag_cache import CacheConfig

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
    r"""Persistent cache for (assertion, chunk) -> grade.

    The graphrag-cache SQLite backend allows readers and writers in separate
    processes to share the cache. Existing JSONL caches are imported once.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        *,
        lease_ttl_seconds: float = _DEFAULT_LEASE_TTL_SECONDS,
    ) -> None:
        """Initialize the cache and import a legacy JSONL cache when present."""
        self.cache_config = cache_config
        base_dir = get_cache_base_dir(cache_config)
        self.cache_path = get_sqlite_cache_path(cache_config)
        self.legacy_cache_path: Path | None = None
        if self.cache_path is not None:
            legacy_path = self.cache_path.with_suffix(".jsonl")
        elif base_dir is not None:
            legacy_path = base_dir / "chunk_assertions.jsonl"
        else:
            legacy_path = None
        if (
            legacy_path is not None
            and legacy_path != self.cache_path
            and legacy_path.exists()
        ):
            self.legacy_cache_path = legacy_path

        self._store = CacheStore(
            cache_config,
            _CACHE_NAMESPACE,
            lease_ttl_seconds=lease_ttl_seconds,
        )
        self.lease_ttl_seconds = lease_ttl_seconds
        self._pending: dict[str, tuple[str, dict[str, Any]]] = {}
        self.new_count: int = 0
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        await self._migrate_legacy_cache()
        self._initialized = True

    async def _migrate_legacy_cache(self) -> None:
        """Import an existing JSONL cache once without modifying the source."""
        if self.legacy_cache_path is None or not self.legacy_cache_path.exists():
            return

        migration_name = f"legacy_import:{self.legacy_cache_path.resolve()}"
        if await self._store.get_property(migration_name) is not None:
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
        except (OSError, UnicodeError) as exc:
            log.warning(
                "Failed to import legacy cache from %s: %s", self.legacy_cache_path, exc
            )
            return

        await self._store.put_many(records)
        await self._store.set_property_if_absent(migration_name, "complete")

    async def get(self, cache_key: str) -> str | None:
        """Retrieve a grade for a cache key."""
        await self._ensure_initialized()
        pending = self._pending.get(cache_key)
        if pending is not None:
            return pending[0]
        entry = await self._store.get(cache_key)
        return str(entry[0]) if entry is not None else None

    async def get_metadata(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieve the inspectable metadata stored with a cache entry."""
        await self._ensure_initialized()
        pending = self._pending.get(cache_key)
        if pending is not None:
            return pending[1]
        entry = await self._store.get(cache_key)
        return entry[1] if entry is not None else None

    async def put(
        self,
        cache_key: str,
        grade: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stage a grade for insertion on the next flush."""
        await self._ensure_initialized()
        if cache_key in self._pending or await self.get(cache_key) is not None:
            return
        self._pending[cache_key] = (grade, metadata or {})
        self.new_count = len(self._pending)

    async def claim(self, cache_key: str, owner_id: str) -> bool:
        """Claim an uncached judgement for this worker."""
        await self._ensure_initialized()
        if await self._store.get(cache_key) is not None:
            return False
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

    async def publish(
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
        await self._ensure_initialized()
        return await self._store.publish(
            cache_key,
            grade,
            metadata,
            owner_id=owner_id,
            logical_key=logical_key,
            config_fingerprint=config_fingerprint,
        )

    async def find_configuration_mismatches(
        self,
        identities: list[tuple[str, str]],
        current_metadata: dict[str, Any],
    ) -> tuple[int, list[str]]:
        """Return the mismatch count and changed configuration fields."""
        await self._ensure_initialized()
        alternatives = await self._store.find_alternate_configurations(identities)
        changed_fields = {
            field
            for metadata in alternatives
            for field in changed_configuration_fields(current_metadata, metadata)
        }
        return len(alternatives), sorted(changed_fields)

    async def flush(self) -> int:
        """Atomically insert staged entries and return the number persisted."""
        await self._ensure_initialized()
        inserted_count = await self._store.put_many([
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
