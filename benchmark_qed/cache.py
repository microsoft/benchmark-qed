# Copyright (c) 2025 Microsoft Corporation.
"""Shared cache helpers backed by graphrag-cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from filelock import FileLock
from graphrag_cache import Cache, CacheConfig, CacheType, create_cache
from graphrag_storage import StorageConfig, StorageType

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "connection_string",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}
_SENSITIVE_KEY_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_auth_token",
    "_connection_string",
    "_credential",
    "_password",
    "_secret",
)
_SQLITE_INITIALIZATION_TIMEOUT_SECONDS = 30.0
_SQLITE_INITIALIZATION_RETRY_SECONDS = 0.01


@dataclass(frozen=True)
class CachePublishResult:
    """Result of publishing a cache entry under a lease."""

    accepted: bool
    inserted: bool
    value: Any | None
    metadata: dict[str, Any] | None


def _is_sensitive_key(key: object) -> bool:
    """Return whether a configuration key conventionally contains a secret."""
    normalized = str(key).lower()
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES)


def redact_sensitive_values(value: Any) -> Any:
    """Remove credentials from configuration persisted in local caches."""
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if _is_sensitive_key(key)
                else redact_sensitive_values(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_values(item) for item in value]
    return value


def stable_fingerprint(value: Any) -> str:
    """Return a deterministic SHA256 fingerprint for JSON-like data."""
    content = json.dumps(
        redact_sensitive_values(value), sort_keys=True, default=str
    ).encode()
    return hashlib.sha256(content).hexdigest()


def changed_configuration_fields(
    current: dict[str, Any], cached: dict[str, Any]
) -> list[str]:
    """Return top-level configuration fields whose values differ."""
    ignored = {"created_at", "schema_version", "source", "timestamp"}
    return sorted(
        key
        for key in current.keys() | cached.keys()
        if key not in ignored and current.get(key) != cached.get(key)
    )


def create_default_cache_config(
    base_dir: Path | str,
    *,
    database_name: str,
) -> CacheConfig:
    """Create the default persistent SQLite cache configuration."""
    return CacheConfig(
        type=CacheType.Sqlite,
        storage=StorageConfig(
            type=StorageType.File,
            base_dir=str(base_dir),
        ),
        database_name=database_name,
    )


def get_cache_base_dir(config: CacheConfig) -> Path | None:
    """Return the configured local cache directory, when one exists."""
    storage = config.storage
    if (
        config.type not in {CacheType.Json, CacheType.Sqlite}
        or storage is None
        or storage.type != StorageType.File
        or storage.base_dir is None
    ):
        return None
    return Path(storage.base_dir)


def get_sqlite_cache_path(config: CacheConfig) -> Path | None:
    """Return the configured SQLite database path, when applicable."""
    base_dir = get_cache_base_dir(config)
    if config.type != CacheType.Sqlite or base_dir is None:
        return None
    return base_dir / config.database_name


def create_configured_cache(config: CacheConfig, namespace: str) -> Cache:
    """Create a namespaced cache from a GraphRAG cache configuration."""
    database_path = get_sqlite_cache_path(config)
    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        _migrate_local_cache_schema(database_path)
        deadline = time.monotonic() + _SQLITE_INITIALIZATION_TIMEOUT_SECONDS
        retry_delay = _SQLITE_INITIALIZATION_RETRY_SECONDS
        while True:
            try:
                return create_cache(config).child(namespace)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 0.25)
    return create_cache(config).child(namespace)


def _migrate_local_cache_schema(path: Path) -> None:
    """Convert the retired benchmark-qed schema to graphrag-cache in place."""
    if not path.exists():
        return
    connection = sqlite3.connect(path, timeout=30)
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(cache_entries)")
        }
        if not columns or columns == {"namespace", "key", "value_json"}:
            return
        required_old_columns = {
            "namespace",
            "key",
            "value_json",
            "metadata_json",
        }
        if not required_old_columns <= columns:
            msg = f"Unsupported cache schema in {path}"
            raise RuntimeError(msg)

        logical_key_column = (
            "logical_key" if "logical_key" in columns else "NULL AS logical_key"
        )
        config_fingerprint_column = (
            "config_fingerprint"
            if "config_fingerprint" in columns
            else "NULL AS config_fingerprint"
        )
        created_at_column = (
            "created_at" if "created_at" in columns else "NULL AS created_at"
        )
        rows = connection.execute(
            f"""
            SELECT namespace, key, value_json, metadata_json,
                   {logical_key_column}, {config_fingerprint_column},
                   {created_at_column}
            FROM cache_entries
            """  # ruff: ignore[hardcoded-sql-expression] -- fixed expressions only
        ).fetchall()
        with connection:
            connection.execute("DROP TABLE IF EXISTS cache_entries_graphrag")
            connection.execute(
                """
                CREATE TABLE cache_entries_graphrag (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            for (
                namespace,
                key,
                value_json,
                metadata_json,
                logical_key,
                config_fingerprint,
                created_at,
            ) in rows:
                metadata = json.loads(metadata_json)
                envelope = {
                    "result": {
                        "value": json.loads(value_json),
                        "metadata": metadata,
                        "logical_key": logical_key,
                        "config_fingerprint": config_fingerprint,
                        "created_at": created_at,
                    }
                }
                connection.execute(
                    """
                    INSERT INTO cache_entries_graphrag(namespace, key, value_json)
                    VALUES (?, ?, ?)
                    """,
                    (namespace, key, json.dumps(envelope, ensure_ascii=False)),
                )
                if logical_key is not None and config_fingerprint is not None:
                    configuration = {
                        "result": {
                            "config_fingerprint": config_fingerprint,
                            "metadata": metadata,
                        }
                    }
                    connection.execute(
                        """
                        INSERT INTO cache_entries_graphrag(
                            namespace, key, value_json
                        )
                        VALUES (?, ?, ?)
                        ON CONFLICT(namespace, key)
                        DO UPDATE SET value_json = excluded.value_json
                        """,
                        (
                            f"{namespace}/configurations",
                            logical_key,
                            json.dumps(configuration, ensure_ascii=False),
                        ),
                    )
            connection.execute("DROP TABLE cache_entries")
            connection.execute(
                "ALTER TABLE cache_entries_graphrag RENAME TO cache_entries"
            )
            connection.execute("DROP TABLE IF EXISTS cache_properties")
            connection.execute("DROP TABLE IF EXISTS cache_leases")
    finally:
        connection.close()


class CacheStore:
    """Application cache metadata layered over graphrag-cache."""

    _memory_lease_lock = threading.Lock()
    _memory_leases: ClassVar[dict[tuple[str, str], tuple[str, float]]] = {}

    def __init__(
        self,
        config: CacheConfig,
        namespace: str,
        *,
        lease_ttl_seconds: float = 120.0,
    ) -> None:
        """Initialize cache namespaces and process-coordination files."""
        self.config = config
        self.namespace = namespace
        root_cache = create_configured_cache(config, namespace)
        self._cache = root_cache
        self._properties = create_configured_cache(config, f"{namespace}/properties")
        self._configurations = create_configured_cache(
            config, f"{namespace}/configurations"
        )
        self._lease_ttl_seconds = lease_ttl_seconds
        self.database_path: Path | None = get_sqlite_cache_path(config)
        self.base_dir: Path | None = get_cache_base_dir(config)
        namespace_hash = hashlib.sha256(namespace.encode()).hexdigest()[:16]
        config_hash = hashlib.sha256(config.model_dump_json().encode()).hexdigest()[:16]
        self._lease_dir = (
            self.base_dir / ".benchmark_qed_cache_leases" / config_hash / namespace_hash
            if self.base_dir is not None
            else None
        )
        self._memory_lease_prefix = (config_hash, namespace)
        self._known_keys: set[str] = set()

    async def get(self, key: str) -> tuple[Any, dict[str, Any]] | None:
        """Return a decoded value and metadata for a key."""
        entry = await self._cache.get(key)
        if not isinstance(entry, dict) or "value" not in entry:
            return None
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            return None
        self._known_keys.add(key)
        return entry["value"], metadata

    async def get_many(self, keys: list[str]) -> dict[str, tuple[Any, dict[str, Any]]]:
        """Return decoded values and metadata for existing keys."""
        entries = await self._cache.get_many(keys)
        results: dict[str, tuple[Any, dict[str, Any]]] = {}
        for key, entry in entries.items():
            if not isinstance(entry, dict) or "value" not in entry:
                continue
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                continue
            results[key] = (entry["value"], metadata)
        self._known_keys.update(results)
        return results

    async def put_many(
        self,
        entries: list[tuple[str, Any, dict[str, Any]]],
        *,
        identities: dict[str, tuple[str, str]] | None = None,
    ) -> int:
        """Insert entries with first-observed-writer semantics."""
        if self.config.type == CacheType.Noop:
            return 0
        inserted = 0
        for key, value, metadata in entries:
            if self._lease_dir is None:
                with self._memory_lease_lock:
                    was_inserted, _ = await self._insert_if_absent(
                        key, value, metadata, identities
                    )
            else:
                was_inserted, _ = await asyncio.to_thread(
                    self._insert_file_entry_if_absent,
                    key,
                    value,
                    metadata,
                    identities,
                )
            inserted += int(was_inserted)
        return inserted

    def _insert_file_entry_if_absent(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any],
        identities: dict[str, tuple[str, str]] | None,
    ) -> tuple[bool, tuple[Any, dict[str, Any]]]:
        with FileLock(self._lease_mutex_path(key)):
            return asyncio.run(self._insert_if_absent(key, value, metadata, identities))

    def _publish_file_entry(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any],
        owner_id: str,
        logical_key: str,
        config_fingerprint: str,
    ) -> CachePublishResult:
        with FileLock(self._lease_mutex_path(key)):
            lease_path = self._lease_path(key)
            try:
                lease = json.loads(lease_path.read_text(encoding="utf-8"))
                owns_lease = (
                    lease.get("owner_id") == owner_id
                    and float(lease["expires_at"]) > time.time()
                )
            except (
                AttributeError,
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
            ):
                owns_lease = False
            if not owns_lease:
                canonical = asyncio.run(self.get(key))
                return self._rejected_publication(canonical)
            inserted, canonical = asyncio.run(
                self._insert_if_absent(
                    key,
                    value,
                    metadata,
                    {key: (logical_key, config_fingerprint)},
                )
            )
            lease_path.unlink(missing_ok=True)
            return CachePublishResult(
                accepted=True,
                inserted=inserted,
                value=canonical[0],
                metadata=canonical[1],
            )

    async def publish(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any],
        *,
        owner_id: str,
        logical_key: str,
        config_fingerprint: str,
    ) -> CachePublishResult:
        """Publish a claimed result, its configuration, and release the lease."""
        if self._lease_dir is None:
            with self._memory_lease_lock:
                lease = self._memory_leases.get(self._memory_lease_key(key))
                if lease is None or lease[0] != owner_id or lease[1] <= time.time():
                    canonical = await self.get(key)
                    return self._rejected_publication(canonical)
                inserted, canonical = await self._insert_if_absent(
                    key,
                    value,
                    metadata,
                    {key: (logical_key, config_fingerprint)},
                )
                self._memory_leases.pop(self._memory_lease_key(key), None)
        else:
            return await asyncio.to_thread(
                self._publish_file_entry,
                key,
                value,
                metadata,
                owner_id,
                logical_key,
                config_fingerprint,
            )

        return CachePublishResult(
            accepted=True,
            inserted=inserted,
            value=canonical[0],
            metadata=canonical[1],
        )

    async def _insert_if_absent(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any],
        identities: dict[str, tuple[str, str]] | None,
    ) -> tuple[bool, tuple[Any, dict[str, Any]]]:
        canonical = await self.get(key)
        if canonical is not None:
            return False, canonical
        logical_key, config_fingerprint = (identities or {}).get(key, (None, None))
        await self._cache.set(
            key,
            {
                "value": value,
                "metadata": metadata,
                "logical_key": logical_key,
                "config_fingerprint": config_fingerprint,
                "created_at": datetime.now(tz=UTC).isoformat(),
            },
        )
        self._known_keys.add(key)
        if logical_key is not None and config_fingerprint is not None:
            await self._configurations.set(
                logical_key,
                {
                    "config_fingerprint": config_fingerprint,
                    "metadata": metadata,
                },
            )
        return True, (value, metadata)

    @staticmethod
    def _rejected_publication(
        canonical: tuple[Any, dict[str, Any]] | None,
    ) -> CachePublishResult:
        return CachePublishResult(
            accepted=False,
            inserted=False,
            value=canonical[0] if canonical is not None else None,
            metadata=canonical[1] if canonical is not None else None,
        )

    async def claim(
        self,
        key: str,
        owner_id: str,
        *,
        ttl_seconds: float | None = None,
    ) -> bool:
        """Claim uncached work and close the check-after-publication race."""
        if await self.get(key) is not None:
            return False
        if not self.try_acquire(key, owner_id, ttl_seconds=ttl_seconds):
            return False
        if await self.get(key) is None:
            return True
        self.release_leases([key], owner_id)
        return False

    def try_acquire(
        self,
        key: str,
        owner_id: str,
        *,
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> bool:
        """Claim work using an expiring local or in-memory lease."""
        current_time = time.time() if now is None else now
        lease_ttl = self._lease_ttl_seconds if ttl_seconds is None else ttl_seconds
        if self._lease_dir is None:
            lease_key = self._memory_lease_key(key)
            with self._memory_lease_lock:
                lease = self._memory_leases.get(lease_key)
                if lease is not None and lease[1] > current_time:
                    return False
                self._memory_leases[lease_key] = (
                    owner_id,
                    current_time + lease_ttl,
                )
            return True

        lease_path = self._lease_path(key)
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "owner_id": owner_id,
            "expires_at": current_time + lease_ttl,
        }

        with FileLock(self._lease_mutex_path(key)):
            try:
                lease = json.loads(lease_path.read_text(encoding="utf-8"))
                if float(lease["expires_at"]) > current_time:
                    return False
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                pass
            self._write_lease(lease_path, payload)
            return True

    def renew_leases(
        self, keys: list[str], owner_id: str, *, ttl_seconds: float | None = None
    ) -> int:
        """Extend leases still owned by a worker."""
        lease_ttl = self._lease_ttl_seconds if ttl_seconds is None else ttl_seconds
        if self._lease_dir is None:
            renewed = 0
            with self._memory_lease_lock:
                for key in keys:
                    lease_key = self._memory_lease_key(key)
                    lease = self._memory_leases.get(lease_key)
                    if lease is not None and lease[0] == owner_id:
                        self._memory_leases[lease_key] = (
                            owner_id,
                            time.time() + lease_ttl,
                        )
                        renewed += 1
            return renewed

        renewed = 0
        for key in keys:
            lease_path = self._lease_path(key)
            with FileLock(self._lease_mutex_path(key)):
                try:
                    lease = json.loads(lease_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if lease.get("owner_id") != owner_id:
                    continue
                lease["expires_at"] = time.time() + lease_ttl
                self._write_lease(lease_path, lease)
                renewed += 1
        return renewed

    def release_leases(self, keys: list[str], owner_id: str) -> None:
        """Release leases owned by a worker without publishing results."""
        if self._lease_dir is None:
            with self._memory_lease_lock:
                for key in keys:
                    lease_key = self._memory_lease_key(key)
                    lease = self._memory_leases.get(lease_key)
                    if lease is not None and lease[0] == owner_id:
                        self._memory_leases.pop(lease_key)
            return

        for key in keys:
            lease_path = self._lease_path(key)
            with FileLock(self._lease_mutex_path(key)):
                try:
                    lease = json.loads(lease_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if lease.get("owner_id") == owner_id:
                    lease_path.unlink(missing_ok=True)

    async def find_alternate_configurations(
        self, requests: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """Return metadata for matching inputs produced by another configuration."""
        alternatives: list[dict[str, Any]] = []
        for logical_key, config_fingerprint in dict(requests).items():
            cached = await self._configurations.get(logical_key)
            if (
                isinstance(cached, dict)
                and cached.get("config_fingerprint") != config_fingerprint
                and isinstance(cached.get("metadata"), dict)
            ):
                alternatives.append(cached["metadata"])
        return alternatives

    async def get_property(self, name: str) -> str | None:
        """Return a cache property."""
        value = await self._properties.get(name)
        return str(value) if value is not None else None

    async def set_property_if_absent(self, name: str, value: str) -> None:
        """Set a cache property without replacing an existing value."""
        if await self._properties.get(name) is None:
            await self._properties.set(name, value)

    def count(self) -> int:
        """Return the number of entries observed by this store instance."""
        return len(self._known_keys)

    async def clear(self) -> None:
        """Delete entries and supporting metadata in this namespace."""
        await self._cache.clear()
        await self._properties.clear()
        await self._configurations.clear()
        if self.config.type == CacheType.Json and self.base_dir is not None:
            for namespace in (
                self.namespace,
                f"{self.namespace}/properties",
                f"{self.namespace}/configurations",
            ):
                (self.base_dir / namespace).mkdir(parents=True, exist_ok=True)
        self._known_keys.clear()
        if self._lease_dir is not None and self._lease_dir.exists():
            for lease_path in self._lease_dir.glob("*.lock"):
                lease_path.unlink()
        elif self._lease_dir is None:
            with self._memory_lease_lock:
                for lease_key in list(self._memory_leases):
                    if lease_key[0] == ":".join(self._memory_lease_prefix):
                        self._memory_leases.pop(lease_key)

    def size_bytes(self) -> int:
        """Return the size of configured local cache files."""
        if self.database_path is not None:
            return sum(
                path.stat().st_size
                for path in (
                    self.database_path,
                    Path(f"{self.database_path}-wal"),
                    Path(f"{self.database_path}-shm"),
                )
                if path.exists()
            )
        if self.base_dir is None or not self.base_dir.exists():
            return 0
        return sum(
            path.stat().st_size for path in self.base_dir.rglob("*") if path.is_file()
        )

    def _lease_path(self, key: str) -> Path:
        if self._lease_dir is None:
            msg = "File lease path requested for a non-file cache"
            raise RuntimeError(msg)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._lease_dir / f"{key_hash}.lock"

    def _lease_mutex_path(self, key: str) -> str:
        return f"{self._lease_path(key)}.mutex"

    @staticmethod
    def _write_lease(lease_path: Path, lease: dict[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=lease_path.parent,
            prefix=f".{lease_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as lease_file:
                json.dump(lease, lease_file)
            temporary_path.replace(lease_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _memory_lease_key(self, key: str) -> tuple[str, str]:
        return (":".join(self._memory_lease_prefix), key)


def inspect_cache(database_path: Path | str) -> dict[str, Any]:
    """Inspect the graphrag-cache SQLite file without mutating it."""
    path = Path(database_path)
    if not path.exists():
        msg = f"Cache database does not exist: {path}"
        raise FileNotFoundError(msg)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(cache_entries)")
        }
        if columns != {"namespace", "key", "value_json"}:
            msg = f"Not a graphrag-cache SQLite database: {path}"
            raise ValueError(msg)
        namespaces = [
            {"namespace": namespace, "entries": entry_count}
            for namespace, entry_count in connection.execute(
                """
                SELECT namespace, COUNT(*)
                FROM cache_entries
                GROUP BY namespace
                ORDER BY namespace
                """
            )
        ]
    finally:
        connection.close()
    lease_root = path.parent / ".benchmark_qed_cache_leases"
    active_leases = 0
    if lease_root.exists():
        now = time.time()
        for lease_path in lease_root.glob("*/*/*.lock"):
            try:
                lease = json.loads(lease_path.read_text(encoding="utf-8"))
                active_leases += int(float(lease["expires_at"]) > now)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return {
        "path": str(path),
        "backend": "graphrag-cache",
        "namespaces": namespaces,
        "active_leases": active_leases,
    }
