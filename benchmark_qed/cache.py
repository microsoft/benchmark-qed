# Copyright (c) 2025 Microsoft Corporation.
"""Shared concurrency-safe local cache storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

_CACHE_SCHEMA_VERSION = 2
_INITIALIZATION_TIMEOUT_SECONDS = 30
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


class SQLiteCache:
    """Namespaced JSON cache backed by SQLite in WAL mode."""

    def __init__(self, database_path: Path | str, namespace: str) -> None:
        """Initialize a cache namespace in a shared SQLite database."""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a configured connection for one short cache operation."""
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        """Initialize the schema, retrying only first-start lock contention."""
        deadline = time.monotonic() + _INITIALIZATION_TIMEOUT_SECONDS
        while True:
            try:
                self._initialize_once()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
            else:
                return

    def _initialize_once(self) -> None:
        """Create or migrate the cache schema."""
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_properties (
                    name TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            version_row = connection.execute(
                "SELECT value FROM cache_properties WHERE name = 'schema_version'"
            ).fetchone()
            if version_row is None:
                self._create_latest_schema(connection)
                connection.execute(
                    "INSERT INTO cache_properties(name, value) VALUES (?, ?)",
                    ("schema_version", str(_CACHE_SCHEMA_VERSION)),
                )
                return

            version = int(version_row[0])
            if version > _CACHE_SCHEMA_VERSION:
                msg = (
                    f"Unsupported cache schema version {version} "
                    f"in {self.database_path}"
                )
                raise RuntimeError(msg)
            if version == 1:
                self._migrate_v1_to_v2(connection)
                version = 2
            if version != _CACHE_SCHEMA_VERSION:
                msg = f"No migration available for cache schema version {version}"
                raise RuntimeError(msg)
            self._create_latest_schema(connection)

    @staticmethod
    def _create_latest_schema(connection: sqlite3.Connection) -> None:
        """Create all tables and indexes for the latest schema."""
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                logical_key TEXT,
                config_fingerprint TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, key)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cache_entries_logical
            ON cache_entries(namespace, logical_key)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_leases (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (namespace, key)
            )
            """
        )

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        """Add diagnostics and lease support to a v1 cache."""
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(cache_entries)")
        }
        if "logical_key" not in columns:
            connection.execute("ALTER TABLE cache_entries ADD COLUMN logical_key TEXT")
        if "config_fingerprint" not in columns:
            connection.execute(
                "ALTER TABLE cache_entries ADD COLUMN config_fingerprint TEXT"
            )
        SQLiteCache._create_latest_schema(connection)
        connection.execute(
            "UPDATE cache_properties SET value = '2' WHERE name = 'schema_version'"
        )

    def get(self, key: str) -> tuple[Any, dict[str, Any]] | None:
        """Return a decoded value and metadata for a key."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT value_json, metadata_json
                FROM cache_entries
                WHERE namespace = ? AND key = ?
                """,
                (self.namespace, key),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]), json.loads(row[1])

    def put_many(
        self,
        entries: list[tuple[str, Any, dict[str, Any]]],
        *,
        identities: dict[str, tuple[str, str]] | None = None,
    ) -> int:
        """Insert entries atomically with first-writer-wins semantics."""
        if not entries:
            return 0
        records = [
            (
                self.namespace,
                key,
                json.dumps(value, sort_keys=True, default=str),
                json.dumps(metadata, sort_keys=True, default=str),
                (identities or {}).get(key, (None, None))[0],
                (identities or {}).get(key, (None, None))[1],
            )
            for key, value, metadata in entries
        ]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changes_before = connection.total_changes
            connection.executemany(
                """
                INSERT INTO cache_entries(
                    namespace, key, value_json, metadata_json,
                    logical_key, config_fingerprint
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO NOTHING
                """,
                records,
            )
            return connection.total_changes - changes_before

    def publish(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any],
        *,
        owner_id: str,
        logical_key: str,
        config_fingerprint: str,
    ) -> bool:
        """Publish a claimed result and release its lease atomically."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changes_before = connection.total_changes
            connection.execute(
                """
                INSERT INTO cache_entries(
                    namespace, key, value_json, metadata_json,
                    logical_key, config_fingerprint
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO NOTHING
                """,
                (
                    self.namespace,
                    key,
                    json.dumps(value, sort_keys=True, default=str),
                    json.dumps(metadata, sort_keys=True, default=str),
                    logical_key,
                    config_fingerprint,
                ),
            )
            inserted = connection.total_changes > changes_before
            connection.execute(
                """
                DELETE FROM cache_leases
                WHERE namespace = ? AND key = ? AND owner_id = ?
                """,
                (self.namespace, key, owner_id),
            )
            return inserted

    def try_acquire(
        self,
        key: str,
        owner_id: str,
        *,
        ttl_seconds: float,
        now: float | None = None,
    ) -> bool:
        """Claim missing work, replacing an expired lease if necessary."""
        current_time = time.time() if now is None else now
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    """
                    SELECT 1 FROM cache_entries
                    WHERE namespace = ? AND key = ?
                    """,
                    (self.namespace, key),
                ).fetchone()
                is not None
            ):
                return False
            connection.execute(
                """
                DELETE FROM cache_leases
                WHERE namespace = ? AND key = ? AND expires_at <= ?
                """,
                (self.namespace, key, current_time),
            )
            changes_before = connection.total_changes
            connection.execute(
                """
                INSERT INTO cache_leases(namespace, key, owner_id, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO NOTHING
                """,
                (self.namespace, key, owner_id, current_time + ttl_seconds),
            )
            return connection.total_changes > changes_before

    def renew_leases(
        self, keys: list[str], owner_id: str, *, ttl_seconds: float
    ) -> int:
        """Extend leases still owned by a worker."""
        if not keys:
            return 0
        with self._connect() as connection:
            changes_before = connection.total_changes
            connection.executemany(
                """
                UPDATE cache_leases
                SET expires_at = ?
                WHERE namespace = ? AND owner_id = ?
                  AND key = ?
                """,
                [
                    (
                        time.time() + ttl_seconds,
                        self.namespace,
                        owner_id,
                        key,
                    )
                    for key in keys
                ],
            )
            return connection.total_changes - changes_before

    def release_leases(self, keys: list[str], owner_id: str) -> None:
        """Release leases owned by a worker without publishing results."""
        if not keys:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                DELETE FROM cache_leases
                WHERE namespace = ? AND owner_id = ?
                  AND key = ?
                """,
                [(self.namespace, owner_id, key) for key in keys],
            )

    def find_alternate_configurations(
        self, requests: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """Return metadata for matching inputs produced by other configurations."""
        requested = dict(requests)
        if not requested:
            return []
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TEMP TABLE requested_cache_configs (
                    logical_key TEXT PRIMARY KEY,
                    config_fingerprint TEXT NOT NULL
                )
                """
            )
            connection.executemany(
                "INSERT INTO requested_cache_configs VALUES (?, ?)",
                requested.items(),
            )
            rows = connection.execute(
                """
                SELECT entries.metadata_json
                FROM cache_entries AS entries
                INNER JOIN requested_cache_configs AS requested
                    ON requested.logical_key = entries.logical_key
                WHERE entries.namespace = ?
                  AND entries.config_fingerprint IS NOT requested.config_fingerprint
                """,
                (self.namespace,),
            ).fetchall()
        return list(starmap(json.loads, rows))

    def get_property(self, name: str) -> str | None:
        """Return a database-level property."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM cache_properties WHERE name = ?", (name,)
            ).fetchone()
        return str(row[0]) if row is not None else None

    def set_property_if_absent(self, name: str, value: str) -> None:
        """Set a database-level property without replacing an existing value."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_properties(name, value)
                VALUES (?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (name, value),
            )

    def count(self) -> int:
        """Return the number of entries in this namespace."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE namespace = ?",
                (self.namespace,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def clear(self) -> None:
        """Delete all entries and leases in this namespace."""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM cache_entries WHERE namespace = ?", (self.namespace,)
            )
            connection.execute(
                "DELETE FROM cache_leases WHERE namespace = ?", (self.namespace,)
            )

    def size_bytes(self) -> int:
        """Return total on-disk size of the database and WAL sidecars."""
        return sum(
            path.stat().st_size
            for path in (
                self.database_path,
                Path(f"{self.database_path}-wal"),
                Path(f"{self.database_path}-shm"),
            )
            if path.exists()
        )


def inspect_cache(database_path: Path | str) -> dict[str, Any]:
    """Return schema, namespace, provenance, and lease details without mutation."""
    path = Path(database_path)
    if not path.exists():
        msg = f"Cache database does not exist: {path}"
        raise FileNotFoundError(msg)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "cache_entries" not in tables or "cache_properties" not in tables:
            msg = f"Not a benchmark-qed cache database: {path}"
            raise ValueError(msg)
        version_row = connection.execute(
            "SELECT value FROM cache_properties WHERE name = 'schema_version'"
        ).fetchone()
        entry_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(cache_entries)")
        }
        if "config_fingerprint" in entry_columns:
            namespace_rows = connection.execute(
                """
                SELECT namespace, COUNT(*), COUNT(DISTINCT config_fingerprint)
                FROM cache_entries
                GROUP BY namespace
                ORDER BY namespace
                """
            )
        else:
            namespace_rows = connection.execute(
                """
                SELECT namespace, COUNT(*), 0
                FROM cache_entries
                GROUP BY namespace
                ORDER BY namespace
                """
            )
        namespaces = [
            {
                "namespace": namespace,
                "entries": entry_count,
                "configurations": config_count,
            }
            for namespace, entry_count, config_count in namespace_rows
        ]
        provenance = [
            {
                "namespace": namespace,
                "created_at": created_at,
                "metadata": json.loads(metadata_json),
            }
            for namespace, created_at, metadata_json in connection.execute(
                """
                SELECT namespace, created_at, metadata_json
                FROM cache_entries
                ORDER BY created_at DESC
                LIMIT 10
                """
            )
        ]
        lease_count = (
            connection.execute(
                "SELECT COUNT(*) FROM cache_leases WHERE expires_at > ?",
                (time.time(),),
            ).fetchone()[0]
            if "cache_leases" in tables
            else 0
        )
    finally:
        connection.close()
    return {
        "path": str(path),
        "schema_version": int(version_row[0]) if version_row else None,
        "namespaces": namespaces,
        "active_leases": lease_count,
        "recent_provenance": provenance,
    }
