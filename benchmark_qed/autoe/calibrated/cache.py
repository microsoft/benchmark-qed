# Copyright (c) 2025 Microsoft Corporation.
"""Persistent cache for calibrated absolute judgments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmark_qed.cache import (
    CachePublishResult,
    CacheStore,
    redact_sensitive_values,
    stable_fingerprint,
)

if TYPE_CHECKING:
    from graphrag_cache import CacheConfig

_CACHE_SCHEMA_VERSION = 1
_DEFAULT_LEASE_TTL_SECONDS = 600.0


def build_cache_metadata(
    *,
    model: str,
    llm_provider: str,
    init_args: dict[str, Any] | None,
    call_args: dict[str, Any] | None,
    custom_providers: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build credential-safe metadata identifying judge behavior."""
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "evaluator": "calibrated_absolute",
        "model": model,
        "llm_provider": llm_provider,
        "init_args": redact_sensitive_values(init_args or {}),
        "call_args": redact_sensitive_values(call_args or {}),
        "custom_providers": redact_sensitive_values(custom_providers or []),
        "policy_version": 1,
    }


def compute_logical_key(payload: dict[str, Any]) -> str:
    """Fingerprint one judgment independently from model configuration."""
    return stable_fingerprint(payload)


def compute_cache_key(logical_key: str, metadata: dict[str, Any]) -> str:
    """Fingerprint a judgment and complete judge configuration."""
    return stable_fingerprint({
        "logical_key": logical_key,
        "configuration": metadata,
    })


class CalibratedAbsoluteCache:
    """Concurrency-safe cache for ranking and exemplar judgments."""

    def __init__(
        self,
        cache_config: CacheConfig,
        *,
        lease_ttl_seconds: float = _DEFAULT_LEASE_TTL_SECONDS,
    ) -> None:
        self._store = CacheStore(
            cache_config,
            "calibrated_absolute",
            lease_ttl_seconds=lease_ttl_seconds,
        )
        self.lease_ttl_seconds = lease_ttl_seconds

    async def get(self, cache_key: str) -> dict[str, Any] | None:
        """Return a cached structured response."""
        entry = await self._store.get(cache_key)
        if entry is None:
            return None
        if not isinstance(entry[0], dict):
            msg = f"Invalid calibrated absolute cache entry for key {cache_key}"
            raise TypeError(msg)
        return entry[0]

    async def claim(self, cache_key: str, owner_id: str) -> bool:
        """Claim an uncached judgment."""
        return await self._store.claim(
            cache_key,
            owner_id,
            ttl_seconds=self.lease_ttl_seconds,
        )

    def renew(self, cache_key: str, owner_id: str) -> bool:
        """Renew an active judgment lease."""
        return (
            self._store.renew_leases(
                [cache_key],
                owner_id,
                ttl_seconds=self.lease_ttl_seconds,
            )
            == 1
        )

    def release(self, cache_key: str, owner_id: str) -> None:
        """Release a judgment lease without publishing."""
        self._store.release_leases([cache_key], owner_id)

    async def publish(
        self,
        cache_key: str,
        value: dict[str, Any],
        metadata: dict[str, Any],
        *,
        owner_id: str,
        logical_key: str,
    ) -> CachePublishResult:
        """Publish a validated judgment and release its lease."""
        return await self._store.publish(
            cache_key,
            value,
            metadata,
            owner_id=owner_id,
            logical_key=logical_key,
            config_fingerprint=stable_fingerprint(metadata),
        )
