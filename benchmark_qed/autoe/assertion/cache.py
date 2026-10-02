# Copyright (c) 2025 Microsoft Corporation.
"""Content-addressed cache for answer-level assertion judgments."""

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

_CACHE_NAMESPACE = "assertion"
_CACHE_SCHEMA_VERSION = 1
_DEFAULT_LEASE_TTL_SECONDS = 120.0


def build_cache_metadata(
    *,
    model: str,
    llm_provider: str,
    init_args: dict[str, Any] | None,
    call_args: dict[str, Any] | None,
    custom_providers: list[dict[str, Any]] | None,
    system_prompt: str,
    user_prompt: str,
    include_score_id_in_prompt: bool,
) -> dict[str, Any]:
    """Build credential-safe metadata that identifies judge configuration."""
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "evaluator": _CACHE_NAMESPACE,
        "model": model,
        "llm_provider": llm_provider,
        "init_args": redact_sensitive_values(init_args or {}),
        "call_args": redact_sensitive_values(call_args or {}),
        "custom_providers": redact_sensitive_values(custom_providers or []),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "include_score_id_in_prompt": include_score_id_in_prompt,
    }


def compute_logical_key(
    *,
    question: str,
    answer: str,
    assertion: str,
    trial: int,
) -> str:
    """Fingerprint one assertion trial independently from judge configuration."""
    return stable_fingerprint({
        "question": question,
        "answer": answer,
        "assertion": assertion,
        "trial": trial,
    })


def compute_cache_key(logical_key: str, metadata: dict[str, Any]) -> str:
    """Fingerprint an assertion trial and complete judge configuration."""
    return stable_fingerprint({
        "logical_key": logical_key,
        "configuration": metadata,
    })


class AssertionScoreCache:
    """Persistent, concurrency-safe cache for assertion score dictionaries."""

    def __init__(
        self,
        cache_config: CacheConfig,
        *,
        lease_ttl_seconds: float = _DEFAULT_LEASE_TTL_SECONDS,
    ) -> None:
        self._store = CacheStore(
            cache_config,
            _CACHE_NAMESPACE,
            lease_ttl_seconds=lease_ttl_seconds,
        )
        self.lease_ttl_seconds = lease_ttl_seconds

    async def get(self, cache_key: str) -> dict[str, Any] | None:
        """Return a cached assertion score."""
        entry = await self._store.get(cache_key)
        if entry is None:
            return None
        if not isinstance(entry[0], dict):
            msg = f"Invalid cached assertion result for key {cache_key}"
            raise TypeError(msg)
        return entry[0]

    async def claim(self, cache_key: str, owner_id: str) -> bool:
        """Claim an uncached assertion score for this worker."""
        return await self._store.claim(
            cache_key,
            owner_id,
            ttl_seconds=self.lease_ttl_seconds,
        )

    def renew(self, cache_key: str, owner_id: str) -> bool:
        """Renew an active assertion-score lease."""
        return (
            self._store.renew_leases(
                [cache_key],
                owner_id,
                ttl_seconds=self.lease_ttl_seconds,
            )
            == 1
        )

    def release(self, cache_key: str, owner_id: str) -> None:
        """Release an assertion-score lease without publishing."""
        self._store.release_leases([cache_key], owner_id)

    async def publish(
        self,
        cache_key: str,
        score: dict[str, Any],
        metadata: dict[str, Any],
        *,
        owner_id: str,
        logical_key: str,
    ) -> CachePublishResult:
        """Publish an assertion score and release its lease."""
        return await self._store.publish(
            cache_key,
            score,
            metadata,
            owner_id=owner_id,
            logical_key=logical_key,
            config_fingerprint=stable_fingerprint(metadata),
        )
