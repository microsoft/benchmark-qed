# Copyright (c) 2025 Microsoft Corporation.
"""Stage-aware cache for differential pairwise scoring."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import TYPE_CHECKING, Any, Literal

from benchmark_qed.cache import CacheStore, redact_sensitive_values, stable_fingerprint

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from graphrag_cache import CacheConfig

DifferentialStage = Literal["extraction", "verdict"]

_CACHE_NAMESPACE = "differential_pairwise"
_CACHE_SCHEMA_VERSION = 1
_DEFAULT_LEASE_TTL_SECONDS = 120.0


def build_stage_metadata(
    *,
    stage: DifferentialStage,
    model: str,
    llm_provider: str,
    init_args: dict[str, Any] | None,
    call_args: dict[str, Any] | None,
    custom_providers: list[dict[str, Any]] | None,
    system_prompt: str,
    user_prompt: str,
    include_score_id_in_prompt: bool,
) -> dict[str, Any]:
    """Build credential-safe metadata for one differential scoring stage."""
    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "evaluator": _CACHE_NAMESPACE,
        "stage": stage,
        "model": model,
        "llm_provider": llm_provider,
        "init_args": redact_sensitive_values(init_args or {}),
        "call_args": redact_sensitive_values(call_args or {}),
        "custom_providers": redact_sensitive_values(custom_providers or []),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "include_score_id_in_prompt": include_score_id_in_prompt,
    }


def compute_logical_key(stage: DifferentialStage, payload: dict[str, Any]) -> str:
    """Fingerprint stage inputs independently from model configuration."""
    return stable_fingerprint({"stage": stage, "inputs": payload})


def compute_cache_key(logical_key: str, metadata: dict[str, Any]) -> str:
    """Fingerprint stage inputs and model configuration."""
    return stable_fingerprint({
        "logical_key": logical_key,
        "configuration": metadata,
    })


class DifferentialPairwiseCache:
    """Persistent cache with independent extraction and verdict namespaces."""

    def __init__(
        self,
        cache_config: CacheConfig,
        *,
        lease_ttl_seconds: float = _DEFAULT_LEASE_TTL_SECONDS,
    ) -> None:
        self._stores = {
            stage: CacheStore(
                cache_config,
                f"{_CACHE_NAMESPACE}/{stage}",
                lease_ttl_seconds=lease_ttl_seconds,
            )
            for stage in ("extraction", "verdict")
        }
        self.lease_ttl_seconds = lease_ttl_seconds

    async def get_or_compute(
        self,
        *,
        stage: DifferentialStage,
        cache_key: str,
        logical_key: str,
        metadata: dict[str, Any],
        compute: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Return a cached stage result or compute and publish it once."""
        store = self._stores[stage]
        owner_id = uuid.uuid4().hex

        while True:
            entry = await store.get(cache_key)
            if entry is not None:
                if isinstance(entry[0], dict):
                    return entry[0]
                msg = f"Invalid cached {stage} result for key {cache_key}"
                raise RuntimeError(msg)
            if await store.claim(
                cache_key,
                owner_id,
                ttl_seconds=self.lease_ttl_seconds,
            ):
                break
            await asyncio.sleep(0.1)

        async def _heartbeat() -> None:
            while True:
                await asyncio.sleep(max(0.1, self.lease_ttl_seconds / 3))
                store.renew_leases(
                    [cache_key],
                    owner_id,
                    ttl_seconds=self.lease_ttl_seconds,
                )

        heartbeat = asyncio.create_task(_heartbeat())
        try:
            result = await compute()
            publication = await store.publish(
                cache_key,
                result,
                metadata,
                owner_id=owner_id,
                logical_key=logical_key,
                config_fingerprint=stable_fingerprint(metadata),
            )
            if publication.accepted and isinstance(publication.value, dict):
                result = publication.value
        except (Exception, asyncio.CancelledError):
            store.release_leases([cache_key], owner_id)
            raise
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

        return result
