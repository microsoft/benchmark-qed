# Copyright (c) 2025 Microsoft Corporation.
"""Base classes for relevance assessment."""

import asyncio
import contextlib
import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.relevance import (
    RelevanceAssessmentItem,
    RelevanceAssessmentResponse,
)
from benchmark_qed.cache import (
    SQLiteCache,
    changed_configuration_fields,
    redact_sensitive_values,
    stable_fingerprint,
)

log: logging.Logger = logging.getLogger(__name__)
_LEASE_TTL_SECONDS = 120.0


class RelevanceRater(ABC):
    """Abstract base class for rating the relevance of text chunks to queries."""

    def __init__(
        self, cache_dir: Path | None = None, cache_enabled: bool = True
    ) -> None:
        """
        Initialize the RelevanceRater with optional caching.

        Args:
            cache_dir: Directory to store cache files. If None, caching is disabled.
            cache_enabled: Whether to enable caching functionality.
        """
        self.cache_dir: Path | None = cache_dir
        self.cache_enabled: bool = cache_enabled and cache_dir is not None
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self._cache_store: SQLiteCache | None = None

        if self.cache_enabled and self.cache_dir:
            self._cache_store = SQLiteCache(
                self.cache_dir / "relevance_cache.sqlite3",
                namespace=self.__class__.__name__,
            )
            self._migrate_legacy_cache_files()

    async def rate_relevance(
        self, query: str, text_units: list[TextUnit]
    ) -> RelevanceAssessmentResponse:
        """
        Rate the relevance of text units to a query with optional per-unit caching.

        Args:
            query: The query to assess relevance against.
            text_units: List of text units to assess.

        Returns
        -------
            RelevanceAssessmentResponse containing assessment results.
        """
        if not self.cache_enabled:
            # No caching - call implementation directly
            return await self._rate_relevance_impl(query, text_units)

        # With caching enabled, check each text unit individually
        cached_assessments: list[tuple[int, RelevanceAssessmentItem]] = []
        uncached_by_key: dict[str, tuple[TextUnit, list[int]]] = {}

        rater_params = self._get_cache_relevant_params()

        for i, text_unit in enumerate(text_units):
            cache_key = self._generate_cache_key(query, text_unit, rater_params)
            cached_assessment = self._load_from_cache(cache_key)

            if cached_assessment is not None:
                cached_assessments.append((i, cached_assessment))
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                pending = uncached_by_key.get(cache_key)
                if pending is None:
                    uncached_by_key[cache_key] = (text_unit, [i])
                else:
                    pending[1].append(i)

        # Process uncached text units if any
        if uncached_by_key:
            if self._cache_store is None:
                msg = "Caching is enabled but the cache store is unavailable"
                raise RuntimeError(msg)

            config_fingerprint = self._generate_config_fingerprint(rater_params)
            current_metadata = self._build_cache_metadata(query, rater_params)
            alternatives = self._cache_store.find_alternate_configurations([
                (
                    self._generate_logical_key(query, text_unit),
                    config_fingerprint,
                )
                for text_unit, _indices in uncached_by_key.values()
            ])
            if alternatives:
                changed_fields = sorted({
                    field
                    for metadata in alternatives
                    for field in changed_configuration_fields(
                        current_metadata, metadata
                    )
                })
                log.warning(
                    "Found %d cached relevance result(s) for the same inputs "
                    "with different %s; they will not be reused",
                    len(alternatives),
                    ", ".join(changed_fields) or "configuration metadata",
                )

            lease_owner = uuid.uuid4().hex
            owned = {
                cache_key: item
                for cache_key, item in uncached_by_key.items()
                if self._cache_store.try_acquire(
                    cache_key,
                    lease_owner,
                    ttl_seconds=_LEASE_TTL_SECONDS,
                )
            }
            waiting = {
                cache_key: item
                for cache_key, item in uncached_by_key.items()
                if cache_key not in owned
            }

            while owned or waiting:
                if owned:
                    await self._evaluate_and_publish_claimed(
                        query=query,
                        claimed=owned,
                        rater_params=rater_params,
                        config_fingerprint=config_fingerprint,
                        lease_owner=lease_owner,
                        cached_assessments=cached_assessments,
                    )
                    owned = {}

                completed_keys: list[str] = []
                recovered: dict[str, tuple[TextUnit, list[int]]] = {}
                for cache_key, item in waiting.items():
                    cached_assessment = self._load_from_cache(cache_key)
                    if cached_assessment is not None:
                        cached_assessments.extend(
                            (index, cached_assessment) for index in item[1]
                        )
                        completed_keys.append(cache_key)
                    elif self._cache_store.try_acquire(
                        cache_key,
                        lease_owner,
                        ttl_seconds=_LEASE_TTL_SECONDS,
                    ):
                        recovered[cache_key] = item
                        completed_keys.append(cache_key)
                for cache_key in completed_keys:
                    waiting.pop(cache_key)
                owned = recovered
                if waiting and not owned:
                    await asyncio.sleep(0.1)

        # Combine cached and uncached results in original order
        assessments_by_index = dict(cached_assessments)
        all_assessments = [
            assessments_by_index[index] for index in range(len(text_units))
        ]

        return RelevanceAssessmentResponse(assessment=all_assessments)

    async def _evaluate_and_publish_claimed(
        self,
        *,
        query: str,
        claimed: dict[str, tuple[TextUnit, list[int]]],
        rater_params: dict[str, Any],
        config_fingerprint: str,
        lease_owner: str,
        cached_assessments: list[tuple[int, RelevanceAssessmentItem]],
    ) -> None:
        """Evaluate claimed units while renewing and atomically releasing leases."""
        if self._cache_store is None:
            msg = "Cache store is unavailable"
            raise RuntimeError(msg)
        cache_store = self._cache_store
        cache_keys = list(claimed)

        async def _heartbeat() -> None:
            while True:
                await asyncio.sleep(max(0.1, _LEASE_TTL_SECONDS / 3))
                cache_store.renew_leases(
                    cache_keys,
                    lease_owner,
                    ttl_seconds=_LEASE_TTL_SECONDS,
                )

        heartbeat = asyncio.create_task(_heartbeat())
        try:
            response = await self._rate_relevance_impl(
                query,
                [text_unit for text_unit, _indices in claimed.values()],
            )
            self._validate_result_count(response, len(claimed))

            for (
                cache_key,
                (text_unit, indices),
            ), assessment in zip(claimed.items(), response.assessment, strict=True):
                cache_store.publish(
                    cache_key,
                    assessment.model_dump(exclude={"text_unit": {"text_embedding"}}),
                    self._build_cache_metadata(query, rater_params),
                    owner_id=lease_owner,
                    logical_key=self._generate_logical_key(query, text_unit),
                    config_fingerprint=config_fingerprint,
                )
                cached_assessments.extend(
                    (original_idx, assessment) for original_idx in indices
                )
        except (Exception, asyncio.CancelledError):
            cache_store.release_leases(cache_keys, lease_owner)
            raise
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    @staticmethod
    def _validate_result_count(
        response: RelevanceAssessmentResponse, expected_count: int
    ) -> None:
        """Raise when a rater violates its one-result-per-input contract."""
        if len(response.assessment) != expected_count:
            msg = (
                "Relevance rater returned "
                f"{len(response.assessment)} results for "
                f"{expected_count} uncached text units"
            )
            raise RuntimeError(msg)

    @abstractmethod
    async def _rate_relevance_impl(
        self, query: str, text_units: list[TextUnit]
    ) -> RelevanceAssessmentResponse:
        """
        Concrete implementation of relevance rating (to be implemented by subclasses).

        Args:
            query: The query to assess relevance against.
            text_units: List of text units to assess.

        Returns
        -------
            RelevanceAssessmentResponse containing assessment results.
        """

    def _generate_cache_key(
        self, query: str, text_unit: TextUnit, rater_params: dict[str, Any]
    ) -> str:
        """Generate deterministic cache key for a single text unit assessment."""
        cache_data = {
            "query": query.strip().lower(),
            "text_content": text_unit.text.strip().lower(),
            "rater_type": self.__class__.__name__,
            "rater_params": redact_sensitive_values(rater_params),
        }

        content_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def _generate_logical_key(self, query: str, text_unit: TextUnit) -> str:
        """Fingerprint query and text independently from rater configuration."""
        return stable_fingerprint({
            "query": query.strip().lower(),
            "text_content": text_unit.text.strip().lower(),
            "rater_type": self.__class__.__name__,
        })

    @staticmethod
    def _generate_config_fingerprint(rater_params: dict[str, Any]) -> str:
        """Fingerprint relevance settings independently from query and text."""
        return stable_fingerprint(rater_params)

    def _build_cache_metadata(
        self, query: str, rater_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Build inspectable metadata for a relevance assessment."""
        return {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "query": query.strip().lower(),
            "rater_type": self.__class__.__name__,
            "rater_params": redact_sensitive_values(rater_params),
        }

    def _load_from_cache(self, cache_key: str) -> RelevanceAssessmentItem | None:
        """Load cached result for a single text unit if available."""
        if self._cache_store is None:
            return None

        try:
            entry = self._cache_store.get(cache_key)
            if entry is None:
                return None
            assessment_data, _metadata = entry
            return RelevanceAssessmentItem(**assessment_data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            log.warning("Ignoring invalid relevance cache entry %s: %s", cache_key, exc)
            return None

    def _migrate_legacy_cache_files(self) -> None:
        """Import legacy per-key JSON files once without deleting them."""
        if self._cache_store is None or self.cache_dir is None:
            return
        migration_name = f"legacy_relevance_import:{self.__class__.__name__}"
        if self._cache_store.get_property(migration_name) is not None:
            return

        entries: list[tuple[str, Any, dict[str, Any]]] = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with cache_file.open(encoding="utf-8") as file:
                    data = json.load(file)
                assessment_data = data["assessment_item"]
                RelevanceAssessmentItem(**assessment_data)
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                log.warning(
                    "Could not import legacy relevance cache file %s: %s",
                    cache_file,
                    exc,
                )
                continue
            metadata = {
                "source": "legacy_json",
                "timestamp": data.get("timestamp"),
                "query": data.get("query"),
                "rater_type": self.__class__.__name__,
            }
            entries.append((cache_file.stem, assessment_data, metadata))

        self._cache_store.put_many(entries)
        self._cache_store.set_property_if_absent(migration_name, "complete")

    def _get_cache_relevant_params(self) -> dict[str, Any]:
        """
        Get parameters that affect the relevance assessment results.

        Subclasses should override this to include their specific parameters
        that could change the assessment output (LLM config, prompts, etc.).

        Returns
        -------
            Dictionary of parameter names to values that affect results.
        """
        return {}

    def get_relevant_contexts(
        self,
        result: RelevanceAssessmentResponse,
        relevance_threshold: int = 2,
    ) -> list[RelevanceAssessmentItem]:
        """
        Filter assessment results to return only items that meet the relevance threshold.

        Args:
            result: The RelevanceAssessmentResponse containing assessment results.
            relevance_threshold: Minimum relevance score threshold (items with score >= threshold are returned).

        Returns
        -------
            List of RelevanceAssessmentItem objects that meet or exceed the threshold.
        """
        return [item for item in result.assessment if item.score >= relevance_threshold]

    def supports_caching(self) -> bool:
        """
        Check if this rater supports caching.

        Returns
        -------
            True if caching is enabled, False otherwise.
        """
        return self.cache_enabled

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
            Dictionary with cache stats.
        """
        if not self.cache_enabled:
            return {"caching_enabled": False}

        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        cache_files = 0
        cache_size_mb = 0

        if self._cache_store is not None:
            cache_files = self._cache_store.count()
            cache_size_mb = self._cache_store.size_bytes() / (1024 * 1024)

        return {
            "caching_enabled": True,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 1),
            "cache_files": cache_files,
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if self._cache_store is None or self.cache_dir is None:
            return

        self._cache_store.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except OSError as exc:
                log.warning(
                    "Failed to remove legacy relevance cache file %s: %s",
                    cache_file,
                    exc,
                )

        # Reset statistics
        self.cache_hits = 0
        self.cache_misses = 0
