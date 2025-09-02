# Copyright (c) 2025 Microsoft Corporation.
"""Base classes for relevance assessment."""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark_qed.autod.data_model.text_unit import TextUnit
from benchmark_qed.autoe.data_model.relevance import RelevanceAssessmentItem, RelevanceAssessmentResponse


class RelevanceRater(ABC):
    """Abstract base class for rating the relevance of text chunks to queries."""

    def __init__(self, cache_dir: Path | None = None, cache_enabled: bool = True):
        """
        Initialize the RelevanceRater with optional caching.
        
        Args:
            cache_dir: Directory to store cache files. If None, caching is disabled.
            cache_enabled: Whether to enable caching functionality.
        """
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled and cache_dir is not None
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.cache_enabled and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def rate_relevance(
        self, query: str, text_units: list[TextUnit]
    ) -> RelevanceAssessmentResponse:
        """
        Rate the relevance of text units to a query with optional per-unit caching.

        Args:
            query: The query to assess relevance against.
            text_units: List of text units to assess.

        Returns:
            RelevanceAssessmentResponse containing assessment results.
        """
        if not self.cache_enabled:
            # No caching - call implementation directly
            return await self._rate_relevance_impl(query, text_units)
        
        # With caching enabled, check each text unit individually
        cached_assessments = []
        uncached_text_units = []
        uncached_indices = []
        
        rater_params = self._get_cache_relevant_params()
        
        for i, text_unit in enumerate(text_units):
            cache_key = self._generate_cache_key(query, text_unit, rater_params)
            cached_assessment = self._load_from_cache(cache_key)
            
            if cached_assessment is not None:
                cached_assessments.append((i, cached_assessment))
                self.cache_hits += 1
            else:
                uncached_text_units.append(text_unit)
                uncached_indices.append(i)
                self.cache_misses += 1
        
        # Process uncached text units if any
        uncached_results = []
        if uncached_text_units:
            uncached_response = await self._rate_relevance_impl(query, uncached_text_units)
            uncached_results = uncached_response.assessment
            
            # Cache individual results
            for j, text_unit in enumerate(uncached_text_units):
                cache_key = self._generate_cache_key(query, text_unit, rater_params)
                self._save_to_cache(cache_key, query, uncached_results[j])
        
        # Combine cached and uncached results in original order
        all_assessments: list[RelevanceAssessmentItem] = [None] * len(text_units)  # type: ignore
        
        # Place cached results
        for original_idx, cached_assessment in cached_assessments:
            all_assessments[original_idx] = cached_assessment
        
        # Place uncached results
        for j, original_idx in enumerate(uncached_indices):
            all_assessments[original_idx] = uncached_results[j]
        
        return RelevanceAssessmentResponse(assessment=all_assessments)

    @abstractmethod
    async def _rate_relevance_impl(
        self, query: str, text_units: list[TextUnit]
    ) -> RelevanceAssessmentResponse:
        """
        Concrete implementation of relevance rating (to be implemented by subclasses).

        Args:
            query: The query to assess relevance against.
            text_units: List of text units to assess.

        Returns:
            RelevanceAssessmentResponse containing assessment results.
        """
        pass

    def _generate_cache_key(self, query: str, text_unit: TextUnit, rater_params: dict[str, Any]) -> str:
        """Generate deterministic cache key for a single text unit assessment."""
        cache_data = {
            "query": query.strip().lower(),
            "text_content": text_unit.text.strip().lower(),
            "rater_type": self.__class__.__name__,
            "rater_params": rater_params
        }
        
        content_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> RelevanceAssessmentItem | None:
        """Load cached result for a single text unit if available."""
        if not self.cache_enabled or not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return RelevanceAssessmentItem(**data['assessment_item'])
        except Exception:
            # If cache file is corrupted, ignore and continue
            return None

    def _save_to_cache(self, cache_key: str, query: str, assessment_item: RelevanceAssessmentItem) -> None:
        """Save single text unit assessment result to cache."""
        if not self.cache_enabled or not self.cache_dir:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Create a copy of the assessment item without text embedding to reduce cache size
        assessment_data = assessment_item.model_dump()
        if "text_unit" in assessment_data and assessment_data["text_unit"] is not None:
            # Remove text_embedding from the text_unit to save space
            if "text_embedding" in assessment_data["text_unit"]:
                assessment_data["text_unit"]["text_embedding"] = None
        
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query.strip().lower(),
            "assessment_item": assessment_data
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            # If we can't write cache, continue without it
            pass

    def _get_cache_relevant_params(self) -> dict[str, Any]:
        """
        Get parameters that affect the relevance assessment results.
        
        Subclasses should override this to include their specific parameters
        that could change the assessment output (LLM config, prompts, etc.).
        
        Returns:
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

        Returns:
            List of RelevanceAssessmentItem objects that meet or exceed the threshold.
        """
        return [
            item for item in result.assessment 
            if item.score >= relevance_threshold
        ]

    def supports_caching(self) -> bool:
        """
        Check if this rater supports caching.
        
        Returns:
            True if caching is enabled, False otherwise.
        """
        return self.cache_enabled

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats.
        """
        if not self.cache_enabled:
            return {"caching_enabled": False}
            
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Count cache files
        cache_files = 0
        cache_size_mb = 0
        
        if self.cache_dir and self.cache_dir.exists():
            all_cache_files = list(self.cache_dir.glob("*.json"))
            cache_files = len(all_cache_files)
            cache_size_mb = sum(f.stat().st_size for f in all_cache_files) / (1024 * 1024)
        
        return {
            "caching_enabled": True,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 1),
            "cache_files": cache_files,
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None
        }

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if not self.cache_enabled or not self.cache_dir:
            return
            
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception:
                pass
        
        # Reset statistics
        self.cache_hits = 0
        self.cache_misses = 0
