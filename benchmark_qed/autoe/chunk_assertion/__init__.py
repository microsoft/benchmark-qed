# Copyright (c) 2025 Microsoft Corporation.
"""Chunk-level assertion evaluation module."""

from benchmark_qed.autoe.chunk_assertion.aggregation import summarize_at_k
from benchmark_qed.autoe.chunk_assertion.cache import (
    ContentAddressedCache,
    compute_cache_key,
)
from benchmark_qed.autoe.chunk_assertion.scoring import (
    run_assertion_eval_chunk_mode,
)

__all__ = [
    "ContentAddressedCache",
    "compute_cache_key",
    "run_assertion_eval_chunk_mode",
    "summarize_at_k",
]
