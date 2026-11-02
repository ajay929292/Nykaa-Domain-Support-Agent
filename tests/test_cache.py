"""Unit Tests for In-Memory Query Response Caching (Task 16).

Tests:
1. Key normalization across casing, whitespace, and punctuation.
2. Cache miss triggers downstream generator function.
3. Cache hit returns stored response with zero downstream invocations.
4. Latency comparison and hit counter validation.
"""

import time
import pytest
from cache.query_cache import (
    normalize_query_key,
    get_cached_response,
    store_cached_response,
    clear_cache,
    get_cache_stats,
    execute_with_cache
)


def test_query_key_normalization():
    """Verify normalization strips whitespace, lowers case, and removes trailing punctuation."""
    q1 = "  What is the Return Window for Lipsticks?  "
    q2 = "what is the return window for lipsticks"
    q3 = "What is the return window for lipsticks???"

    norm1 = normalize_query_key(q1)
    norm2 = normalize_query_key(q2)
    norm3 = normalize_query_key(q3)

    assert norm1 == norm2 == norm3 == "what is the return window for lipsticks"


def test_cache_hit_and_miss_lifecycle():
    """Verify lifecycle: Miss on first call, Hit on second call, avoiding downstream execution."""
    clear_cache()

    calls_executed = 0

    def mock_generator(q: str):
        nonlocal calls_executed
        calls_executed += 1
        time.sleep(0.01)  # Simulate small generation latency
        return {"answer": f"Answer for {q}", "confidence": 1.0}

    query = "What is the COD refund timeline?"

    # Call 1: Miss
    t0 = time.perf_counter()
    res1 = execute_with_cache(query, mock_generator)
    lat1 = time.perf_counter() - t0

    assert res1["cache_hit"] is False
    assert calls_executed == 1

    stats1 = get_cache_stats()
    assert stats1["hits"] == 0
    assert stats1["misses"] == 1
    assert stats1["downstream_calls"] == 1

    # Call 2: Hit with query variation
    query_variation = "what is the cod refund timeline???"
    t1 = time.perf_counter()
    res2 = execute_with_cache(query_variation, mock_generator)
    lat2 = time.perf_counter() - t1

    assert res2["cache_hit"] is True
    # Crucial assertion: Downstream generator was NOT called again
    assert calls_executed == 1

    stats2 = get_cache_stats()
    assert stats2["hits"] == 1
    assert stats2["downstream_calls"] == 1

    # Cache hit must be significantly faster
    assert lat2 < lat1

