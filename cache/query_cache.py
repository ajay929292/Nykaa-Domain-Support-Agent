"""In-Memory Normalized Response Caching for Grounded Generation (Task 16).

Implements:
- Key normalization: lowercased, whitespace-stripped, punctuation-normalized.
- Hit / miss / downstream call counters.
- Latency benchmarking demonstrating that cache hits completely avoid redundant tool calls.
"""

import re
import time
from typing import Dict, Any, Optional, Callable

_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
_STATS = {
    "hits": 0,
    "misses": 0,
    "downstream_calls": 0
}


def normalize_query_key(query: str) -> str:
    """Normalizes query string for robust cache key matching."""
    # Strip whitespace, lowercase, remove trailing punctuation
    clean = " ".join(query.strip().lower().split())
    clean = re.sub(r"[?!.,;:]+$", "", clean)
    return clean


def get_cached_response(query: str) -> Optional[Dict[str, Any]]:
    """Retrieves response from cache if present; updates hit/miss stats."""
    key = normalize_query_key(query)
    if key in _QUERY_CACHE:
        _STATS["hits"] += 1
        return _QUERY_CACHE[key]
    _STATS["misses"] += 1
    return None


def store_cached_response(query: str, response_data: Dict[str, Any]) -> None:
    """Stores response data in the in-memory cache."""
    key = normalize_query_key(query)
    _QUERY_CACHE[key] = response_data


def clear_cache() -> None:
    """Clears cache entries and resets statistics."""
    global _QUERY_CACHE, _STATS
    _QUERY_CACHE.clear()
    _STATS = {"hits": 0, "misses": 0, "downstream_calls": 0}


def get_cache_stats() -> Dict[str, int]:
    """Returns current cache statistics."""
    return dict(_STATS)


def execute_with_cache(
    query: str,
    generator_fn: Callable[[str], Dict[str, Any]]
) -> Dict[str, Any]:
    """Executes generator_fn on cache miss; returns cached entry on cache hit."""
    cached = get_cached_response(query)
    if cached is not None:
        return {
            **cached,
            "cache_hit": True
        }

    # Cache Miss: Invoke downstream generator function
    _STATS["downstream_calls"] += 1
    fresh_response = generator_fn(query)
    store_cached_response(query, fresh_response)

    return {
        **fresh_response,
        "cache_hit": False
    }


def demonstrate_caching():
    """Demonstrates Task 16: repeated identical query producing a cache hit."""
    clear_cache()
    from rag.retriever import generate_grounded_answer

    query_1 = "What is the return window for makeup and beauty products?"
    query_2 = "what is the return window for makeup and beauty products"  # Case and punctuation variation

    print("=" * 80)
    print("IN-MEMORY RESPONSE CACHING DEMONSTRATION (Task 16)")
    print("=" * 80)

    # First Call: Cache Miss
    t0 = time.perf_counter()
    res1 = execute_with_cache(query_1, lambda q: generate_grounded_answer(q))
    lat1_ms = (time.perf_counter() - t0) * 1000.0

    stats1 = get_cache_stats()
    print("\n[Call 1: Initial Query]")
    print(f"Query           : \"{query_1}\"")
    print(f"Cache Hit       : {res1['cache_hit']}")
    print(f"Latency         : {lat1_ms:.2f} ms")
    print(f"Downstream Calls: {stats1['downstream_calls']}")
    print(f"Stats           : {stats1}")

    # Second Call: Cache Hit
    t1 = time.perf_counter()
    res2 = execute_with_cache(query_2, lambda q: generate_grounded_answer(q))
    lat2_ms = (time.perf_counter() - t1) * 1000.0

    stats2 = get_cache_stats()
    print("\n[Call 2: Repeated Normalized Query]")
    print(f"Query           : \"{query_2}\"")
    print(f"Cache Hit       : {res2['cache_hit']}")
    print(f"Latency         : {lat2_ms:.4f} ms")
    print(f"Downstream Calls: {stats2['downstream_calls']} (Zero increase!)")
    print(f"Stats           : {stats2}")

    print("\n" + "-" * 80)
    print(f"Speedup Factor  : {lat1_ms / max(0.0001, lat2_ms):.1f}x faster")
    print(f"Tool Invocations Avoided on Call 2: YES (downstream_calls remained {stats2['downstream_calls']})")
    print("=" * 80)

    assert res1["cache_hit"] is False
    assert res2["cache_hit"] is True
    assert stats2["hits"] == 1
    assert stats2["downstream_calls"] == 1


if __name__ == "__main__":
    demonstrate_caching()

