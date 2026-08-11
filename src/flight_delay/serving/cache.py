"""Small thread-safe inference-result cache."""

from __future__ import annotations

from threading import RLock
from typing import Any

from cachetools import TTLCache


class PredictionCache:
    """Bounded TTL cache whose values contain inference output only."""

    def __init__(self, *, maxsize: int = 1_024, ttl_seconds: float = 300) -> None:
        if maxsize <= 0 or ttl_seconds <= 0:
            raise ValueError("cache maxsize and TTL must be positive")
        self._cache: TTLCache[str, Any] = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        """Return a cached inference result, if it remains live."""

        with self._lock:
            return self._cache.get(key)

    def put(self, key: str, value: Any) -> None:
        """Cache only the deterministic model/route result."""

        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        """Discard runtime cache state during shutdown."""

        with self._lock:
            self._cache.clear()
