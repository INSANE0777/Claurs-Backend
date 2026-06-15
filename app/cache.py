import hashlib
import json
from typing import Any, Optional
from functools import lru_cache

from app.config import get_settings

settings = get_settings()


class MemoryCache:
    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: dict = {}
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        if len(self._cache) >= self._maxsize and key not in self._cache:
            # Simple eviction: drop oldest key
            if self._cache:
                self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


def _make_cache_key(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return f"{prefix}:{hashlib.sha256(payload.encode()).hexdigest()}"


class CacheManager:
    def __init__(self) -> None:
        self._redis = None
        self._memory = MemoryCache(maxsize=settings.CACHE_MAX_SIZE)
        self._ttl = settings.CACHE_TTL_SECONDS
        if settings.REDIS_URL:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
            except Exception:
                self._redis = None

    async def get(self, key: str) -> Optional[Any]:
        if self._redis:
            try:
                val = await self._redis.get(key)
                if val is not None:
                    return json.loads(val)
            except Exception:
                pass
        return self._memory.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl or self._ttl
        if self._redis:
            try:
                await self._redis.set(key, json.dumps(value, default=str), ex=ttl)
                return
            except Exception:
                pass
        self._memory.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception:
                pass
        self._memory.delete(key)

    async def invalidate_pattern(self, pattern: str) -> None:
        if self._redis:
            try:
                keys = await self._redis.keys(pattern)
                if keys:
                    await self._redis.delete(*keys)
            except Exception:
                pass
        # Memory cache: clear all to keep it simple
        self._memory.clear()

    def make_key(self, *parts: Any) -> str:
        return _make_cache_key("search", *parts)


cache_manager = CacheManager()
