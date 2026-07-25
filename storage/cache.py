"""
cache.py

Single Responsibility: this module knows about Redis and nothing else.
The rest of the app depends on this small interface, not on the redis
library directly -- if we ever swapped Redis for Memcached, only this
file changes.
"""

import json
import logging
import redis

logger = logging.getLogger("recsys-api")


class RedisCache:
    def __init__(self, host: str, port: int, ttl_seconds: int):
        self._client = redis.Redis(host=host, port=port, decode_responses=True)
        self._ttl = ttl_seconds

    def get(self, key: str):
        """Returns the cached value (deserialized), or None on miss or Redis failure."""
        try:
            cached = self._client.get(key)
            return json.loads(cached) if cached else None
        except redis.RedisError as e:
            logger.warning(f"Redis unavailable on GET, treating as cache miss: {e}")
            return None

    def set(self, key: str, value) -> None:
        """Writes to cache. Failures are swallowed -- a cache write failure
        should never break the actual response to the user."""
        try:
            self._client.setex(key, self._ttl, json.dumps(value))
        except redis.RedisError as e:
            logger.warning(f"Redis unavailable on SET, skipping cache write: {e}")
