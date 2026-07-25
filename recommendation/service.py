"""
service.py

Dependency Injection: RecommendationService doesn't construct its own
cache or strategies -- they're passed in. This means we can unit test
this class with fake/mock cache and strategy objects, without needing a
real Redis or DynamoDB running.

Orchestration logic: cache -> primary strategy (CF) -> fallback strategy
(trending). This is the read path we've already validated end-to-end.
"""


class RecommendationService:
    def __init__(self, cache, primary_strategy, fallback_strategy):
        self._cache = cache
        self._primary_strategy = primary_strategy
        self._fallback_strategy = fallback_strategy

    def get_recommendations(self, item_id: str, k: int = 10):
        cache_key = f"item_rec:{item_id}:{k}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return {"item_id": item_id, "source": "cache", "recommendations": cached}

        recs = self._primary_strategy.get_recommendations(item_id, k)
        if recs is not None:
            self._cache.set(cache_key, recs)
            return {"item_id": item_id, "source": "dynamodb", "recommendations": recs}

        fallback_recs = self._fallback_strategy.get_recommendations(item_id, k)
        return {"item_id": item_id, "source": "trending_fallback", "recommendations": fallback_recs}
