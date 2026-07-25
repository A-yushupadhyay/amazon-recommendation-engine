"""
strategies.py

Strategy pattern: RecommendationStrategy defines the interface, and we
swap concrete implementations (CF-based via DynamoDB, or trending
fallback) without the calling code needing to know which one it's using.
This is what makes it easy to add a third strategy later (e.g. a
content-based fallback) without touching the orchestration logic.
"""

from abc import ABC, abstractmethod


class RecommendationStrategy(ABC):
    @abstractmethod
    def get_recommendations(self, item_id: str, k: int):
        """Returns a list of {item_id, score} dicts, or None if this
        strategy can't produce a recommendation for this item."""
        raise NotImplementedError


class CollaborativeFilteringStrategy(RecommendationStrategy):
    """Recommends via the precomputed item-item similarity table in DynamoDB."""

    def __init__(self, repository):
        self._repository = repository  # injected dependency, not hardcoded

    def get_recommendations(self, item_id: str, k: int):
        return self._repository.get_similar_items(item_id, k)


class TrendingFallbackStrategy(RecommendationStrategy):
    """Cold-start / failure fallback: just returns the most popular items."""

    def __init__(self, trending_items: list):
        self._trending_items = trending_items  # injected dependency

    def get_recommendations(self, item_id: str, k: int):
        return [{"item_id": i, "score": None} for i in self._trending_items[:k]]
