"""
main.py

Phase 5 (refactored): main.py is now just routing + startup wiring.
All the actual logic lives in storage/ and recommendation/, each with a
single responsibility. This file's only job is to construct the
concrete objects once at startup and inject them into the service --
everything downstream depends on interfaces, not concrete classes.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8001 --limit-concurrency 100
"""

import os
import logging

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from storage.cache import RedisCache
from storage.dynamo_repository import ItemSimilarityRepository
from recommendation.strategies import CollaborativeFilteringStrategy, TrendingFallbackStrategy
from recommendation.service import RecommendationService

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recsys-api")

app = FastAPI(title="Recommendation Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

TRENDING_PARQUET = "data/processed/interactions_train.parquet"

_service = None
_trending_items_cache = []


@app.on_event("startup")
def startup():
    global _service, _trending_items_cache

    try:
        df = pd.read_parquet(TRENDING_PARQUET)
        _trending_items_cache = df["item_id"].value_counts().head(50).index.tolist()
        logger.info(f"Loaded {len(_trending_items_cache)} trending items for fallback")
    except Exception as e:
        logger.error(f"Failed to load trending items fallback: {e}")
        _trending_items_cache = []

    cache = RedisCache(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", 300)),
    )

    repository = ItemSimilarityRepository(
        table_name=os.getenv("DYNAMODB_TABLE_NAME", "ItemSimilarity"),
        endpoint_url=os.getenv("DYNAMODB_ENDPOINT", "http://localhost:8000"),
        region=os.getenv("AWS_REGION", "us-east-1"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID", "fakeMyKeyId"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", "fakeSecretAccessKey"),
    )

    primary_strategy = CollaborativeFilteringStrategy(repository)
    fallback_strategy = TrendingFallbackStrategy(_trending_items_cache)

    _service = RecommendationService(cache, primary_strategy, fallback_strategy)
    logger.info("RecommendationService wired up and ready")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/items")
def list_items(limit: int = 30):
    return {"items": _trending_items_cache[:limit]}


@app.get("/recommendations/{item_id}")
def get_recommendations(item_id: str, k: int = 10):
    return _service.get_recommendations(item_id, k)


@app.get("/recommendations/user/{user_id}")
def get_user_recommendations(user_id: str, k: int = 10):
    return {
        "user_id": user_id,
        "source": "trending_fallback",
        "recommendations": [{"item_id": i, "score": None} for i in _trending_items_cache[:k]],
    }
