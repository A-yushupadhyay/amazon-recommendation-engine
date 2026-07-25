"""
dynamo_repository.py

Single Responsibility: this module knows about DynamoDB and nothing else.
Includes a simple circuit breaker so repeated DynamoDB failures don't
keep hammering a dead dependency -- after a threshold of failures, we
skip straight to returning None (letting the caller fall back) for a
cooldown period.
"""

import time
import logging
import boto3

logger = logging.getLogger("recsys-api")


class ItemSimilarityRepository:
    def __init__(self, table_name: str, endpoint_url: str, region: str,
                 access_key: str, secret_key: str,
                 failure_threshold: int = 3, cooldown_seconds: int = 30):
        self._dynamodb = boto3.resource(
            "dynamodb",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self._table = self._dynamodb.Table(table_name)
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._circuit_open_until = 0

    def _circuit_is_open(self) -> bool:
        return time.time() < self._circuit_open_until

    def _record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._circuit_open_until = time.time() + self._cooldown_seconds
            logger.warning(
                f"DynamoDB circuit opened for {self._cooldown_seconds}s "
                f"after {self._failure_count} consecutive failures"
            )

    def _record_success(self):
        self._failure_count = 0

    def get_similar_items(self, item_id: str, k: int):
        """
        Returns a list of {item_id, score} dicts, or None if the item
        isn't found OR the circuit breaker is currently open (DynamoDB
        assumed unhealthy). Callers should treat None as "fall back".
        """
        if self._circuit_is_open():
            logger.warning("DynamoDB circuit open, skipping request")
            return None

        try:
            response = self._table.get_item(Key={"item_id": item_id})
            self._record_success()
        except Exception as e:
            logger.error(f"DynamoDB error: {e}")
            self._record_failure()
            return None

        if "Item" not in response:
            return None

        similar_items = response["Item"].get("similar_items", [])
        similar_items_sorted = sorted(similar_items, key=lambda x: float(x["score"]), reverse=True)
        return [
            {"item_id": s["item_id"], "score": float(s["score"])}
            for s in similar_items_sorted[:k]
        ]
