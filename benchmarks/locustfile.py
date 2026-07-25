"""
locustfile.py

Phase 2: Load test for the recommendations API. Measures p50/p95/p99
latency under concurrent load.

Usage (from project root, with the API already running on port 8001):
    locust -f benchmarks/locustfile.py --host http://localhost:8001

Then open http://localhost:8089 in your browser, set number of users
(e.g. 20) and spawn rate (e.g. 5/sec), and start the test.
"""

import random
from locust import HttpUser, task, between

SAMPLE_ITEM_IDS = [
    "B00005MOTE",
    "B0084PK8CS",
    "B003VIVYGM",
    "B002LTJIAG",
    "B002RS8F8W",
    "B000H2049I",
    "B00G6CLNCK",
    "B004Q0T0LU",
]


class RecsysUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def get_recommendations(self):
        item_id = random.choice(SAMPLE_ITEM_IDS)
        self.client.get(f"/recommendations/{item_id}")
