# Amazon-Style Recommendation Engine
### Item-Item Collaborative Filtering with a Production-Style Serving Layer

A recommendation engine built end-to-end — from the ML model to a low-latency serving API — using the same core algorithm Amazon pioneered in 2003 for "customers who bought this also bought."

This isn't a notebook demo. It's a working system with a trained model, a real database-backed serving layer, verified fault tolerance, and measured performance under load.

---

## What this project demonstrates

| Area | What was built |
|---|---|
| **Machine Learning** | Item-item collaborative filtering, evaluated against a baseline with real metrics |
| **Backend Systems** | FastAPI + DynamoDB + Redis, with caching, fallback, and circuit-breaker logic |
| **Distributed Systems Thinking** | Offline/online split, graceful degradation, verified under simulated failure |
| **Data Science Rigor** | A/B test simulation with a proper significance test and power analysis |
| **Software Engineering** | Clean architecture — Strategy pattern, Single Responsibility, Dependency Injection |

---

## The Problem

When a customer looks at a product, what else should we show them? Amazon solved this in 2003 with **item-item collaborative filtering**: instead of comparing users to each other (which changes constantly and doesn't scale), compare items to each other based on which users interacted with both. Item relationships are far more stable than user behavior, which is why this approach is still foundational today.

This project builds that algorithm from scratch, then wraps it in the kind of serving infrastructure a real product would need: caching, a fast database lookup, and fallback behavior for new users or system failures.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Raw Review Data    │
                    │   (Amazon dataset)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   OFFLINE PIPELINE    │
                    │  Clean → Split →      │
                    │  Train CF Model       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Item Similarity      │
                    │  Table (top-20/item)  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │      DynamoDB         │
                    │ (single-key lookup)   │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                ONLINE SERVING PATH            │
        │                                                │
    Request → Redis Cache? ──Yes──> Return (fast)        │
        │         │No                                    │
        │         ▼                                       │
        │    DynamoDB lookup ──Found──> Cache + Return    │
        │         │Not found / DynamoDB down               │
        │         ▼                                       │
        │    Trending Items Fallback ──> Return           │
        └────────────────────────────────────────────────┘
```

**Why this split matters:** training a model is slow and can happen on a schedule (nightly batch job). Serving a recommendation needs to happen in milliseconds. Keeping these two concerns separate — precompute offline, serve fast online — is the core pattern behind most real recommendation systems, not just Amazon's.

---

## Results

### 1. Model Quality (Offline Evaluation)

Item-item collaborative filtering vs. a "trending items" popularity baseline, measured with **precision@10** and **recall@10** on held-out test data (properly time-split — trained only on past interactions, evaluated against future ones, avoiding lookahead bias):

| Model | Precision@10 | Recall@10 |
|---|---|---|
| **Collaborative Filtering** | **0.0295** | **0.0605** |
| Popularity Baseline | 0.0263 | 0.0539 |
| **Improvement** | **+12.2%** | **+12.2%** |

### 2. Serving Performance (Load Test)

Tested with [Locust](https://locust.io) at 20 concurrent users:

| Metric | Value |
|---|---|
| Median latency (p50) | 10 ms |
| p95 latency | 20-28 ms |
| p99 latency | 27-31 ms |
| Throughput | 63.6 requests/sec |
| Failure rate | 0% |

**A real bug found and fixed along the way:** the first load test showed p99 latency spiking to **4,800ms** despite a healthy 10ms median — a classic thread-pool starvation issue where FastAPI's default synchronous request handling queued requests under concurrent load. Fixing the concurrency configuration brought p99 down to 28ms — a **~170x improvement**. This kind of gap between median and tail latency is a common real production issue, and diagnosing it correctly (rather than just seeing "average looks fine") is what this project practiced.

### 3. Fault Tolerance (Verified, Not Just Designed)

Both failure modes below were tested by actually stopping the dependency mid-run, not just written as theoretical code paths:

| Scenario | Behavior |
|---|---|
| Redis unavailable | Skips cache, serves directly from DynamoDB — no error to the user |
| DynamoDB unavailable | Falls back to trending items after 3 failed attempts, then stops retrying for a cooldown period (circuit breaker) — protects the system from hammering a dead dependency |
| Item not in the model (cold start) | Falls back to trending items automatically |

### 4. A/B Test Simulation — Offline vs. Online Metrics

Simulated a real A/B test using consistent-hash variant assignment (the same technique Amazon's internal Weblab experimentation platform uses — same user always sees the same variant):

| Variant | Users | CTR |
|---|---|---|
| A — Collaborative Filtering | 1,462 | 18.33% |
| B — Popularity Baseline | 1,444 | 18.07% |
| **Relative lift** | | **+1.4%** |

Ran a two-proportion z-test: **p = 0.858 (not statistically significant)**. Rather than stopping there, a power analysis was run to quantify *why*: detecting a lift this small reliably would require **~356,000 users per variant** — far beyond this project's ~1,450/variant sample. This distinction matters: the **offline metric (precision@10) showed a clear 12% improvement**, while the **online metric (CTR) showed a smaller, statistically inconclusive lift** — a realistic illustration of why production teams track both, and why small effects often need much larger experiments than intuition suggests.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ML | Python, pandas, scikit-learn, scipy (sparse matrices) | Standard, memory-efficient for sparse interaction data |
| Online store | DynamoDB (Local) | Single-key lookup pattern (`item_id` → similar items) is exactly what DynamoDB is built for |
| Cache | Redis | Sub-millisecond reads for hot items |
| Analytical store | PostgreSQL | Relational data (interaction history, evaluation logs) that DynamoDB isn't suited for |
| API | FastAPI | Async-capable Python web framework |
| Frontend | Plain HTML/JS | No build tooling needed for a demo — deliberately simple |

**Why two databases, not one:** Postgres handles offline, relational work — training data, evaluation logs, anything you'd query with joins or aggregations. DynamoDB handles the online hot path — a single-key lookup that needs to be fast and predictable under load. Using the right store for each access pattern, instead of forcing one database to do both jobs, was a deliberate design decision.

---

## Code Architecture

The serving layer is split into focused modules rather than one large file:

```
storage/
  cache.py                 # Redis — knows nothing about DynamoDB or recommendations
  dynamo_repository.py     # DynamoDB — knows nothing about caching or recommendations
recommendation/
  strategies.py            # Strategy pattern: CollaborativeFilteringStrategy, TrendingFallbackStrategy
  service.py                # Orchestrates cache -> strategies via dependency injection
api/
  main.py                   # Routing only — wires dependencies together at startup
```

- **Single Responsibility** — each module owns exactly one concern (Redis, DynamoDB, or orchestration)
- **Strategy pattern** — `CollaborativeFilteringStrategy` and `TrendingFallbackStrategy` implement the same interface, swappable without touching the calling code
- **Dependency Injection** — `RecommendationService` receives its cache and strategies as constructor arguments rather than constructing them itself, making it testable with fake dependencies

---

## What I'd Do at a Larger Scale

Being explicit about this project's boundaries, and how I'd address them with more scale:

- **Similarity computation**: currently brute-force cosine similarity (fine for ~5K items). At millions of items, I'd use approximate nearest neighbor search (FAISS or pgvector) instead of a full pairwise matrix.
- **User-level recommendations**: currently a simplified fallback. A full version would look up a user's recent interaction history (from Postgres) and feed it through the same CF strategy.
- **DynamoDB access patterns**: currently a single partition-key lookup (`item_id`). A production version serving user-level recommendations too would add a Global Secondary Index or a second table, designed access-pattern-first (the standard DynamoDB design approach).
- **Async I/O**: the current API uses synchronous Redis/DynamoDB calls running in a thread pool. A fully async rewrite (`redis.asyncio`, `aioboto3`) would remove the thread-pool bottleneck entirely rather than just tuning around it.
- **Retraining**: currently a manual batch job. In production this would run on a schedule (e.g. nightly via Airflow or a cron job), with monitoring for model staleness.

---

## Running This Locally

**Prerequisites:** Docker, Python 3.10+

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Process the data (raw reviews -> clean train/test split)
python ml/data_loader.py --input data/raw/reviews.jsonl --output data/processed/interactions.parquet

# 4. Train the model and evaluate against baseline
python ml/similarity_engine.py --train data/processed/interactions_train.parquet --test data/processed/interactions_test.parquet --output data/processed/item_similarity.parquet

# 5. Load the model into DynamoDB
python ml/dynamo_writer.py --input data/processed/item_similarity.parquet

# 6. Run the A/B test simulation
python ml/ab_test_simulation.py --train data/processed/interactions_train.parquet --test data/processed/interactions_test.parquet --similarity data/processed/item_similarity.parquet

# 7. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8001 --limit-concurrency 100

# 8. Open frontend/index.html in a browser for the interactive demo
```

**Note:** DynamoDB Local runs in-memory for this project — data is cleared on restart. Re-run step 5 after any `docker compose restart`.

---

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — Software category. Sparse users/items (fewer than 10 interactions) filtered out before training. Split by timestamp (not randomly) to avoid training on future data relative to the test set.
