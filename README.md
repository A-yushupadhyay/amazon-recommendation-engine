# Recommendation Engine — Phase 0 Setup

Follow these steps in order. Each one is a checkpoint — don't move to the next until the current one works.

## 1. Install prerequisites (if you don't already have them)

- **Python 3.10+**: `python3 --version` to check
- **Docker Desktop**: needed to run Postgres, Redis, and DynamoDB Local without installing each one manually
- **Git**: to version this project (recommended, not strictly required)

## 2. Get this project onto your machine

Download all the files from this conversation into a folder called `recsys/` on your machine, keeping this structure:

```
recsys/
  docker-compose.yml
  requirements.txt
  .env.example
  .gitignore
  data/
    raw/
    processed/
  ml/
    data_loader.py
  api/
  tests/
  benchmarks/
  README.md
```

## 3. Create a Python virtual environment

```bash
cd recsys
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

You'll know it worked because your terminal prompt will show `(venv)` at the start.

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, pandas, scikit-learn, boto3 (for DynamoDB), redis, psycopg2, etc. — everything for the whole project, not just Phase 0. Installing it all now saves you from re-running this at every phase.

## 5. Set up your environment file

```bash
cp .env.example .env
```

Leave the values as-is for now — they're already configured to match the Docker Compose services in step 6.

## 6. Start Postgres, Redis, and DynamoDB Local

```bash
docker compose up -d
```

This starts all three services in the background. Verify they're running:

```bash
docker compose ps
```

You should see three containers: `recsys-postgres`, `recsys-redis`, `recsys-dynamodb`, all with status "Up".

**If this fails:** most common cause is a port already in use (5432, 6379, or 8000 taken by something else on your machine). Tell me the exact error and I'll help you fix it.

## 7. Download the dataset

Go to the Amazon Reviews dataset (2023 version) hosted here: https://amazon-reviews-2023.github.io/

- Pick **one category** to start — "Electronics" or "Books" work well (not too huge, not too sparse)
- Download the **"5-core"** version if available — this is Amazon's own pre-filtered version (users/items with 5+ reviews), which saves you a filtering step later
- Save the file into `recsys/data/raw/reviews.jsonl` (or `.csv`, whichever format you get)

Alternative if that site is slow/unavailable: search Kaggle for "Amazon Reviews 2023" — there are mirrored versions there too.

## 8. Run the data loader to verify everything works end-to-end

```bash
python ml/data_loader.py --input data/raw/reviews.jsonl --output data/processed/interactions.parquet
```

You should see output like:
```
Loading raw data from data/raw/reviews.jsonl ...
Loaded 234567 raw interactions
Kept 198234 interactions after filtering sparse users/items
Train: 158587 rows | Test: 39647 rows
Done. Saved train/test parquet files.
```

If you see this, **Phase 0 is complete** — you have a working environment, all three databases running, and clean train/test data ready for Phase 1 (building the actual collaborative filtering model).

## Troubleshooting checklist

- `docker compose up -d` fails → check Docker Desktop is actually running (not just installed)
- `pip install` fails on `psycopg2-binary` → on Mac you may need `brew install postgresql` first for build headers
- Data loader errors on column names → your dataset's column names might differ slightly (e.g. `asin` instead of `item_id`, `reviewerID` instead of `user_id`) — check the first few rows with `df.head()` and tell me the actual column names, I'll adjust the script
