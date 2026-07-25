"""
data_loader.py

Phase 0/1: Loads the raw Amazon Reviews dataset and turns it into a clean
user-item interaction table we can build collaborative filtering on top of.

Usage:
    python ml/data_loader.py --input data/raw/reviews.jsonl --output data/processed/interactions.parquet
"""

import argparse
import json
import pandas as pd


def load_raw_reviews(path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load the raw Amazon Reviews JSONL file, streaming line-by-line instead
    of loading the whole file into memory at once (pd.read_json with
    lines=True reads the entire file as one string first, which caused a
    MemoryError on larger files).

    Also maps this dataset's actual field names to what the rest of the
    pipeline expects:
        asin       -> item_id
        user_id    -> user_id (already matches)
        rating     -> rating
        timestamp  -> timestamp
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append({
                "user_id": row.get("user_id"),
                "item_id": row.get("asin"),
                "rating": row.get("rating"),
                "timestamp": row.get("timestamp"),
            })
    df = pd.DataFrame(records)
    return df


def clean_interactions(df: pd.DataFrame, min_user_interactions: int = 5,
                        min_item_interactions: int = 5) -> pd.DataFrame:
    """
    Filter out sparse users/items. This is a real production step, not
    just cleanup: sparse users/items poison similarity scores because
    there's not enough signal to compute meaningful similarity.
    """
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()

    valid_users = user_counts[user_counts >= min_user_interactions].index
    valid_items = item_counts[item_counts >= min_item_interactions].index

    filtered = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
    return filtered.reset_index(drop=True)


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Split by timestamp, NOT randomly. A random split leaks future
    interactions into training, which would never happen in real
    production (you can't train on data from the future).
    """
    df_sorted = df.sort_values("timestamp")
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw reviews file")
    parser.add_argument("--output", required=True, help="Path to write processed parquet file")
    parser.add_argument("--min-user", type=int, default=5)
    parser.add_argument("--min-item", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Optional cap on rows read, useful for fast local iteration on large files")
    args = parser.parse_args()

    print(f"Loading raw data from {args.input} ...")
    df = load_raw_reviews(args.input, max_rows=args.max_rows)
    print(f"Loaded {len(df)} raw interactions")

    df_clean = clean_interactions(df, args.min_user, args.min_item)
    print(f"Kept {len(df_clean)} interactions after filtering sparse users/items")

    train, test = time_based_split(df_clean)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")

    train.to_parquet(args.output.replace(".parquet", "_train.parquet"), index=False)
    test.to_parquet(args.output.replace(".parquet", "_test.parquet"), index=False)
    print("Done. Saved train/test parquet files.")


if __name__ == "__main__":
    main()