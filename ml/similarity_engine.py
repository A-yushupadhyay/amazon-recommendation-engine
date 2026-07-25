"""
similarity_engine.py

Phase 1: Builds an item-item collaborative filtering model using cosine
similarity on the user-item interaction matrix (Amazon's original 2003
"customers who bought this also bought" approach).

Evaluates against a popularity ("trending items") baseline using
precision@K and recall@K on held-out test data.

Usage:
    python ml/similarity_engine.py \
        --train data/processed/interactions_train.parquet \
        --test data/processed/interactions_test.parquet \
        --output data/processed/item_similarity.parquet \
        --top-n 20 \
        --k 10
"""

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_interaction_matrix(df: pd.DataFrame):
    """
    Build a sparse user-item interaction matrix from the interaction
    DataFrame. We use implicit feedback (1 = interacted) rather than raw
    rating values — this is more robust for CF and matches how Amazon's
    original item-item approach was framed (purchased/not purchased).

    Returns the sparse matrix plus the id<->index mappings needed to
    translate back and forth between real user_id/item_id and matrix
    row/column indices.
    """
    user_ids = df["user_id"].unique()
    item_ids = df["item_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}

    rows = df["user_id"].map(user_to_idx).values
    cols = df["item_id"].map(item_to_idx).values
    data = np.ones(len(df))  # implicit feedback: interacted = 1

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return matrix, user_to_idx, item_to_idx, idx_to_item


def compute_item_similarity(matrix: csr_matrix):
    """
    Compute item-item cosine similarity. We transpose so rows = items,
    columns = users, then compute pairwise cosine similarity between
    item vectors (each item's vector = which users interacted with it).

    This is the actual core of Amazon's original item-to-item CF paper.
    """
    item_user_matrix = matrix.T  # items x users
    similarity = cosine_similarity(item_user_matrix, dense_output=False)
    return similarity


def get_top_n_similar_items(similarity, idx_to_item, top_n: int = 20):
    """
    For each item, find its top-N most similar items (excluding itself).
    Returns a dict: item_id -> [(similar_item_id, score), ...]
    """
    similarity_dense = similarity.toarray()
    n_items = similarity_dense.shape[0]
    results = {}

    for i in range(n_items):
        scores = similarity_dense[i].copy()
        scores[i] = -1  # exclude self
        top_indices = np.argsort(scores)[::-1][:top_n]
        item_id = idx_to_item[i]
        results[item_id] = [
            (idx_to_item[j], float(scores[j]))
            for j in top_indices if scores[j] > 0
        ]
    return results


def recommend_for_item(item_similarities: dict, item_id, k: int = 10):
    """Return top-K recommended item_ids for a given item_id."""
    if item_id not in item_similarities:
        return []
    return [it for it, _ in item_similarities[item_id][:k]]


def popularity_baseline(train_df: pd.DataFrame, k: int = 10):
    """Simple 'trending items' baseline: most-interacted-with items overall."""
    top_items = train_df["item_id"].value_counts().head(k * 5).index.tolist()
    return top_items


def precision_recall_at_k(train_df: pd.DataFrame, test_df: pd.DataFrame, item_similarities: dict,
                           popularity_fallback: list, k: int = 10):
    """
    For each user, generate recommendations based on what they interacted
    with in TRAIN, then check how many of their held-out TEST items appear
    in the top-K recommendations. This mirrors real production usage: you
    only ever know a user's past behavior, never their future interactions,
    when generating a recommendation.

    (Earlier version of this function incorrectly used test items as the
    query too, which is circular — recommending based on the answer you're
    trying to predict. Fixed here to use train->test properly.)
    """
    cf_hits, cf_total_recommended, cf_total_relevant = 0, 0, 0
    pop_hits, pop_total_recommended, pop_total_relevant = 0, 0, 0

    train_grouped = train_df.groupby("user_id")["item_id"].apply(list)
    test_grouped = test_df.groupby("user_id")["item_id"].apply(list)

    # Only evaluate users who appear in both train and test
    common_users = set(train_grouped.index) & set(test_grouped.index)

    for user_id in common_users:
        query_items = train_grouped[user_id]
        actual_set = set(test_grouped[user_id])

        cf_candidates = {}
        for item in query_items:
            for sim_item, score in item_similarities.get(item, []):
                if sim_item in query_items:
                    continue  # don't recommend what they've already interacted with
                cf_candidates[sim_item] = cf_candidates.get(sim_item, 0) + score
        cf_recs = sorted(cf_candidates, key=cf_candidates.get, reverse=True)[:k]

        cf_hits += len(set(cf_recs) & actual_set)
        cf_total_recommended += len(cf_recs)
        cf_total_relevant += len(actual_set)

        pop_recs = [it for it in popularity_fallback if it not in query_items][:k]
        pop_hits += len(set(pop_recs) & actual_set)
        pop_total_recommended += len(pop_recs)
        pop_total_relevant += len(actual_set)

    cf_precision = cf_hits / cf_total_recommended if cf_total_recommended else 0
    cf_recall = cf_hits / cf_total_relevant if cf_total_relevant else 0
    pop_precision = pop_hits / pop_total_recommended if pop_total_recommended else 0
    pop_recall = pop_hits / pop_total_relevant if pop_total_relevant else 0

    return {
        "cf_precision_at_k": cf_precision,
        "cf_recall_at_k": cf_recall,
        "popularity_precision_at_k": pop_precision,
        "popularity_recall_at_k": pop_recall,
        "users_evaluated": len(common_users),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True, help="Where to save the item similarity table (parquet)")
    parser.add_argument("--top-n", type=int, default=20, help="How many similar items to store per item")
    parser.add_argument("--k", type=int, default=10, help="K for precision@K / recall@K evaluation")
    args = parser.parse_args()

    print("Loading train/test data...")
    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    print("Building interaction matrix...")
    matrix, user_to_idx, item_to_idx, idx_to_item = build_interaction_matrix(train_df)
    print(f"Matrix shape: {matrix.shape[0]} users x {matrix.shape[1]} items")

    print("Computing item-item cosine similarity (this is the ML core)...")
    similarity = compute_item_similarity(matrix)

    print(f"Extracting top-{args.top_n} similar items per item...")
    item_similarities = get_top_n_similar_items(similarity, idx_to_item, top_n=args.top_n)

    print("Computing popularity baseline...")
    pop_baseline = popularity_baseline(train_df, k=args.k)

    print(f"Evaluating precision@{args.k} / recall@{args.k} (CF vs. popularity baseline)...")
    metrics = precision_recall_at_k(train_df, test_df, item_similarities, pop_baseline, k=args.k)

    print("\n=== RESULTS ===")
    print(f"Users evaluated: {metrics['users_evaluated']}")
    print(f"Collaborative Filtering  -> precision@{args.k}: {metrics['cf_precision_at_k']:.4f} | recall@{args.k}: {metrics['cf_recall_at_k']:.4f}")
    print(f"Popularity Baseline      -> precision@{args.k}: {metrics['popularity_precision_at_k']:.4f} | recall@{args.k}: {metrics['popularity_recall_at_k']:.4f}")

    # Save the similarity table for later use in the serving API (Phase 2)
    rows = []
    for item_id, sims in item_similarities.items():
        for sim_item_id, score in sims:
            rows.append({"item_id": item_id, "similar_item_id": sim_item_id, "score": score})
    similarity_df = pd.DataFrame(rows)
    similarity_df.to_parquet(args.output, index=False)
    print(f"\nSaved item similarity table ({len(similarity_df)} rows) to {args.output}")


if __name__ == "__main__":
    main()
