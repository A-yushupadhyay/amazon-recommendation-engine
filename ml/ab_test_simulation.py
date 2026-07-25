"""
ab_test_simulation.py

Phase 3: Simulates an A/B test comparing the CF model (Variant A) against
the popularity baseline (Variant B), the same way Amazon's internal
Weblab experimentation platform assigns users to variants.

- Users are deterministically assigned to a variant via consistent
  hashing on user_id (same user always sees the same variant -- this
  is real production behavior, not random-per-request).
- A "click" is simulated when a recommended item appears in the user's
  held-out TEST interactions (their actual future behavior).
- Computes CTR (click-through rate) per variant.
- Runs a two-proportion z-test for statistical significance.

This is the piece that demonstrates the difference between OFFLINE
metrics (precision@10 / recall@10 from Phase 1) and ONLINE / BUSINESS
metrics (CTR) -- a distinction real ML systems in production care about
a lot more than offline accuracy alone.

Usage:
    python ml/ab_test_simulation.py \
        --train data/processed/interactions_train.parquet \
        --test data/processed/interactions_test.parquet \
        --similarity data/processed/item_similarity.parquet \
        --k 10
"""

import argparse
import hashlib
import math
import pandas as pd


def assign_variant(user_id: str) -> str:
    """
    Deterministic variant assignment via consistent hashing, so the same
    user always lands in the same variant across the whole 'experiment'
    (mirrors real experimentation platforms like Amazon Weblab).
    """
    h = hashlib.md5(user_id.encode()).hexdigest()
    return "A" if int(h, 16) % 2 == 0 else "B"


def load_item_similarities(similarity_df: pd.DataFrame) -> dict:
    """Rebuild the item_id -> [(similar_item_id, score), ...] lookup dict."""
    result = {}
    for item_id, group in similarity_df.groupby("item_id"):
        result[item_id] = list(zip(group["similar_item_id"], group["score"]))
    return result


def recommend_cf(query_items: list, item_similarities: dict, k: int) -> list:
    candidates = {}
    for item in query_items:
        for sim_item, score in item_similarities.get(item, []):
            if sim_item in query_items:
                continue
            candidates[sim_item] = candidates.get(sim_item, 0) + score
    return sorted(candidates, key=candidates.get, reverse=True)[:k]


def recommend_popularity(popularity_list: list, query_items: list, k: int) -> list:
    return [it for it in popularity_list if it not in query_items][:k]


def two_proportion_z_test(clicks_a, impressions_a, clicks_b, impressions_b):
    """
    Standard two-proportion z-test. Returns (z_score, p_value, ctr_a, ctr_b).
    Manual implementation (no statsmodels dependency needed) using the
    normal approximation, which is standard for this kind of CTR test at
    reasonable sample sizes.
    """
    p_a = clicks_a / impressions_a if impressions_a else 0
    p_b = clicks_b / impressions_b if impressions_b else 0
    p_pool = (clicks_a + clicks_b) / (impressions_a + impressions_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / impressions_a + 1 / impressions_b))
    if se == 0:
        return 0.0, 1.0, p_a, p_b

    z = (p_a - p_b) / se
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return z, p_value, p_a, p_b


def required_sample_size(p_baseline: float, relative_lift: float, power: float = 0.8, alpha: float = 0.05):
    """
    Calculates the required sample size PER VARIANT to detect a given
    relative lift over a baseline conversion rate, at the given
    statistical power and significance level. Standard two-proportion
    sample size formula, using the normal approximation (same approach
    tools like Optimizely/Amazon Weblab use under the hood).
    """
    p1 = p_baseline
    p2 = p_baseline * (1 + relative_lift)

    z_alpha = _inverse_normal_cdf(1 - alpha / 2)
    z_beta = _inverse_normal_cdf(power)

    p_avg = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2

    if denominator == 0:
        return float("inf")
    return math.ceil(numerator / denominator)


def _inverse_normal_cdf(p, tol=1e-10):
    """
    Approximates the inverse standard normal CDF (quantile function) via
    binary search over the CDF -- avoids needing scipy.stats.norm.ppf.
    """
    lo, hi = -10.0, 10.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _normal_cdf(x):
    """Standard normal CDF via the error function (math.erf), avoids a scipy dependency."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--similarity", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print("Loading data...")
    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    similarity_df = pd.read_parquet(args.similarity)

    item_similarities = load_item_similarities(similarity_df)
    popularity_list = train_df["item_id"].value_counts().head(args.k * 5).index.tolist()

    train_grouped = train_df.groupby("user_id")["item_id"].apply(list)
    test_grouped = test_df.groupby("user_id")["item_id"].apply(list)
    common_users = sorted(set(train_grouped.index) & set(test_grouped.index))

    print(f"Simulating A/B test across {len(common_users)} users...")

    clicks_a, impressions_a = 0, 0
    clicks_b, impressions_b = 0, 0

    for user_id in common_users:
        variant = assign_variant(user_id)
        query_items = train_grouped[user_id]
        actual_set = set(test_grouped[user_id])

        if variant == "A":
            recs = recommend_cf(query_items, item_similarities, args.k)
            impressions_a += 1
            if set(recs) & actual_set:
                clicks_a += 1
        else:
            recs = recommend_popularity(popularity_list, query_items, args.k)
            impressions_b += 1
            if set(recs) & actual_set:
                clicks_b += 1

    z, p_value, ctr_a, ctr_b = two_proportion_z_test(clicks_a, impressions_a, clicks_b, impressions_b)

    print("\n=== A/B TEST RESULTS ===")
    print(f"Variant A (Collaborative Filtering): {impressions_a} users, {clicks_a} clicks, CTR = {ctr_a:.4f} ({ctr_a*100:.2f}%)")
    print(f"Variant B (Popularity Baseline):     {impressions_b} users, {clicks_b} clicks, CTR = {ctr_b:.4f} ({ctr_b*100:.2f}%)")

    if ctr_b > 0:
        lift = (ctr_a - ctr_b) / ctr_b * 100
        print(f"\nRelative lift (A vs B): {lift:+.1f}%")

    print(f"\nZ-score: {z:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: STATISTICALLY SIGNIFICANT at p < 0.05")
    else:
        print("Result: NOT statistically significant at p < 0.05 (could be due to sample size)")

    if ctr_b > 0 and ctr_a != ctr_b:
        observed_relative_lift = (ctr_a - ctr_b) / ctr_b
        needed_n = required_sample_size(ctr_b, observed_relative_lift)
        print(f"\n=== POWER ANALYSIS ===")
        print(f"Observed relative lift: {observed_relative_lift*100:+.2f}%")
        print(f"Users per variant needed to detect this lift at 80% power: ~{needed_n:,}")
        print(f"Users per variant actually observed: A={impressions_a:,}, B={impressions_b:,}")
        if needed_n > max(impressions_a, impressions_b):
            print("Conclusion: current sample size is UNDERPOWERED to detect this effect reliably.")
        else:
            print("Conclusion: current sample size should be sufficient to detect this effect.")


if __name__ == "__main__":
    main()
