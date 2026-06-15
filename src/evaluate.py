import argparse

import numpy as np
import pandas as pd

from src.recommender import HybridMovieRecommender


def precision_at_k(recommended: set[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return len(recommended & relevant) / k


def recall_at_k(recommended: set[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    return len(recommended & relevant) / len(relevant)


def evaluate_sample(sample_users: int, k: int) -> None:
    recommender = HybridMovieRecommender()

    print("ALS users:", recommender.als_model.user_factors.shape[0])
    print("User-item rows:", recommender.user_item.shape[0])
    print("User map size:", len(recommender.user_id_to_idx))
    print("Max user_idx:", recommender.user_map["user_idx"].max())

    rng = np.random.default_rng(42)
    user_ids = recommender.user_map["userId"].to_numpy()
    if sample_users < len(user_ids):
        user_ids = rng.choice(user_ids, size=sample_users, replace=False)

    precisions = []
    recalls = []

    for user_id in user_ids:
        user_idx = recommender.user_id_to_idx[int(user_id)]
        max_users = recommender.als_model.user_factors.shape[0]
        if user_idx >= max_users:
            continue
        
        user_row = recommender.user_item[user_idx]
        positive_cf_indices = user_row.indices.astype(int)
        if len(positive_cf_indices) < 2:
            continue

        hidden_cf_idx = int(rng.choice(positive_cf_indices))
        hidden_movie_id = recommender.cf_idx_to_movie_id.get(hidden_cf_idx)
        if hidden_movie_id is None:
            continue

        masked_row = user_row.copy().tolil()
        masked_row[0, hidden_cf_idx] = 0
        masked_row = masked_row.tocsr()
        masked_row.eliminate_zeros()

        item_ids, _ = recommender.als_model.recommend(
            user_idx,
            masked_row,
            N=k,
            filter_already_liked_items=True,
        )
        recommended = {
            recommender.cf_idx_to_movie_id[int(movie_idx)]
            for movie_idx in item_ids
            if int(movie_idx) in recommender.cf_idx_to_movie_id
        }
        relevant = {int(hidden_movie_id)}

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant))

    print("Evaluasi sample selesai.")
    print("Catatan: model tetap sudah dilatih pada seluruh data, jadi angka ini optimistis.")
    print(f"Users evaluated: {len(precisions):,}")
    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}: {np.mean(recalls):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick recommender sanity evaluation.")
    parser.add_argument("--sample-users", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_sample(sample_users=args.sample_users, k=args.k)
