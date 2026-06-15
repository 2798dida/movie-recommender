import argparse

import joblib
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    ALS_MODEL_FILE,
    DEFAULT_RANDOM_STATE,
    INTERACTIONS_PROCESSED,
    MODELS_DIR,
    MOVIE_TFIDF_FILE,
    MOVIES_PROCESSED,
    TFIDF_VECTORIZER_FILE,
    USER_ITEM_FILE,
)


def _build_user_item(interactions: pd.DataFrame) -> csr_matrix:
    n_users = int(interactions["user_idx"].max()) + 1
    n_items = int(interactions["movie_idx"].max()) + 1
    values = interactions["rating"].astype("float32").to_numpy()
    rows = interactions["user_idx"].astype("int32").to_numpy()
    cols = interactions["movie_idx"].astype("int32").to_numpy()
    return csr_matrix((values, (rows, cols)), shape=(n_users, n_items), dtype="float32")


def train_models(
    factors: int,
    regularization: float,
    iterations: int,
    alpha: float,
    max_features: int,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    interactions = pd.read_parquet(INTERACTIONS_PROCESSED)
    movies = pd.read_parquet(MOVIES_PROCESSED)

    user_item = _build_user_item(interactions)

    als_model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=DEFAULT_RANDOM_STATE,
    )

    item_user_confidence = (user_item.T * alpha).astype("float32")
    als_model.fit(item_user_confidence)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
    )
    movie_tfidf = vectorizer.fit_transform(movies["content_text"].fillna(""))

    joblib.dump(als_model, ALS_MODEL_FILE)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_FILE)
    save_npz(USER_ITEM_FILE, user_item)
    save_npz(MOVIE_TFIDF_FILE, movie_tfidf)

    print("Training selesai.")
    print(f"ALS model       : {ALS_MODEL_FILE}")
    print(f"TF-IDF vectorizer: {TFIDF_VECTORIZER_FILE}")
    print(f"User-item matrix: {USER_ITEM_FILE}")
    print(f"Movie TF-IDF    : {MOVIE_TFIDF_FILE}")
    print(f"Shape user-item : {user_item.shape}")
    print(f"Shape TF-IDF    : {movie_tfidf.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid recommender artifacts.")
    parser.add_argument("--factors", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--max-features", type=int, default=50000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        alpha=args.alpha,
        max_features=args.max_features,
    )

