import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_MIN_RATING,
    INTERACTIONS_PROCESSED,
    MOVIE_CF_MAP_FILE,
    MOVIES_FILE,
    MOVIES_PROCESSED,
    PROCESSED_DIR,
    RATINGS_FILE,
    TAGS_FILE,
    USER_MAP_FILE,
)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Pastikan MovieLens 32M sudah diekstrak ke data/raw/ml-32m/."
        )


def _read_ratings(max_ratings: int | None) -> pd.DataFrame:
    usecols = ["userId", "movieId", "rating", "timestamp"]
    dtype = {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int64",
    }
    return pd.read_csv(RATINGS_FILE, usecols=usecols, dtype=dtype, nrows=max_ratings)


def _build_tag_text() -> pd.DataFrame:
    if not TAGS_FILE.exists():
        return pd.DataFrame(columns=["movieId", "tags_text"])

    tags = pd.read_csv(
        TAGS_FILE,
        usecols=["movieId", "tag"],
        dtype={"movieId": "int32", "tag": "string"},
    )
    tags["tag"] = (
        tags["tag"]
        .fillna("")
        .str.lower()
        .str.replace("|", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    tags = tags[tags["tag"] != ""]

    def join_limited(values: pd.Series) -> str:
        unique_values = pd.unique(values)[:80]
        return " ".join(str(value) for value in unique_values)

    return tags.groupby("movieId", as_index=False)["tag"].agg(join_limited).rename(
        columns={"tag": "tags_text"}
    )


def _normalize(series: pd.Series) -> pd.Series:
    series = series.fillna(0).astype("float32")
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value == min_value:
        return pd.Series(np.ones(len(series), dtype="float32"), index=series.index)
    return ((series - min_value) / (max_value - min_value)).astype("float32")


def prepare_data(min_rating: float, max_ratings: int | None) -> None:
    _require_file(RATINGS_FILE)
    _require_file(MOVIES_FILE)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ratings = _read_ratings(max_ratings=max_ratings)
    movies = pd.read_csv(
        MOVIES_FILE,
        usecols=["movieId", "title", "genres"],
        dtype={"movieId": "int32", "title": "string", "genres": "string"},
    )

    rating_stats = (
        ratings.groupby("movieId")
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
        .reset_index()
    )
    rating_stats["rating_count"] = rating_stats["rating_count"].astype("int32")
    rating_stats["mean_rating"] = rating_stats["mean_rating"].astype("float32")

    tags_text = _build_tag_text()

    movies = movies.merge(rating_stats, on="movieId", how="left")
    movies = movies.merge(tags_text, on="movieId", how="left")
    movies["rating_count"] = movies["rating_count"].fillna(0).astype("int32")
    movies["mean_rating"] = movies["mean_rating"].fillna(0).astype("float32")
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    movies["tags_text"] = movies["tags_text"].fillna("")

    popularity_raw = np.log1p(movies["rating_count"]) * (movies["mean_rating"] / 5.0)
    movies["popularity_score"] = _normalize(popularity_raw)
    movies["content_text"] = (
        movies["title"].fillna("")
        + " "
        + movies["genres"].str.replace("|", " ", regex=False)
        + " "
        + movies["tags_text"]
    ).str.lower()
    movies = movies.sort_values("movieId").reset_index(drop=True)
    movies["content_idx"] = movies.index.astype("int32")

    positive = ratings[ratings["rating"] >= min_rating].copy()
    positive = positive[positive["movieId"].isin(movies["movieId"])]

    user_map = pd.DataFrame({"userId": np.sort(positive["userId"].unique())})
    user_map["user_idx"] = np.arange(len(user_map), dtype="int32")

    movie_cf_map = pd.DataFrame({"movieId": np.sort(positive["movieId"].unique())})
    movie_cf_map["movie_idx"] = np.arange(len(movie_cf_map), dtype="int32")

    positive = positive.merge(user_map, on="userId", how="inner")
    positive = positive.merge(movie_cf_map, on="movieId", how="inner")
    positive = positive[
        ["userId", "movieId", "user_idx", "movie_idx", "rating", "timestamp"]
    ].sort_values(["user_idx", "timestamp"])

    movies.to_parquet(MOVIES_PROCESSED, index=False)
    positive.to_parquet(INTERACTIONS_PROCESSED, index=False)
    user_map.to_parquet(USER_MAP_FILE, index=False)
    movie_cf_map.to_parquet(MOVIE_CF_MAP_FILE, index=False)

    print("Preprocessing selesai.")
    print(f"Movies              : {len(movies):,}")
    print(f"Positive ratings    : {len(positive):,}")
    print(f"Users with positives: {len(user_map):,}")
    print(f"Movies for CF       : {len(movie_cf_map):,}")
    print(f"Output folder       : {PROCESSED_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens 32M.")
    parser.add_argument(
        "--min-rating",
        type=float,
        default=DEFAULT_MIN_RATING,
        help="Rating minimum yang dianggap interaksi positif.",
    )
    parser.add_argument(
        "--max-ratings",
        type=int,
        default=None,
        help="Batasi jumlah baris ratings.csv untuk eksperimen cepat.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(min_rating=args.min_rating, max_ratings=args.max_ratings)

