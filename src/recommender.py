from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    ALS_MODEL_FILE,
    MOVIE_CF_MAP_FILE,
    MOVIE_TFIDF_FILE,
    MOVIES_PROCESSED,
    TFIDF_VECTORIZER_FILE,
    USER_ITEM_FILE,
    USER_MAP_FILE,
)


@dataclass(frozen=True)
class HybridWeights:
    als: float = 0.55
    content: float = 0.30
    popularity: float = 0.15


class HybridMovieRecommender:
    def __init__(self) -> None:
        self.movies = pd.read_parquet(MOVIES_PROCESSED).reset_index(drop=True)
        self.user_map = pd.read_parquet(USER_MAP_FILE)
        self.movie_cf_map = pd.read_parquet(MOVIE_CF_MAP_FILE)

        self.als_model = joblib.load(ALS_MODEL_FILE)
        self.vectorizer = joblib.load(TFIDF_VECTORIZER_FILE)
        self.user_item = load_npz(USER_ITEM_FILE).tocsr()
        self.movie_tfidf = load_npz(MOVIE_TFIDF_FILE).tocsr()

        self.movies_by_id = self.movies.set_index("movieId", drop=False)
        self.user_id_to_idx = dict(
            zip(self.user_map["userId"].astype(int), self.user_map["user_idx"].astype(int))
        )
        self.cf_idx_to_movie_id = dict(
            zip(
                self.movie_cf_map["movie_idx"].astype(int),
                self.movie_cf_map["movieId"].astype(int),
            )
        )
        self.movie_id_to_content_idx = dict(
            zip(self.movies["movieId"].astype(int), self.movies["content_idx"].astype(int))
        )

    @staticmethod
    def artifacts_exist() -> bool:
        required = [
            MOVIES_PROCESSED,
            USER_MAP_FILE,
            MOVIE_CF_MAP_FILE,
            ALS_MODEL_FILE,
            TFIDF_VECTORIZER_FILE,
            USER_ITEM_FILE,
            MOVIE_TFIDF_FILE,
        ]
        return all(Path(path).exists() for path in required)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype="float32")
        if values.size == 0:
            return values
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value == min_value:
            return np.ones_like(values, dtype="float32")
        return (values - min_value) / (max_value - min_value)

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> np.ndarray:
        pool = min(max(int(limit), 0), scores.size)
        if pool == 0:
            return np.asarray([], dtype="int64")
        if pool == scores.size:
            return np.argsort(scores)[::-1]
        top_indices = np.argpartition(scores, -pool)[-pool:]
        return top_indices[np.argsort(scores[top_indices])[::-1]]

    def _format_results(
        self,
        scored_movie_ids: list[tuple[int, float]],
        n: int,
        source: str,
    ) -> pd.DataFrame:
        rows = []
        for movie_id, score in scored_movie_ids[:n]:
            if movie_id not in self.movies_by_id.index:
                continue
            row = self.movies_by_id.loc[movie_id]
            rows.append(
                {
                    "movieId": int(row["movieId"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "mean_rating": round(float(row["mean_rating"]), 3),
                    "rating_count": int(row["rating_count"]),
                    "score": round(float(score), 6),
                    "source": source,
                }
            )
        return pd.DataFrame(rows)

    def popular_movies(self, n: int = 10) -> pd.DataFrame:
        popular = (
            self.movies.sort_values(
                ["popularity_score", "rating_count", "mean_rating"],
                ascending=False,
            )
            .head(n)
            .copy()
        )
        popular["score"] = popular["popularity_score"]
        popular["source"] = "popular"
        return popular[
            ["movieId", "title", "genres", "mean_rating", "rating_count", "score", "source"]
        ]

    def search_movies(self, query: str, limit: int = 20) -> pd.DataFrame:
        query = query.strip().lower()
        if not query:
            return pd.DataFrame(columns=["movieId", "title", "genres"])

        mask = self.movies["title"].str.lower().str.contains(query, regex=False, na=False)
        return (
            self.movies.loc[mask]
            .sort_values(["rating_count", "mean_rating"], ascending=False)
            .head(limit)[["movieId", "title", "genres"]]
            .reset_index(drop=True)
        )

    from scipy.sparse import csr_matrix

    def _content_scores_from_movie_ids(
        self,
        liked_movie_ids: list[int],
        seen_movie_ids: set[int],
        candidate_pool: int,
    ) -> list[tuple[int, float]]:
        content_indices = [
            self.movie_id_to_content_idx[movie_id]
            for movie_id in liked_movie_ids
            if movie_id in self.movie_id_to_content_idx
        ]

        if not content_indices:
            return []

        profile = self.movie_tfidf[content_indices].mean(axis=0)

        print(type(profile))
        print(profile.shape)
        
        # convert numpy.matrix -> numpy.ndarray
        profile = np.asarray(profile)
        print(type(profile))
        print(profile.shape)

        # ensure shape = (1, n_features)
        profile = profile.reshape(1, -1)

        scores = cosine_similarity(
            profile,
            self.movie_tfidf
        ).ravel()

        top_indices = self._top_indices(scores, candidate_pool)

        output = []
        for content_idx in top_indices:
            movie_id = int(self.movies.iloc[int(content_idx)]["movieId"])
            if movie_id not in seen_movie_ids:
                output.append((movie_id, float(scores[content_idx])))

        return output

    def _content_scores_from_text(
        self,
        text: str,
        seen_movie_ids: set[int],
        candidate_pool: int,
    ) -> list[tuple[int, float]]:
        text = text.strip()
        if not text:
            return []
        query_vector = self.vectorizer.transform([text.lower()])
        scores = cosine_similarity(query_vector, self.movie_tfidf).ravel()
        top_indices = self._top_indices(scores, candidate_pool)

        output = []
        for content_idx in top_indices:
            movie_id = int(self.movies.iloc[int(content_idx)]["movieId"])
            if movie_id not in seen_movie_ids and scores[content_idx] > 0:
                output.append((movie_id, float(scores[content_idx])))
        return output

    def recommend_for_user_id(
        self,
        user_id: int,
        n: int = 10,
        weights: HybridWeights = HybridWeights(),
    ) -> pd.DataFrame:
        if user_id not in self.user_id_to_idx:
            return self.popular_movies(n=n)

        user_idx = self.user_id_to_idx[user_id]
        user_row = self.user_item[user_idx]
        seen_cf_indices = set(user_row.indices.astype(int))
        seen_movie_ids = {
            self.cf_idx_to_movie_id[idx]
            for idx in seen_cf_indices
            if idx in self.cf_idx_to_movie_id
        }
        candidate_pool = max(n * 30, 300)
        als_pool = min(candidate_pool, max(self.user_item.shape[1] - 1, 1))

        als_item_ids, als_scores = self.als_model.recommend(
            user_idx,
            user_row,
            N=als_pool,
            filter_already_liked_items=True,
        )
        als_movie_ids = [
            self.cf_idx_to_movie_id[int(movie_idx)]
            for movie_idx in als_item_ids
            if int(movie_idx) in self.cf_idx_to_movie_id
        ]

        liked_movie_ids = list(seen_movie_ids)
        content_candidates = self._content_scores_from_movie_ids(
            liked_movie_ids=liked_movie_ids,
            seen_movie_ids=seen_movie_ids,
            candidate_pool=candidate_pool,
        )

        popular_candidates = [
            (int(row["movieId"]), float(row["popularity_score"]))
            for _, row in self.movies.sort_values("popularity_score", ascending=False)
            .head(candidate_pool)
            .iterrows()
            if int(row["movieId"]) not in seen_movie_ids
        ]

        combined = defaultdict(float)

        als_norm = self._normalize(np.asarray(als_scores[: len(als_movie_ids)]))
        for movie_id, score in zip(als_movie_ids, als_norm):
            combined[movie_id] += weights.als * float(score)

        if content_candidates:
            movie_ids, scores = zip(*content_candidates)
            for movie_id, score in zip(movie_ids, self._normalize(np.asarray(scores))):
                combined[movie_id] += weights.content * float(score)

        if popular_candidates:
            movie_ids, scores = zip(*popular_candidates)
            for movie_id, score in zip(movie_ids, self._normalize(np.asarray(scores))):
                combined[movie_id] += weights.popularity * float(score)

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        return self._format_results(ranked, n=n, source="hybrid")

    def recommend_cold_start(
        self,
        selected_genres: list[str] | None = None,
        liked_movie_ids: list[int] | None = None,
        free_text: str = "",
        n: int = 10,
    ) -> pd.DataFrame:
        selected_genres = selected_genres or []
        liked_movie_ids = liked_movie_ids or []
        seen_movie_ids = set(int(movie_id) for movie_id in liked_movie_ids)
        candidate_pool = max(n * 30, 300)

        liked_titles = []
        for movie_id in liked_movie_ids:
            if movie_id in self.movies_by_id.index:
                row = self.movies_by_id.loc[movie_id]
                liked_titles.append(f"{row['title']} {row['genres']}")

        query_text = " ".join(selected_genres + liked_titles + [free_text])

        content_candidates = self._content_scores_from_text(
            text=query_text,
            seen_movie_ids=seen_movie_ids,
            candidate_pool=candidate_pool,
        )
        if not content_candidates and liked_movie_ids:
            content_candidates = self._content_scores_from_movie_ids(
                liked_movie_ids=liked_movie_ids,
                seen_movie_ids=seen_movie_ids,
                candidate_pool=candidate_pool,
            )

        popular_candidates = [
            (int(row["movieId"]), float(row["popularity_score"]))
            for _, row in self.movies.sort_values("popularity_score", ascending=False)
            .head(candidate_pool)
            .iterrows()
            if int(row["movieId"]) not in seen_movie_ids
        ]

        combined = defaultdict(float)

        if content_candidates:
            movie_ids, scores = zip(*content_candidates)
            for movie_id, score in zip(movie_ids, self._normalize(np.asarray(scores))):
                combined[movie_id] += 0.75 * float(score)

        if popular_candidates:
            movie_ids, scores = zip(*popular_candidates)
            for movie_id, score in zip(movie_ids, self._normalize(np.asarray(scores))):
                combined[movie_id] += 0.25 * float(score)

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        return self._format_results(ranked, n=n, source="cold_start")

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        if movie_id not in self.movie_id_to_content_idx:
            return pd.DataFrame()

        content_idx = self.movie_id_to_content_idx[movie_id]
        query_vector = self.movie_tfidf[content_idx]
        scores = cosine_similarity(query_vector, self.movie_tfidf).ravel()
        top_indices = self._top_indices(scores, n + 1)

        ranked = []
        for idx in top_indices:
            candidate_id = int(self.movies.iloc[int(idx)]["movieId"])
            if candidate_id != movie_id:
                ranked.append((candidate_id, float(scores[idx])))

        return self._format_results(ranked, n=n, source="content_similar")
